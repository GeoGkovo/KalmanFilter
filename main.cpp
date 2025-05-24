#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <random>

#include "kalman_filter.hpp"
#include "utils.hpp"
#include "measurement_container.hpp"
#include "matrix_eval.hpp"

/*TODO
    - Add automatic generation of test matrices
    - Check for validity
    - Benchmark
*/

int main(int argc, char **argv)
{
    const int state_vector_size = 3;
    const int measurement_vector_size = 3;
    const int control_input_vector_size = 1;

    std::vector<float> observation_mat{1, 0, 1, 3, 5, 0, 3, 6, 9}; // H
    auto observation_matrix = getMatrix<Eigen::Matrix<float, 3, 3>>(observation_mat);
    std::vector<float> measurement_noise_covariance{4, 2, 7, 1, 4, 7, 8, 9, 3};
    auto measurement_noise_covariance_matrix = getMatrix<Eigen::Matrix<float, 3, 3>>(measurement_noise_covariance); // R

    Eigen::Vector<float, state_vector_size>
        state_vector(0, 1, 0); // state at t=0

    Eigen::Matrix<float, state_vector_size, state_vector_size> state_covariance_matrix; // P
    state_covariance_matrix << 1, 0, 0, 1, 0, 1, 0, 1, 1;

    // state transition F

    auto state_transition = [](float x) -> Eigen::Matrix<float, 3, 3>
    {
        Eigen::Matrix<float, 3, 3> mat;
        mat << 1 * pow(x, 1), 1 * pow(x, 1), 0,
            1, 2.3, 2 * pow(x, 2),
            6 * pow(x, 3), 4.2 * 1, 1.8 * 1;
        return mat;
    };

    auto StateTransition_fs = make_matrix_function_wrapper<float, 3, 3>(state_transition);

    auto process_noise = [](float x) -> Eigen::Matrix<float, 3, 3>
    {
        Eigen::Matrix<float, 3, 3> mat;
        mat << 1 * pow(x, 1), 1 * pow(x, 1), 0,
            1, 2.3, 2 * pow(x, 2),
            6 * pow(x, 3), 4.2 * 1, 1.8 * 1;
        return mat;
    };

    auto ProcessNoiseCov_fs = make_matrix_function_wrapper<float, 3, 3>(process_noise);

    auto control = [](float x) -> Eigen::Matrix<float, 3, 1>
    {
        Eigen::Matrix<float, 3, 1> mat;
        mat << 1 * pow(x, 1), 1 * pow(x, 1), 0;
        return mat;
    };

    auto Control_fs = make_matrix_function_wrapper<float, 3, 1>(control);

    using MeasurementVectorLayout = MeasurementContainer<Eigen::Vector<float, 2>, Eigen::Vector<float, 1>>;
    KalmanFilter<MeasurementVectorLayout, decltype(StateTransition_fs), decltype(ProcessNoiseCov_fs), decltype(Control_fs)> kf(measurement_noise_covariance_matrix, observation_matrix, StateTransition_fs, ProcessNoiseCov_fs, Control_fs);

    std::vector<float> init_state{1.0, 0.0, 1.0};
    std::vector<float> init_state_cov{1.0, 0.0, 0.0, 1.0, 0, 1, 0, 1, 1};

    kf.init(init_state, init_state_cov);

    /*********** Benchmarking session *************/
    int num_iter = 100000;
    std::random_device rd;  // Non-deterministic seed
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    std::uniform_real_distribution<float> dist_dt(0.0f, 1.0f);
    std::vector<Eigen::Vector<float, 3>> values(num_iter);
    std::vector<Eigen::Vector<float, 2>> values_part1(num_iter);
    std::vector<Eigen::Vector<float, 1>> values_part2(num_iter);
    std::vector<Eigen::Vector<float, 1>> input_values(num_iter);
    std::vector<float> dt_values(num_iter);
    std::vector<std::vector<float>> partial_values(num_iter);
    std::vector<std::vector<float>> partial_values1(num_iter);

    std::cout << "Num of measurements: " << values.size() << std::endl;

    for (int i = 0; i < num_iter; ++i)
    {
        values[i] = Eigen::Vector<float, 3>(dist(gen), dist(gen), dist(gen));
        values_part1[i] = Eigen::Vector<float, 2>(dist(gen), dist(gen));

        input_values[i] = Eigen::Vector<float, 1>(dist(gen));

        dt_values[i] = dist_dt(gen);

        partial_values[i] = {dist(gen), dist(gen)};
        partial_values1[i] = {dist(gen)};
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < values.size(); ++i)
    {
        kf.filter_full(values[i], dt_values[i], input_values[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "Average time per iteration: " << duration / num_iter << " microseconds\n";

    constexpr std::array<int, 2> solo_meas = {0, 1};
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < values.size(); ++i)
    {
        kf.filter_partial<0>(values_part1[i], dt_values[i], input_values[i]);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "Average time per iteration for partial updates: " << duration / num_iter << " microseconds\n";

    return 0;
}
