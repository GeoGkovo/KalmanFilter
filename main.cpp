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

template <int StateSize, int MeasurementSize, int ControlSize, int SampleNum>
class KalmanTest
{
public:
    using StateVector = Eigen::Vector<float, StateSize>;

    KalmanTest() : dist(1.0f, 10.0f), dist_dt(0.0f, 1.0f), gen(std::random_device{}())
    {
        generateRandomMatrix(initial_state_vec_);
        generateRandomMatrix(initial_state_cov_matrix_);
        generateRandomMatrix(obs_mat_);
        generateRandomMatrix(meas_noise_cov_mat_);
        dts_.resize(SampleNum);
        control_vec_.resize(SampleNum);
        meas_vec_.resize(SampleNum);
        for (int i = 0; i < SampleNum; i++)
        {
            dts_[i] = dist_dt(gen);
            control_vec_[i] = generateRandomMatrix<float, ControlSize, 1>();
            meas_vec_[i] = generateRandomMatrix<float, MeasurementSize, 1>();
        }
    }

    template <typename T, int Rows, int Cols>
    void generateRandomMatrix(Eigen::Matrix<T, Rows, Cols> &mat)
    {
        if (Rows * Cols <= 0)
        {
            throw std::invalid_argument("Rows an Cols need to be positive numbers.");
        }
        for (int i = 0; i < Rows * Cols; i++)
        {
            mat(i) = dist(gen);
        }
    }

    template <typename T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols> generateRandomMatrix()
    {
        if (Rows * Cols <= 0)
        {
            throw std::invalid_argument("Rows an Cols need to be positive numbers.");
        }
        Eigen::Matrix<float, Rows, Cols> mat;
        for (int i = 0; i < Rows * Cols; i++)
        {
            mat(i) = dist(gen);
        }
        return mat;
    }

private:
    Eigen::Vector<float, StateSize> initial_state_vec_;
    Eigen::Matrix<float, StateSize, StateSize> initial_state_cov_matrix_;

    Eigen::Matrix<float, MeasurementSize, StateSize> obs_mat_;
    Eigen::Matrix<float, MeasurementSize, MeasurementSize> meas_noise_cov_mat_;

    std::vector<float> dts_;
    std::vector<Eigen::Vector<float, ControlSize>> control_vec_;
    std::vector<Eigen::Vector<float, MeasurementSize>> meas_vec_;

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
    std::uniform_real_distribution<float> dist_dt;
};

int main(int argc, char **argv)
{
    //KalmanTest<3, 3, 1, 1000> kt;

    const int state_vector_size = 3;
    const int measurement_vector_size = 3;
    const int control_input_vector_size = 1;

    auto observation_matrix = getMatrix<Eigen::Matrix<float, 3, 3>>(std::vector<float>{1, 0, 1, 3, 5, 0, 3, 6, 9});
    auto measurement_noise_covariance_matrix = getMatrix<Eigen::Matrix<float, 3, 3>>(std::vector<float>{4, 2, 7, 1, 4, 7, 8, 9, 3}); // R

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

    // Process Noise Covariance Q
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
    int num_iter = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    std::uniform_real_distribution<float> dist_dt(0.0f, 1.0f);
    std::vector<Eigen::Vector<float, 3>> values(num_iter);
    std::vector<Eigen::Vector<float, 2>> values_part1(num_iter);
    std::vector<Eigen::Vector<float, 1>> values_part2(num_iter);
    std::vector<Eigen::Vector<float, 1>> input_values(num_iter);
    std::vector<float> dt_values(num_iter);

    std::cout << "Num of measurements: " << values.size() << std::endl;

    for (int i = 0; i < num_iter; ++i)
    {
        values[i] = Eigen::Vector<float, 3>(dist(gen), dist(gen), dist(gen));
        values_part1[i] = Eigen::Vector<float, 2>(dist(gen), dist(gen));

        input_values[i] = Eigen::Vector<float, 1>(dist(gen));

        dt_values[i] = dist_dt(gen);
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
