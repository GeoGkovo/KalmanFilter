#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "measurement_container.hpp"
#include "utils.hpp"
#include "matrix_eval.hpp"

/*TODO
    - Make an example.
    - Optimize the multiplication flow by reducing duplication.
    - Use smaller names for typedefs. Directly from math.
    - Add initialization strategy (plugin)
*/

template <typename MeasurementContainer, typename StateTransitionMatrix_dt, typename ProcessNoiseCov_dt, typename ControlMatrix_dt = decltype(default_dummy_wrapper)>
class KalmanFilter
{
    static constexpr int MeasSize = MeasurementContainer::total_rows_compiletime;
    static constexpr int StateSize = StateTransitionMatrix_dt::rows;
    static constexpr int ControlSize = ControlMatrix_dt::cols;
    using StateVector = Eigen::Vector<float, StateSize>;
    using ControlVector = Eigen::Vector<float, ControlSize>;
    using MeasurementVector = Eigen::Vector<float, MeasSize>;
    using StateCovMat = Eigen::Matrix<float, StateSize, StateSize>;
    using ObservationMat = Eigen::Matrix<float, MeasSize, StateSize>;
    using MeasNoiseCovMat = Eigen::Matrix<float, MeasSize, MeasSize>;
    using VectorTypes = typename MeasurementContainer::vector_types;

public:
    KalmanFilter(MeasNoiseCovMat meas_noise_cov_mat, ObservationMat obs_mat, StateTransitionMatrix_dt func_set1, ProcessNoiseCov_dt func_set2, ControlMatrix_dt func_set3 = default_dummy_wrapper)
        : meas_noise_cov_mat_(meas_noise_cov_mat), obs_mat_(obs_mat), state_trans_dt_(std::move(func_set1)), process_noise_dt_(std::move(func_set2)), control_dt_(std::move(func_set3))
    {
    }

    void filter_partial(std::vector<float> new_measurement, int measurement_idx, float dt, ControlVector control_vec = ControlVector{})
    {
        if (!initialized_)
        {
            std::cout << "Filter needs to be initialized with initial state and covariance" << std::endl;
            return;
        }
        partial_update(predict(dt, control_vec), new_measurement, measurement_idx);
    }

    void filter_full(MeasurementVector new_measurement, float dt, ControlVector control_vec = ControlVector{})
    {
        if (!initialized_)
        {
            std::cout << "Filter needs to be initialized with initial state and covariance" << std::endl;
            return;
        }
        update(predict(dt, control_vec), new_measurement);
    }

    void init(std::vector<float> initial_state, std::vector<float> initial_state_covariance)
    {
        toMatrix(initial_state, initial_state_);
        toMatrix(initial_state_covariance, initial_state_cov_);
        initialized_ = true;
    }

    const StateVector &getState()
    {
        return initial_state_;
    }

    const StateCovMat &getStateCov()
    {
        return initial_state_cov_;
    }

private:
    StateVector initial_state_;
    StateCovMat initial_state_cov_;

    const ObservationMat obs_mat_;
    const MeasNoiseCovMat meas_noise_cov_mat_;
    const StateTransitionMatrix_dt state_trans_dt_;
    const ProcessNoiseCov_dt process_noise_dt_;
    const ControlMatrix_dt control_dt_;

    bool initialized_;

    struct PredictionResult
    {
        PredictionResult(StateVector predicted_state, StateCovMat predicted_covariance) : predicted_state_(predicted_state), predicted_covariance_(predicted_covariance) {

                                                                                          };
        StateVector predicted_state_;
        StateCovMat predicted_covariance_;
    };

    PredictionResult
    predict(float dt, ControlVector control_input = ControlVector{})
    {
        if constexpr (ControlSize == 0)
        {

            return PredictionResult{
                state_trans_dt_.evaluate(dt) * initial_state_,
                state_trans_dt_.evaluate(dt) * initial_state_cov_ * state_trans_dt_.evaluate(dt).transpose() + process_noise_dt_.evaluate(dt)};
        }
        else
        {
            return PredictionResult{
                (state_trans_dt_.evaluate(dt) * initial_state_) + control_dt_.evaluate(dt) * control_input,
                state_trans_dt_.evaluate(dt) * initial_state_cov_ * (state_trans_dt_.evaluate(dt)).transpose() + process_noise_dt_.evaluate(dt)};
        }
    }
    void update(PredictionResult pred_res, MeasurementVector new_measurement)
    {
        auto kalman_gain = pred_res.predicted_covariance_ * obs_mat_.transpose() * (obs_mat_ * pred_res.predicted_covariance_ * obs_mat_.transpose() + meas_noise_cov_mat_).inverse();
        auto inovation = new_measurement - obs_mat_ * pred_res.predicted_state_;
        initial_state_ = pred_res.predicted_state_ + kalman_gain * inovation;
        initial_state_cov_ = (Eigen::Matrix<float, StateSize, StateSize>::Identity() - kalman_gain * obs_mat_) * pred_res.predicted_covariance_;
    }

    // partial_update static implementation
    inline void partial_update(const PredictionResult &pred_res,
                               const std::vector<float> &new_measurement,
                               int measurement_idx)
    {
        constexpr std::size_t N = std::tuple_size<VectorTypes>::value;
        static constexpr auto dispatch_table = make_dispatch_table(std::make_index_sequence<N>{});

        if (measurement_idx < 0 || static_cast<std::size_t>(measurement_idx) >= N)
        {
            throw std::out_of_range("Invalid measurement index");
        }

        auto fn = dispatch_table[measurement_idx];
        (this->*fn)(pred_res, new_measurement);
    }

    // templated private function
    template <int Index>
    inline void partial_update_impl(const PredictionResult &pred_res,
                                    const std::vector<float> &new_measurement)
    {
        constexpr std::size_t start = MeasurementContainer::template start_index<Index>();
        constexpr std::size_t end = MeasurementContainer::template end_index<Index>();
        constexpr std::size_t rows = end - start;

        using SubVec = Eigen::Matrix<float, rows, 1>;
        Eigen::Map<const SubVec> new_meas(new_measurement.data() + start);

        auto H_partial = obs_mat_.block(start, 0, rows, obs_mat_.cols());
        auto R_partial = meas_noise_cov_mat_.block(start, start, rows, rows);

        auto kalman_gain = pred_res.predicted_covariance_ * H_partial.transpose() *
                           (H_partial * pred_res.predicted_covariance_ * H_partial.transpose() + R_partial).inverse();

        auto innovation = new_meas - H_partial * pred_res.predicted_state_;
        initial_state_.noalias() = pred_res.predicted_state_ + kalman_gain * innovation;
        initial_state_cov_.noalias() = (Eigen::Matrix<float, StateSize, StateSize>::Identity() - kalman_gain * H_partial) * pred_res.predicted_covariance_;
    }

    // static dispatch table of pointers to templated member functions.
    template <std::size_t... Is>
    static constexpr auto make_dispatch_table(std::index_sequence<Is...>)
    {
        using FnPtr = void (KalmanFilter::*)(const PredictionResult &, const std::vector<float> &);
        return std::array<FnPtr, sizeof...(Is)>{&KalmanFilter::partial_update_impl<Is>...};
    }
};
