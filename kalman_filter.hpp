#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "measurement_container.hpp"
#include "utils.hpp"
#include "matrix_eval.hpp"

/*TODO
    - Make an example.
    - Optimize the multiplication flow by reducing duplication. Not needed because of lazy evalution of Eigen?
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
    using StateCovMat = Eigen::Matrix<float, StateSize, StateSize>;
    using ControlVector = Eigen::Vector<float, ControlSize>;
    using MeasurementVector = Eigen::Vector<float, MeasSize>;
    using ObservationMat = Eigen::Matrix<float, MeasSize, StateSize>;
    using MeasNoiseCovMat = Eigen::Matrix<float, MeasSize, MeasSize>;
    using VectorTypes = typename MeasurementContainer::vector_types;

public:
    KalmanFilter(MeasNoiseCovMat meas_noise_cov_mat, ObservationMat obs_mat, StateTransitionMatrix_dt state_trans_mat, ProcessNoiseCov_dt proc_noise_cov_mat, ControlMatrix_dt control_mat = default_dummy_wrapper)
        : meas_noise_cov_mat_(meas_noise_cov_mat), obs_mat_(obs_mat), state_trans_dt_(std::move(state_trans_mat)), process_noise_dt_(std::move(proc_noise_cov_mat)), control_dt_(std::move(control_mat))
    {
    }

    // Initialization functions using an estimate of the initial state and covariance.
    //  The std::vector version requires that the matrices are vectorized in a row major manner.
    void init(std::vector<float> initial_state, std::vector<float> initial_state_covariance)
    {
        toMatrix(initial_state, initial_state_);
        toMatrix(initial_state_covariance, initial_state_cov_);
        initialized_ = true;
    }

    void init(StateVector initial_state, StateCovMat initial_state_covariance)
    {
        initial_state_(initial_state);
        initial_state_cov_(initial_state_covariance);
        initialized_ = true;
    }

    // Filtering function using only one measurement (from the declared ones) along with the index it has in the MeasurementContainer, indexes start from 0.
    template <std::size_t Idx>
    void filter_partial(const typename std::tuple_element<Idx, typename MeasurementContainer::vector_types>::type &vec, float dt, ControlVector control_vec = ControlVector{})
    {
        if (!initialized_)
        {
            std::cerr << "Filter needs to be initialized with initial state and covariance" << std::endl;
            return;
        }

        partial_update<Idx>(predict(dt, control_vec), vec);
    }

    // Filtering using the full measurement vector. The measurements need to be appended in the same way they were declared in
    // the MeasurementContainer.
    void filter_full(MeasurementVector new_measurement, float dt, ControlVector control_vec = ControlVector{})
    {
        if (!initialized_)
        {
            std::cerr << "Filter needs to be initialized with initial state and covariance" << std::endl;
            return;
        }
        update(predict(dt, control_vec), new_measurement);
    }

    // Getters of the current estimated state and state covariance
    const StateVector &getState()
    {
        return initial_state_;
    }

    const StateCovMat &getStateCov()
    {
        return initial_state_cov_;
    }

    std::pair<const StateVector &, const StateCovMat &> get_estimate()
    {
        return std::make_pair(initial_state_, initial_state_cov_);
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
        initial_state_ = pred_res.predicted_state_ + kalman_gain * /*innovation part*/ (new_measurement - obs_mat_ * pred_res.predicted_state_);
        initial_state_cov_ = (Eigen::Matrix<float, StateSize, StateSize>::Identity() - kalman_gain * obs_mat_) * pred_res.predicted_covariance_;
    }

    template <int Idx>
    void partial_update(const PredictionResult &pred_res,
                               const typename std::tuple_element<Idx, typename MeasurementContainer::vector_types>::type &new_measurement)
    {
        constexpr std::size_t N = std::tuple_size<VectorTypes>::value;

        if (Idx < 0 || static_cast<std::size_t>(Idx) >= N)
        {
            throw std::out_of_range("Invalid measurement index");
        }
        constexpr std::size_t start = MeasurementContainer::template start_index<Idx>();
        constexpr std::size_t end = MeasurementContainer::template end_index<Idx>();
        constexpr std::size_t rows = end - start + 1;
        auto H_partial = obs_mat_.block(start, 0, rows , obs_mat_.cols());
        auto R_partial = meas_noise_cov_mat_.block(start, start, rows, rows);

        auto kalman_gain = pred_res.predicted_covariance_ * H_partial.transpose() *
                           (H_partial * pred_res.predicted_covariance_ * H_partial.transpose() + R_partial).inverse();

        initial_state_.noalias() = pred_res.predicted_state_ + kalman_gain * /*innovation part*/ (new_measurement - H_partial * pred_res.predicted_state_);
        initial_state_cov_.noalias() = (Eigen::Matrix<float, StateSize, StateSize>::Identity() - kalman_gain * H_partial) * pred_res.predicted_covariance_;
    }
};
