#ifndef MATRIX_EVAL_HPP
#define MATRIX_EVAL_HPP

#include <Eigen/Dense>

template <typename T, int Rows, int Cols, typename F>
struct MatrixFunctionWrapper
{
    F func;

    static constexpr int rows = Rows;
    static constexpr int cols = Cols;

    constexpr MatrixFunctionWrapper(F f) : func(std::move(f)) {}

    Eigen::Matrix<T, Rows, Cols> evaluate(T input) const
    {
        return func(input);
    }
};

template <typename T, int Rows, int Cols, typename F>
auto make_matrix_function_wrapper(F f)
{
    return MatrixFunctionWrapper<T, Rows, Cols, F>{std::move(f)};
}

const auto default_dummy_wrapper = make_matrix_function_wrapper<double, 0, 0>(
    [](double) -> Eigen::Matrix<double, 0, 0>
    {
        return Eigen::Matrix<double, 0, 0>();
    });

#endif
