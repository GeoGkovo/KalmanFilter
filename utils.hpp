#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

template <typename Derived>
void toMatrix(const std::vector<typename Derived::Scalar> &data, Eigen::MatrixBase<Derived> &mat)
{
    using Scalar = typename Derived::Scalar;
    constexpr int Rows = Derived::RowsAtCompileTime;
    constexpr int Cols = Derived::ColsAtCompileTime;

    if (data.size() != Rows * Cols)
    {
        throw std::invalid_argument("Input vector size does not match matrix dimensions.");
    }

    // If it's a column vector, avoid RowMajor mapping
    if constexpr (Cols == 1)
    {
        Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>> data_view(data.data());
        mat = data_view;
    }
    else
    {
        Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>> data_view(data.data());
        mat = data_view;
    }
}

template <typename Derived>
Derived getMatrix(const std::vector<typename Derived::Scalar> &data)
{
    using Scalar = typename Derived::Scalar;
    constexpr int Rows = Derived::RowsAtCompileTime;
    constexpr int Cols = Derived::ColsAtCompileTime;
    Derived mat;

    if (data.size() != Rows * Cols)
    {
        throw std::invalid_argument("Input vector size does not match matrix dimensions.");
    }

    // If it's a column vector, avoid RowMajor mapping
    if constexpr (Cols == 1)
    {
        Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols>> data_view(data.data());
        mat = data_view;
    }
    else
    {
        Eigen::Map<const Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor>> data_view(data.data());
        mat = data_view;
    }
    return mat;
}

#endif
