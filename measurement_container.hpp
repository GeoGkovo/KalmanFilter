#ifndef MEASUREMENT_CONTAINER_HPP
#define MEASUREMENT_CONTAINER_HPP

#include <Eigen/Dense>
#include <tuple>

template <typename... VectorTypes>
class MeasurementContainer
{
public:
    using vector_types = std::tuple<VectorTypes...>;

    template <std::size_t Idx>
    using vector_type_at = typename std::tuple_element<Idx, vector_types>::type;

    MeasurementContainer() = default;

    template <std::size_t I>
    static constexpr std::size_t rows()
    {
        using Vec = typename std::tuple_element<I, vector_types>::type;
        return Vec::RowsAtCompileTime;
    }

    // --- Total number of rows at compile time ---
    static constexpr int total_rows_compiletime = (VectorTypes::RowsAtCompileTime + ...);

    // Start index of the I-th vector in the full concatenated vector
    template <std::size_t I>
    static constexpr std::size_t start_index()
    {
        return sum_rows_before<I>();
    }

    // End index (exclusive) of the I-th vector
    template <std::size_t I>
    static constexpr std::size_t end_index()
    {
        return start_index<I>() + rows<I>() - 1;
    }

    // Total size
    static constexpr std::size_t total_rows()
    {
        return sum_rows_before<sizeof...(VectorTypes)>();
    }

    // Helper: sum rows of all vectors before index I
    template <std::size_t I>
    static constexpr std::size_t sum_rows_before()
    {
        std::size_t sum = 0;
        if constexpr (I == 0)
        {
            return 0;
        }
        else
        {
            return sum_rows_before<I - 1>() + rows<I - 1>();
        }
    }

private:
    std::tuple<VectorTypes...> vectors_;
};

#endif
