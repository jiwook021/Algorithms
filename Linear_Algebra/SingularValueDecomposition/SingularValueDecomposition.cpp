/**
 * @file SingularValueDecomposition.cpp
 * @brief Explicit template instantiations for common scalar types.
 * @details The full implementation lives in SingularValueDecomposition.hpp
 *          because it is a class template.  This translation unit provides
 *          explicit instantiations so the linker can find symbols when
 *          the header is not included in every TU.
 */

#include "SingularValueDecomposition.hpp"

// Explicit instantiations for the most common floating-point types.
template class SingularValueDecomposition<float>;
template class SingularValueDecomposition<double>;
template class SingularValueDecomposition<long double>;
