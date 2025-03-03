/**
 * @file LinearSvm.cpp
 * @brief Explicit template instantiations for LinearSvm.
 *
 * All template logic lives in LinearSvm.hpp.  This translation unit
 * provides explicit instantiations so that common specializations are
 * available without requiring every consumer to include the full
 * template definition (though the header is self-contained).
 */

#include "LinearSvm.hpp"

// Explicit instantiations for the two standard floating-point types.
template class LinearSvm<double>;
template class LinearSvm<float>;
