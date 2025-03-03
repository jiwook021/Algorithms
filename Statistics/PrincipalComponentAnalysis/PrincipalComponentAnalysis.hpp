/**
 * @file PrincipalComponentAnalysis.hpp
 * @brief N-dimensional Principal Component Analysis (PCA).
 *
 * @details
 * PCA finds the most important directions in data.  If you have data
 * with many features (columns), PCA discovers which features vary
 * together and compresses them into a smaller set of new features
 * called principal components.
 *
 * Each principal component is a direction in the original feature
 * space.  The first component (PC1) points in the direction of
 * greatest variance, the second (PC2) captures the most remaining
 * variance while being perpendicular to PC1, and so on.
 *
 * In machine learning, PCA is used to:
 *   - Reduce dimensionality (fewer features, faster training)
 *   - Remove noise (high-variance = signal, low-variance = noise)
 *   - Visualize high-dimensional data in 2D or 3D
 *   - Decorrelate features for algorithms that assume independence
 *
 * Internally, PCA centers the data (subtracts the mean of each
 * feature), computes how features co-vary (the covariance matrix),
 * then finds the directions of maximum variance using Jacobi
 * eigendecomposition (iterative rotations that diagonalize the
 * covariance matrix).
 */

#ifndef PRINCIPAL_COMPONENT_ANALYSIS_HPP
#define PRINCIPAL_COMPONENT_ANALYSIS_HPP

#include <concepts>
#include <cstddef>
#include <vector>

/**
 * @class PrincipalComponentAnalysis
 * @brief N-dimensional PCA via covariance eigendecomposition (Jacobi method).
 * @tparam Scalar Floating-point type (double by default).
 */
template <std::floating_point Scalar = double>
class PrincipalComponentAnalysis {
public:
    using Matrix = std::vector<std::vector<Scalar>>;
    using Vector = std::vector<Scalar>;

    /**
     * @brief Fit PCA to the dataset (compute mean, covariance, eigenvectors).
     * @param data Input matrix: N_samples x N_features.
     */
    void Fit(const Matrix& data);

    /**
     * @brief Project data onto the top n_components principal components.
     * @param data Input matrix: N_samples x N_features.
     * @param n_components Number of components to keep.
     * @return Transformed matrix: N_samples x n_components.
     */
    Matrix Transform(const Matrix& data, std::size_t n_components) const;

    /**
     * @brief Fit the model and transform in one call.
     * @param data Input matrix: N_samples x N_features.
     * @param n_components Number of components to keep.
     * @return Transformed matrix: N_samples x n_components.
     */
    Matrix FitTransform(const Matrix& data, std::size_t n_components);

    /** @brief Fraction of total variance explained by each component (descending). */
    const Vector& GetExplainedVarianceRatio() const;

    /** @brief Eigenvectors (rows) sorted by descending eigenvalue. */
    const Matrix& GetComponents() const;

    /** @brief Per-feature mean computed during Fit. */
    const Vector& GetMean() const;

private:
    /**
     * @brief Compute the covariance matrix of already-centered data.
     * @param centered Centered data matrix (N_samples x N_features).
     * @return Covariance matrix (N_features x N_features).
     */
    Matrix ComputeCovarianceMatrix(const Matrix& centered) const;

    /**
     * @brief Center the data by subtracting the per-feature mean.
     * @param data Data matrix (modified in-place to centered form).
     * @return The computed mean vector.
     */
    Vector CenterData(Matrix& data) const;

    /**
     * @brief Jacobi eigendecomposition of a symmetric matrix.
     * @param matrix Symmetric input matrix (destroyed during computation).
     * @param eigenvalues Output eigenvalues.
     * @param eigenvectors Output eigenvectors (columns).
     */
    void JacobiEigendecomposition(Matrix& matrix, Vector& eigenvalues,
                                  Matrix& eigenvectors) const;

    Matrix components_;              ///< Eigenvectors as rows, sorted descending
    Vector explained_variance_;      ///< Fraction of variance per component
    Vector mean_;                    ///< Per-feature mean
    bool fitted_{false};             ///< Whether Fit has been called
};

#endif // PRINCIPAL_COMPONENT_ANALYSIS_HPP
