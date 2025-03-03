/**
 * @file GaussianMixtureModel.hpp
 * @brief K-component, N-dimensional Gaussian Mixture Model (GMM) with EM.
 *
 * @details
 * A GMM models data as a weighted combination of K Gaussian (bell-curve)
 * distributions. Each component has its own mean (center), variance
 * (spread), and weight (what fraction of the data it explains).
 *
 * For a data point x, the overall probability is:
 *
 *   p(x) = w1 * Gauss(x, mean1, var1) + w2 * Gauss(x, mean2, var2) + ...
 *
 * where each Gauss() is a bell curve centered at the component's mean.
 * In N dimensions with diagonal (independent) covariance, each
 * dimension contributes independently:
 *
 *   Gauss(x, mean, var) = product over each dimension d of:
 *       (1 / sqrt(2*pi*var_d)) * exp(-(x_d - mean_d)^2 / (2*var_d))
 *
 * The EM algorithm alternates E-step (estimate which component each
 * point likely came from) and M-step (update component parameters),
 * monitoring log-likelihood for convergence.
 */

#ifndef GAUSSIAN_MIXTURE_MODEL_HPP
#define GAUSSIAN_MIXTURE_MODEL_HPP

#include <concepts>
#include <cstddef>
#include <vector>

/**
 * @class GaussianMixtureModel
 * @brief K-component N-dimensional Gaussian Mixture Model trained with EM.
 * @tparam Scalar Floating-point type for all calculations.
 */
template <std::floating_point Scalar = double>
class GaussianMixtureModel {
public:
    using Vector = std::vector<Scalar>;
    using Matrix = std::vector<Vector>;

    /**
     * @brief Parameters for a single Gaussian component.
     */
    struct Component {
        Vector mean;      ///< Mean vector (N-dimensional)
        Vector variance;  ///< Diagonal variance vector (N-dimensional)
        Scalar weight;    ///< Mixing weight (prior probability)
    };

    /**
     * @brief Construct a K-component GMM.
     * @param n_components Number of Gaussian components.
     * @param max_iterations Maximum EM iterations.
     * @param tolerance Convergence threshold for log-likelihood change.
     */
    GaussianMixtureModel(std::size_t n_components,
                         int max_iterations = 100,
                         Scalar tolerance = static_cast<Scalar>(1e-6));

    /**
     * @brief Fit the GMM to data using EM.
     * @param data Input data matrix (each row is an N-dimensional sample).
     */
    void Fit(const Matrix& data);

    /**
     * @brief Predict the most likely component index for each data point.
     * @param data Input data matrix.
     * @return Vector of component indices (hard assignment).
     */
    std::vector<int> Predict(const Matrix& data) const;

    /**
     * @brief For each data point, compute the probability it belongs to each cluster.
     * @param data Input data matrix (each row is one data point).
     * @return A matrix where result[i][k] is the probability (0.0 to 1.0) that
     *         data point i belongs to cluster k.  Each row sums to 1.0.
     *         Example: result[5] = {0.9, 0.05, 0.05} means point 5 has a 90%
     *         chance of being in cluster 0 and 5% each for clusters 1 and 2.
     */
    Matrix Responsibilities(const Matrix& data) const;

    /**
     * @brief Get the fitted component parameters.
     * @return Const reference to the vector of Component structs.
     */
    const std::vector<Component>& GetComponents() const;

    /**
     * @brief Compute the log-likelihood of the data under the current model.
     * @param data Input data matrix.
     * @return Total log-likelihood.
     */
    Scalar LogLikelihood(const Matrix& data) const;

private:
    /**
     * @brief E-step: compute responsibilities for all data points.
     * @param data Input data matrix.
     * @param responsibilities Output matrix of soft assignments.
     */
    void EStep(const Matrix& data, Matrix& responsibilities) const;

    /**
     * @brief M-step: update component parameters from responsibilities.
     * @param data Input data matrix.
     * @param responsibilities Matrix of soft assignments from E-step.
     */
    void MStep(const Matrix& data, const Matrix& responsibilities);

    /**
     * @brief Evaluate the diagonal-covariance Gaussian PDF at point x.
     * @param x Data point (N-dimensional).
     * @param mean Mean vector.
     * @param variance Diagonal variance vector.
     * @return PDF value.
     */
    static Scalar GaussianPdf(const Vector& x,
                               const Vector& mean,
                               const Vector& variance);

    std::size_t n_components_;   ///< Number of mixture components
    int max_iterations_;         ///< Maximum EM iterations
    Scalar tolerance_;           ///< Log-likelihood convergence threshold
    std::vector<Component> components_;  ///< Fitted component parameters
};

#endif // GAUSSIAN_MIXTURE_MODEL_HPP
