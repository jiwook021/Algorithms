/**
 * @file NeuralNetworkV2.cpp
 * @brief Implementation of Matrix, Vector, LinearRegression, LogisticRegression, and NeuralNetworkV2.
 */
#include "NeuralNetworkV2.hpp"

namespace Ml {

// ======================== Matrix ========================

Matrix::Matrix(size_t rows, size_t cols)
    : Data_(rows, std::vector<double>(cols, 0.0)), Rows_(rows), Cols_(cols) {}

Matrix::Matrix(const std::vector<std::vector<double>>& mat) : Data_(mat) {
    if (mat.empty()) { Rows_ = 0; Cols_ = 0; return; }
    Rows_ = mat.size();
    Cols_ = mat[0].size();
    for (const auto& row : mat) {
        if (row.size() != Cols_) throw std::invalid_argument("All rows must have equal columns");
    }
}

double& Matrix::At(size_t i, size_t j) {
    if (i >= Rows_ || j >= Cols_) throw std::out_of_range("Matrix index out of range");
    return Data_[i][j];
}
double Matrix::At(size_t i, size_t j) const {
    if (i >= Rows_ || j >= Cols_) throw std::out_of_range("Matrix index out of range");
    return Data_[i][j];
}

Matrix Matrix::Transpose() const {
    Matrix r(Cols_, Rows_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(j, i) = Data_[i][j];
    return r;
}

Matrix Matrix::operator+(const Matrix& o) const {
    if (Rows_ != o.Rows_ || Cols_ != o.Cols_) throw std::invalid_argument("Dimension mismatch (+)");
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(i, j) = Data_[i][j] + o.Data_[i][j];
    return r;
}

Matrix Matrix::operator-(const Matrix& o) const {
    if (Rows_ != o.Rows_ || Cols_ != o.Cols_) throw std::invalid_argument("Dimension mismatch (-)");
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(i, j) = Data_[i][j] - o.Data_[i][j];
    return r;
}

Matrix Matrix::operator*(const Matrix& o) const {
    if (Cols_ != o.Rows_) throw std::invalid_argument("Dimension mismatch (*)");
    Matrix r(Rows_, o.Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < o.Cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < Cols_; ++k) sum += Data_[i][k] * o.Data_[k][j];
            r.At(i, j) = sum;
        }
    return r;
}

Matrix Matrix::operator*(double s) const {
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(i, j) = Data_[i][j] * s;
    return r;
}

Matrix Matrix::Hadamard(const Matrix& o) const {
    if (Rows_ != o.Rows_ || Cols_ != o.Cols_) throw std::invalid_argument("Dimension mismatch (hadamard)");
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(i, j) = Data_[i][j] * o.Data_[i][j];
    return r;
}

Matrix Matrix::Apply(std::function<double(double)> fn) const {
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r.At(i, j) = fn(Data_[i][j]);
    return r;
}

Matrix Matrix::Sigmoid() const {
    return Apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

Matrix Matrix::Relu() const {
    return Apply([](double x) { return std::max(0.0, x); });
}

Matrix Matrix::Zeros(size_t r, size_t c) { return Matrix(r, c); }

Matrix Matrix::Ones(size_t r, size_t c) {
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m.At(i, j) = 1.0;
    return m;
}

Matrix Matrix::Identity(size_t n) {
    Matrix m(n, n);
    for (size_t i = 0; i < n; ++i) m.At(i, i) = 1.0;
    return m;
}

Matrix Matrix::Random(size_t r, size_t c, double lo, double hi) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(lo, hi);
    Matrix m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m.At(i, j) = dist(gen);
    return m;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (size_t i = 0; i < m.Rows_; ++i) {
        os << "[";
        for (size_t j = 0; j < m.Cols_; ++j) {
            os << m.Data_[i][j];
            if (j + 1 < m.Cols_) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

// ======================== Vector ========================

Vector::Vector(size_t size) : Data_(size, 0.0), Size_(size) {}
Vector::Vector(const std::vector<double>& v) : Data_(v), Size_(v.size()) {}

double& Vector::At(size_t i) {
    if (i >= Size_) throw std::out_of_range("Vector index out of range");
    return Data_[i];
}
double Vector::At(size_t i) const {
    if (i >= Size_) throw std::out_of_range("Vector index out of range");
    return Data_[i];
}

Matrix Vector::ToMatrix() const {
    std::vector<std::vector<double>> mat(Size_, std::vector<double>(1));
    for (size_t i = 0; i < Size_; ++i) mat[i][0] = Data_[i];
    return Matrix(mat);
}

Vector Vector::FromMatrix(const Matrix& m) {
    if (m.GetCols() != 1) throw std::invalid_argument("Matrix must have 1 col");
    Vector v(m.GetRows());
    for (size_t i = 0; i < m.GetRows(); ++i) v.At(i) = m.At(i, 0);
    return v;
}

double Vector::Dot(const Vector& o) const {
    if (Size_ != o.Size_) throw std::invalid_argument("Dimension mismatch (dot)");
    double r = 0.0;
    for (size_t i = 0; i < Size_; ++i) r += Data_[i] * o.Data_[i];
    return r;
}

double Vector::Norm() const {
    double s = 0.0;
    for (double v : Data_) s += v * v;
    return std::sqrt(s);
}

Vector Vector::Normalize() const {
    double n = Norm();
    if (n == 0) throw std::runtime_error("Cannot normalize zero vector");
    Vector r(Size_);
    for (size_t i = 0; i < Size_; ++i) r.At(i) = Data_[i] / n;
    return r;
}

Vector Vector::operator+(const Vector& o) const {
    if (Size_ != o.Size_) throw std::invalid_argument("Dimension mismatch (+)");
    Vector r(Size_);
    for (size_t i = 0; i < Size_; ++i) r.At(i) = Data_[i] + o.Data_[i];
    return r;
}

Vector Vector::operator-(const Vector& o) const {
    if (Size_ != o.Size_) throw std::invalid_argument("Dimension mismatch (-)");
    Vector r(Size_);
    for (size_t i = 0; i < Size_; ++i) r.At(i) = Data_[i] - o.Data_[i];
    return r;
}

Vector Vector::operator*(double s) const {
    Vector r(Size_);
    for (size_t i = 0; i < Size_; ++i) r.At(i) = Data_[i] * s;
    return r;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "[";
    for (size_t i = 0; i < v.Size_; ++i) {
        os << v.Data_[i];
        if (i + 1 < v.Size_) os << ", ";
    }
    os << "]";
    return os;
}

// ======================== LinearRegression ========================

LinearRegression::LinearRegression(size_t nFeatures, double lr, int maxIter)
    : Weights_(nFeatures), Bias_(0.0), LearningRate_(lr), MaxIterations_(maxIter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (size_t i = 0; i < nFeatures; ++i) Weights_.At(i) = dist(gen);
}

double LinearRegression::Predict(const Vector& x) const {
    return Weights_.Dot(x) + Bias_;
}

void LinearRegression::Fit(const std::vector<Vector>& x, const std::vector<double>& y, bool verbose) {
    if (x.empty() || x.size() != y.size()) throw std::invalid_argument("Invalid training data");
    size_t n = x.size();
    for (int iter = 0; iter < MaxIterations_; ++iter) {
        Vector dw(Weights_.GetSize());
        double db = 0.0, totalErr = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double err = Predict(x[i]) - y[i];
            totalErr += err * err;
            for (size_t j = 0; j < Weights_.GetSize(); ++j) dw.At(j) += err * x[i].At(j);
            db += err;
        }
        for (size_t j = 0; j < Weights_.GetSize(); ++j)
            Weights_.At(j) -= LearningRate_ * dw.At(j) / static_cast<double>(n);
        Bias_ -= LearningRate_ * db / static_cast<double>(n);
        if (verbose && iter % 100 == 0)
            std::cout << "Iteration " << iter << ": MSE = " << totalErr / n << "\n";
    }
}

Vector LinearRegression::GetWeights() const { return Weights_; }
double LinearRegression::GetBias() const { return Bias_; }

// ======================== LogisticRegression ========================

LogisticRegression::LogisticRegression(size_t nFeatures, double lr, int maxIter)
    : Weights_(nFeatures), Bias_(0.0), LearningRate_(lr), MaxIterations_(maxIter) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (size_t i = 0; i < nFeatures; ++i) Weights_.At(i) = dist(gen);
}

double LogisticRegression::SigmoidFn(double x) const { return 1.0 / (1.0 + std::exp(-x)); }

double LogisticRegression::PredictProba(const Vector& x) const {
    return SigmoidFn(Weights_.Dot(x) + Bias_);
}

int LogisticRegression::Predict(const Vector& x) const {
    return PredictProba(x) >= 0.5 ? 1 : 0;
}

void LogisticRegression::Fit(const std::vector<Vector>& x, const std::vector<int>& y, bool verbose) {
    if (x.empty() || x.size() != y.size()) throw std::invalid_argument("Invalid training data");
    size_t n = x.size();
    for (int iter = 0; iter < MaxIterations_; ++iter) {
        Vector dw(Weights_.GetSize());
        double db = 0.0, totalLoss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double p = PredictProba(x[i]);
            double err = p - y[i];
            totalLoss += y[i] == 1 ? -std::log(p + 1e-15) : -std::log(1 - p + 1e-15);
            for (size_t j = 0; j < Weights_.GetSize(); ++j) dw.At(j) += err * x[i].At(j);
            db += err;
        }
        for (size_t j = 0; j < Weights_.GetSize(); ++j)
            Weights_.At(j) -= LearningRate_ * dw.At(j) / static_cast<double>(n);
        Bias_ -= LearningRate_ * db / static_cast<double>(n);
        if (verbose && iter % 100 == 0)
            std::cout << "Iteration " << iter << ": Loss = " << totalLoss / n << "\n";
    }
}

Vector LogisticRegression::GetWeights() const { return Weights_; }
double LogisticRegression::GetBias() const { return Bias_; }

// ======================== NeuralNetworkV2 ========================

NeuralNetworkV2::NeuralNetworkV2(const std::vector<size_t>& layerSizes,
                                  double lr, int maxIter)
    : LearningRate_(lr), MaxIterations_(maxIter) {
    if (layerSizes.size() < 2) throw std::invalid_argument("Need >= 2 layers");
    for (size_t i = 1; i < layerSizes.size(); ++i) {
        Weights_.push_back(Matrix::Random(layerSizes[i], layerSizes[i - 1], -0.1, 0.1));
        Biases_.emplace_back(layerSizes[i]);
    }
}

Matrix NeuralNetworkV2::SigmoidMat(const Matrix& z) const { return z.Sigmoid(); }

Matrix NeuralNetworkV2::SigmoidDerivative(const Matrix& a) const {
    return a.Hadamard(Matrix::Ones(a.GetRows(), a.GetCols()) - a);
}

Matrix NeuralNetworkV2::Forward(const Matrix& x) const {
    Matrix act = x;
    for (size_t i = 0; i < Weights_.size(); ++i) {
        Matrix z = Weights_[i] * act.Transpose();
        for (size_t r = 0; r < z.GetRows(); ++r)
            for (size_t c = 0; c < z.GetCols(); ++c)
                z.At(r, c) += Biases_[i].At(r);
        act = SigmoidMat(z).Transpose();
    }
    return act;
}

void NeuralNetworkV2::Fit(const Matrix& x, const Matrix& y, bool verbose) {
    size_t n = x.GetRows();
    for (int iter = 0; iter < MaxIterations_; ++iter) {
        // Forward
        std::vector<Matrix> activations{x};
        Matrix act = x;
        for (size_t i = 0; i < Weights_.size(); ++i) {
            Matrix z = Weights_[i] * act.Transpose();
            for (size_t r = 0; r < z.GetRows(); ++r)
                for (size_t c = 0; c < z.GetCols(); ++c)
                    z.At(r, c) += Biases_[i].At(r);
            act = SigmoidMat(z).Transpose();
            activations.push_back(act);
        }

        // Loss
        Matrix output = activations.back();
        double totalLoss = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < y.GetCols(); ++j) {
                double p = output.At(i, j), t = y.At(i, j);
                totalLoss += -t * std::log(p + 1e-15) - (1 - t) * std::log(1 - p + 1e-15);
            }

        // Backprop
        std::vector<Matrix> deltas;
        deltas.push_back(output - y);
        for (int i = static_cast<int>(Weights_.size()) - 2; i >= 0; --i) {
            Matrix d = (Weights_[static_cast<size_t>(i) + 1].Transpose() * deltas.back().Transpose())
                           .Transpose()
                           .Hadamard(SigmoidDerivative(activations[static_cast<size_t>(i) + 1]));
            deltas.push_back(d);
        }
        std::reverse(deltas.begin(), deltas.end());

        // Update
        for (size_t i = 0; i < Weights_.size(); ++i) {
            Matrix dw = deltas[i].Transpose() * activations[i];
            for (size_t r = 0; r < Weights_[i].GetRows(); ++r)
                for (size_t c = 0; c < Weights_[i].GetCols(); ++c)
                    Weights_[i].At(r, c) -= LearningRate_ * dw.At(r, c) / static_cast<double>(n);
            for (size_t j = 0; j < Biases_[i].GetSize(); ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < deltas[i].GetRows(); ++k) sum += deltas[i].At(k, j);
                Biases_[i].At(j) -= LearningRate_ * sum / static_cast<double>(n);
            }
        }

        if (verbose && iter % 100 == 0)
            std::cout << "Iteration " << iter << ": Loss = " << totalLoss / n << "\n";
    }
}

Matrix NeuralNetworkV2::Predict(const Matrix& x) const { return Forward(x); }

} // namespace Ml
