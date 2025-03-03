/**
 * @file NeuralNetworkImplementation.cpp
 * @brief Implementation of Vector, Matrix, and gradient descent.
 */
#include "NeuralNetworkImplementation.hpp"

namespace Ml {

// ======================== Vector ========================
Vector::Vector(size_t size) : Data_(size, 0.0) {}
Vector::Vector(const std::vector<double>& data) : Data_(data) {}
size_t Vector::Size() const { return Data_.size(); }
double& Vector::operator[](size_t i) { return Data_[i]; }
const double& Vector::operator[](size_t i) const { return Data_[i]; }

Vector Vector::operator+(const Vector& o) const {
    if (Size() != o.Size()) throw std::invalid_argument("Vector size mismatch (+)");
    Vector r(Size());
    for (size_t i = 0; i < Size(); ++i) r[i] = Data_[i] + o[i];
    return r;
}

Vector Vector::operator*(double s) const {
    Vector r(Size());
    for (size_t i = 0; i < Size(); ++i) r[i] = Data_[i] * s;
    return r;
}

double Vector::Dot(const Vector& o) const {
    if (Size() != o.Size()) throw std::invalid_argument("Vector size mismatch (dot)");
    double sum = 0.0;
    for (size_t i = 0; i < Size(); ++i) sum += Data_[i] * o[i];
    return sum;
}

// ======================== Matrix ========================
Matrix::Matrix(size_t rows, size_t cols) : Rows_(rows), Cols_(cols), Data_(rows * cols, 0.0) {}

double& Matrix::operator()(size_t i, size_t j) { return Data_[i * Cols_ + j]; }
const double& Matrix::operator()(size_t i, size_t j) const { return Data_[i * Cols_ + j]; }
size_t Matrix::GetRows() const { return Rows_; }
size_t Matrix::GetCols() const { return Cols_; }

Matrix Matrix::operator+(const Matrix& o) const {
    if (Rows_ != o.Rows_ || Cols_ != o.Cols_) throw std::invalid_argument("Matrix dim mismatch (+)");
    Matrix r(Rows_, Cols_);
    for (size_t i = 0; i < Rows_ * Cols_; ++i) r.Data_[i] = Data_[i] + o.Data_[i];
    return r;
}

Matrix Matrix::operator*(const Matrix& o) const {
    if (Cols_ != o.Rows_) throw std::invalid_argument("Matrix dim mismatch (*)");
    Matrix r(Rows_, o.Cols_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < o.Cols_; ++j)
            for (size_t k = 0; k < Cols_; ++k)
                r(i, j) += (*this)(i, k) * o(k, j);
    return r;
}

Matrix Matrix::Transpose() const {
    Matrix r(Cols_, Rows_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r(j, i) = (*this)(i, j);
    return r;
}

Vector Matrix::operator*(const Vector& v) const {
    if (Cols_ != v.Size()) throw std::invalid_argument("Matrix-vector dim mismatch (*)");
    Vector r(Rows_);
    for (size_t i = 0; i < Rows_; ++i)
        for (size_t j = 0; j < Cols_; ++j)
            r[i] += (*this)(i, j) * v[j];
    return r;
}

// ======================== LinearRegressionGD ========================
Vector LinearRegressionGD(const Matrix& x, const Vector& y,
                           double learningRate, size_t iterations) {
    size_t n = x.GetRows();
    size_t d = x.GetCols();
    Vector w(d);
    for (size_t iter = 0; iter < iterations; ++iter) {
        Vector pred = x * w;
        Vector error = pred + (y * (-1.0));
        Matrix xt = x.Transpose();
        Vector grad = xt * error;
        grad = grad * (2.0 / static_cast<double>(n));
        w = w + (grad * (-learningRate));
    }
    return w;
}

} // namespace Ml
