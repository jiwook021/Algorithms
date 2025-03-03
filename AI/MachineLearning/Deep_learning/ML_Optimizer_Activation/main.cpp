#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>
#include <unordered_map>
#include <functional>
#include <numeric>
#include <cassert>
#include <set>

// ----- Vector Class -----
class Vector {
private:
    std::vector<double> data;
public:
    Vector() : data() {}
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& Vec) : data(Vec) {}
    double& operator[](size_t Index) { return data[Index]; }
    const double& operator[](size_t Index) const { return data[Index]; }
    size_t size() const { return data.size(); }
    
    Vector operator+(const Vector& Other) const {
        if (size() != Other.size())
            throw std::invalid_argument("Vectors must have the same size for addition");
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = data[i] + Other[i];
        return Result;
    }
    
    Vector operator-(const Vector& Other) const {
        if (size() != Other.size())
            throw std::invalid_argument("Vectors must have the same size for subtraction");
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = data[i] - Other[i];
        return Result;
    }
    
    Vector operator*(double Scalar) const {
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = data[i] * Scalar;
        return Result;
    }
    
    Vector ElementWiseMultiply(const Vector& Other) const {
        if (size() != Other.size())
            throw std::invalid_argument("Vectors must have the same size for element-wise multiplication");
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = data[i] * Other[i];
        return Result;
    }
    
    Vector ElementWiseDivide(const Vector& Other) const {
        if (size() != Other.size())
            throw std::invalid_argument("Vectors must have the same size for element-wise division");
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = (std::abs(Other[i]) < 1e-10) ? 0.0 : data[i] / Other[i];
        return Result;
    }
    
    Vector sqrt() const {
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i)
            Result[i] = std::sqrt(std::max(0.0, data[i]));
        return Result;
    }
    
    double Dot(const Vector& Other) const {
        if (size() != Other.size())
            throw std::invalid_argument("Vectors must have the same size for dot product");
        double Result = 0.0;
        for (size_t i = 0; i < size(); ++i)
            Result += data[i] * Other[i];
        return Result;
    }
    
    double Sum() const {
        double Result = 0.0;
        for (const auto& Val : data)
            Result += Val;
        return Result;
    }
    
    double Mean() const {
        return (size() == 0) ? 0.0 : Sum() / size();
    }
    
    double Variance() const {
        if (size() <= 1) return 0.0;
        double m = Mean();
        double SumSqDiff = 0.0;
        for (const auto& Val : data) {
            double Diff = Val - m;
            SumSqDiff += Diff * Diff;
        }
        return SumSqDiff / size();
    }
    
    double StdDev() const {
        return std::sqrt(Variance());
    }
    
    double Correlation(const Vector& Other) const {
        if (size() != Other.size() || size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double MeanX = Mean(), MeanY = Other.Mean();
        double SumXy = 0.0, SumX2 = 0.0, SumY2 = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            double XDiff = data[i] - MeanX;
            double YDiff = Other[i] - MeanY;
            SumXy += XDiff * YDiff;
            SumX2 += XDiff * XDiff;
            SumY2 += YDiff * YDiff;
        }
        if (SumX2 == 0.0 || SumY2 == 0.0)
            return 0.0;
        return SumXy / std::sqrt(SumX2 * SumY2);
    }
    
    const std::vector<double>& GetData() const { return data; }
    double max() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute max of empty vector");
        return *std::max_element(data.begin(), data.end());
    }
    
    double min() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute min of empty vector");
        return *std::min_element(data.begin(), data.end());
    }
    
    size_t Argmax() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute argmax of empty vector");
        return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    }
    
    size_t Argmin() const {
        if (size() == 0)
            throw std::invalid_argument("Cannot compute argmin of empty vector");
        return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
    }
    
    bool HasNan() const {
        for (const auto& Val : data)
            if (std::isnan(Val))
                return true;
        return false;
    }
    
    void Clip(double MinVal, double MaxVal) {
        for (auto& Val : data)
            Val = std::max(MinVal, std::min(MaxVal, Val));
    }
};

// ----- Matrix Class -----
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t Rows;
    size_t Cols;
public:
    Matrix() : data(), Rows(0), Cols(0) {}
    Matrix(size_t Rows, size_t Cols, double value = 0.0)
        : data(Rows, std::vector<double>(Cols, value)), Rows(Rows), Cols(Cols) {}
    Matrix(const std::vector<std::vector<double>>& mat) {
        Rows = mat.size();
        Cols = Rows > 0 ? mat[0].size() : 0;
        data = mat;
    }
    std::vector<double>& operator[](size_t Row) { return data[Row]; }
    const std::vector<double>& operator[](size_t Row) const { return data[Row]; }
    size_t NumRows() const { return Rows; }
    size_t NumCols() const { return Cols; }
    
    Matrix operator+(const Matrix& Other) const {
        if (Rows != Other.Rows || Cols != Other.Cols)
            throw std::invalid_argument("Matrix dimensions must match for addition");
        Matrix Result(Rows, Cols);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                Result[i][j] = data[i][j] + Other[i][j];
        return Result;
    }
    
    Matrix operator-(const Matrix& Other) const {
        if (Rows != Other.Rows || Cols != Other.Cols)
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        Matrix Result(Rows, Cols);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                Result[i][j] = data[i][j] - Other[i][j];
        return Result;
    }
    
    Matrix operator*(double Scalar) const {
        Matrix Result(Rows, Cols);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                Result[i][j] = data[i][j] * Scalar;
        return Result;
    }
    
    Matrix operator*(const Matrix& Other) const {
        if (Cols != Other.Rows)
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        Matrix Result(Rows, Other.Cols, 0.0);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Other.Cols; ++j)
                for (size_t k = 0; k < Cols; ++k)
                    Result[i][j] += data[i][k] * Other[k][j];
        return Result;
    }
    
    Vector operator*(const Vector& Vec) const {
        if (Cols != Vec.size())
            throw std::invalid_argument("Matrix-vector dimensions incompatible for multiplication");
        Vector Result(Rows, 0.0);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                Result[i] += data[i][j] * Vec[j];
        return Result;
    }
    
    Vector GetCol(size_t Col) const {
        if (Col >= Cols)
            throw std::out_of_range("Column index out of range");
        Vector Result(Rows);
        for (size_t i = 0; i < Rows; ++i)
            Result[i] = data[i][Col];
        return Result;
    }
    
    Vector GetRow(size_t Row) const {
        if (Row >= Rows)
            throw std::out_of_range("Row index out of range");
        return Vector(data[Row]);
    }
    
    Matrix Transpose() const {
        Matrix Result(Cols, Rows);
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                Result[j][i] = data[i][j];
        return Result;
    }
    
    bool HasNan() const {
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                if (std::isnan(data[i][j]))
                    return true;
        return false;
    }
};

// ----- Activation Functions -----
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double Compute(double x) const = 0;
    virtual Vector Compute(const Vector& x) const = 0;
    virtual double Derivative(double x) const = 0;
    virtual Vector Derivative(const Vector& x) const = 0;
    virtual std::string Name() const = 0;
};

class Sigmoid : public ActivationFunction {
public:
    double Compute(double x) const override {
        x = std::max(-500.0, std::min(500.0, x));
        return 1.0 / (1.0 + std::exp(-x));
    }
    Vector Compute(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Compute(x[i]);
        return Result;
    }
    double Derivative(double x) const override {
        double s = Compute(x);
        return s * (1.0 - s);
    }
    Vector Derivative(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Derivative(x[i]);
        return Result;
    }
    std::string Name() const override { return "sigmoid"; }
};

class ReLU : public ActivationFunction {
public:
    double Compute(double x) const override { return std::max(0.0, x); }
    Vector Compute(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Compute(x[i]);
        return Result;
    }
    double Derivative(double x) const override { return x > 0.0 ? 1.0 : 0.0; }
    Vector Derivative(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Derivative(x[i]);
        return Result;
    }
    std::string Name() const override { return "relu"; }
};

class LeakyReLU : public ActivationFunction {
private:
    double Alpha;
public:
    LeakyReLU(double Alpha = 0.01) : Alpha(Alpha) {}
    double Compute(double x) const override { return x > 0.0 ? x : Alpha * x; }
    Vector Compute(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Compute(x[i]);
        return Result;
    }
    double Derivative(double x) const override { return x > 0.0 ? 1.0 : Alpha; }
    Vector Derivative(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Derivative(x[i]);
        return Result;
    }
    std::string Name() const override { return "leaky_relu"; }
};

class Tanh : public ActivationFunction {
public:
    double Compute(double x) const override {
        x = std::max(-500.0, std::min(500.0, x));
        return std::tanh(x);
    }
    Vector Compute(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Compute(x[i]);
        return Result;
    }
    double Derivative(double x) const override {
        double t = Compute(x);
        return 1.0 - t * t;
    }
    Vector Derivative(const Vector& x) const override {
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = Derivative(x[i]);
        return Result;
    }
    std::string Name() const override { return "tanh"; }
};

class Softmax : public ActivationFunction {
public:
    double Compute(double x) const override {
        throw std::invalid_argument("Softmax not defined for a single value");
    }
    Vector Compute(const Vector& x) const override {
        double MaxVal = x.max();
        Vector ExpValues(x.size());
        double SumExp = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double Val = std::max(-500.0, std::min(500.0, x[i] - MaxVal));
            ExpValues[i] = std::exp(Val);
            SumExp += ExpValues[i];
        }
        Vector Result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            Result[i] = (SumExp > 0.0) ? ExpValues[i] / SumExp : 1.0 / x.size();
        return Result;
    }
    double Derivative(double x) const override {
        throw std::invalid_argument("Softmax derivative not defined for a single value");
    }
    Vector Derivative(const Vector& x) const override {
        throw std::invalid_argument("Softmax derivative requires a matrix and is typically handled in the loss function");
    }
    std::string Name() const override { return "softmax"; }
};

class Identity : public ActivationFunction {
public:
    double Compute(double x) const override { return x; }
    Vector Compute(const Vector& x) const override { return x; }
    double Derivative(double x) const override { return 1.0; }
    Vector Derivative(const Vector& x) const override { return Vector(x.size(), 1.0); }
    std::string Name() const override { return "identity"; }
};

// ----- Loss Functions -----
class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual double Compute(const Vector& YTrue, const Vector& YPred) const = 0;
    virtual Vector Gradient(const Vector& YTrue, const Vector& YPred) const = 0;
    virtual std::string Name() const = 0;
};

class MSELoss : public LossFunction {
public:
    double Compute(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double Sum = 0.0;
        for (size_t i = 0; i < YTrue.size(); ++i) {
            double Diff = YTrue[i] - YPred[i];
            Sum += Diff * Diff;
        }
        return Sum / YTrue.size();
    }
    Vector Gradient(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector Result(YTrue.size());
        for (size_t i = 0; i < YTrue.size(); ++i)
            Result[i] = -2.0 * (YTrue[i] - YPred[i]) / YTrue.size();
        return Result;
    }
    std::string Name() const override { return "mse"; }
};

class MAELoss : public LossFunction {
public:
    double Compute(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double Sum = 0.0;
        for (size_t i = 0; i < YTrue.size(); ++i)
            Sum += std::abs(YTrue[i] - YPred[i]);
        return Sum / YTrue.size();
    }
    Vector Gradient(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector Result(YTrue.size());
        for (size_t i = 0; i < YTrue.size(); ++i) {
            double Diff = YPred[i] - YTrue[i];
            Result[i] = (Diff > 0.0 ? 1.0 : (Diff < 0.0 ? -1.0 : 0.0)) / YTrue.size();
        }
        return Result;
    }
    std::string Name() const override { return "mae"; }
};

class BinaryCrossEntropyLoss : public LossFunction {
private:
    const double Epsilon = 1e-10;
public:
    double Compute(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double Sum = 0.0;
        for (size_t i = 0; i < YTrue.size(); ++i) {
            double p = std::max(std::min(YPred[i], 1.0 - Epsilon), Epsilon);
            Sum += YTrue[i] * std::log(p) + (1.0 - YTrue[i]) * std::log(1.0 - p);
        }
        return -Sum / YTrue.size();
    }
    Vector Gradient(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector Result(YTrue.size());
        for (size_t i = 0; i < YTrue.size(); ++i) {
            double p = std::max(std::min(YPred[i], 1.0 - Epsilon), Epsilon);
            Result[i] = -(YTrue[i] / p - (1.0 - YTrue[i]) / (1.0 - p)) / YTrue.size();
        }
        return Result;
    }
    std::string Name() const override { return "binary_crossentropy"; }
};

class CategoricalCrossEntropyLoss : public LossFunction {
private:
    const double Epsilon = 1e-10;
public:
    double Compute(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        double Sum = 0.0;
        for (size_t i = 0; i < YTrue.size(); ++i) {
            if (YTrue[i] > 0.0) {
                double p = std::max(YPred[i], Epsilon);
                Sum += YTrue[i] * std::log(p);
            }
        }
        return -Sum;
    }
    Vector Gradient(const Vector& YTrue, const Vector& YPred) const override {
        if (YTrue.size() != YPred.size() || YTrue.size() == 0)
            throw std::invalid_argument("Vectors must have the same non-zero size");
        Vector Result(YTrue.size());
        for (size_t i = 0; i < YTrue.size(); ++i)
            Result[i] = YPred[i] - YTrue[i];
        return Result;
    }
    std::string Name() const override { return "categorical_crossentropy"; }
};

// ----- Optimizers -----
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void Update(Vector& Weights, const Vector& Gradients) = 0;
    virtual void UpdateBias(double& Bias, double Gradient) = 0;
    virtual void Reset() = 0;
    virtual std::string Name() const = 0;
    
    void ClipGradient(Vector& Gradients, double MaxNorm) {
        double Norm = 0.0;
        for (size_t i = 0; i < Gradients.size(); ++i)
            Norm += Gradients[i] * Gradients[i];
        Norm = std::sqrt(Norm);
        if (Norm > MaxNorm) {
            double Scale = MaxNorm / Norm;
            for (size_t i = 0; i < Gradients.size(); ++i)
                Gradients[i] *= Scale;
        }
    }
};

class SGDMomentum : public Optimizer {
private:
    double LearningRate;
    double Momentum;
    Vector Velocity;
    double BiasVelocity;
    double MaxGradNorm;
public:
    SGDMomentum(double LearningRate = 0.01, double Momentum = 0.9, double MaxGradNorm = 1.0)
        : LearningRate(LearningRate), Momentum(Momentum), BiasVelocity(0.0), MaxGradNorm(MaxGradNorm) {}
    
    void Update(Vector& Weights, const Vector& Gradients) override {
        if (Velocity.size() != Weights.size())
            Velocity = Vector(Weights.size(), 0.0);
        Vector ClippedGradients = Gradients;
        if (MaxGradNorm > 0.0)
            ClipGradient(ClippedGradients, MaxGradNorm);
        for (size_t i = 0; i < Weights.size(); ++i) {
            Velocity[i] = Momentum * Velocity[i] + LearningRate * ClippedGradients[i];
            Weights[i] -= Velocity[i];
        }
    }
    
    void UpdateBias(double& Bias, double Gradient) override {
        Gradient = std::max(-MaxGradNorm, std::min(MaxGradNorm, Gradient));
        BiasVelocity = Momentum * BiasVelocity + LearningRate * Gradient;
        Bias -= BiasVelocity;
    }
    
    void Reset() override {
        Velocity = Vector();
        BiasVelocity = 0.0;
    }
    
    std::string Name() const override { return "sgd_momentum"; }
};

class Adam : public Optimizer {
private:
    double LearningRate;
    double Beta1;
    double Beta2;
    double Epsilon;
    Vector m;
    Vector v;
    double MBias;
    double VBias;
    int t;
    double MaxGradNorm;
public:
    Adam(double LearningRate = 0.001, double Beta1 = 0.9, double Beta2 = 0.999,
         double Epsilon = 1e-8, double MaxGradNorm = 1.0)
         : LearningRate(LearningRate), Beta1(Beta1), Beta2(Beta2), Epsilon(Epsilon),
           MBias(0.0), VBias(0.0), t(0), MaxGradNorm(MaxGradNorm) {}
    
    void Update(Vector& Weights, const Vector& Gradients) override {
        if (m.size() != Weights.size()) {
            m = Vector(Weights.size(), 0.0);
            v = Vector(Weights.size(), 0.0);
        }
        t++;
        Vector ClippedGradients = Gradients;
        if (MaxGradNorm > 0.0)
            ClipGradient(ClippedGradients, MaxGradNorm);
        for (size_t i = 0; i < Weights.size(); ++i) {
            m[i] = Beta1 * m[i] + (1.0 - Beta1) * ClippedGradients[i];
            v[i] = Beta2 * v[i] + (1.0 - Beta2) * ClippedGradients[i] * ClippedGradients[i];
            double MHat = m[i] / (1.0 - std::pow(Beta1, t));
            double VHat = v[i] / (1.0 - std::pow(Beta2, t));
            Weights[i] -= LearningRate * MHat / (std::sqrt(VHat) + Epsilon);
        }
    }
    
    void UpdateBias(double& Bias, double Gradient) override {
        Gradient = std::max(-MaxGradNorm, std::min(MaxGradNorm, Gradient));
        MBias = Beta1 * MBias + (1.0 - Beta1) * Gradient;
        VBias = Beta2 * VBias + (1.0 - Beta2) * Gradient * Gradient;
        double MHat = MBias / (1.0 - std::pow(Beta1, t));
        double VHat = VBias / (1.0 - std::pow(Beta2, t));
        Bias -= LearningRate * MHat / (std::sqrt(VHat) + Epsilon);
    }
    
    void Reset() override {
        m = Vector();
        v = Vector();
        MBias = 0.0;
        VBias = 0.0;
        t = 0;
    }
    
    std::string Name() const override { return "adam"; }
};

class RMSProp : public Optimizer {
private:
    double LearningRate;
    double DecayRate;
    double Epsilon;
    Vector Cache;
    double BiasCache;
    double MaxGradNorm;
public:
    RMSProp(double LearningRate = 0.01, double DecayRate = 0.9,
            double Epsilon = 1e-8, double MaxGradNorm = 1.0)
            : LearningRate(LearningRate), DecayRate(DecayRate), Epsilon(Epsilon),
              BiasCache(0.0), MaxGradNorm(MaxGradNorm) {}
    
    void Update(Vector& Weights, const Vector& Gradients) override {
        if (Cache.size() != Weights.size())
            Cache = Vector(Weights.size(), 0.0);
        Vector ClippedGradients = Gradients;
        if (MaxGradNorm > 0.0)
            ClipGradient(ClippedGradients, MaxGradNorm);
        for (size_t i = 0; i < Weights.size(); ++i) {
            Cache[i] = DecayRate * Cache[i] + (1.0 - DecayRate) * ClippedGradients[i] * ClippedGradients[i];
            Weights[i] -= LearningRate * ClippedGradients[i] / (std::sqrt(Cache[i]) + Epsilon);
        }
    }
    
    void UpdateBias(double& Bias, double Gradient) override {
        Gradient = std::max(-MaxGradNorm, std::min(MaxGradNorm, Gradient));
        BiasCache = DecayRate * BiasCache + (1.0 - DecayRate) * Gradient * Gradient;
        Bias -= LearningRate * Gradient / (std::sqrt(BiasCache) + Epsilon);
    }
    
    void Reset() override {
        Cache = Vector();
        BiasCache = 0.0;
    }
    
    std::string Name() const override { return "rmsprop"; }
};

// ----- Preprocessing Tools -----
std::pair<Matrix, std::vector<std::pair<double, double>>> ScaleFeatures(const Matrix& X) {
    size_t NSamples = X.NumRows(), NFeatures = X.NumCols();
    std::vector<std::pair<double, double>> MinMax(NFeatures);
    for (size_t j = 0; j < NFeatures; ++j) {
        double MinVal = std::numeric_limits<double>::max();
        double MaxVal = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < NSamples; ++i) {
            MinVal = std::min(MinVal, X[i][j]);
            MaxVal = std::max(MaxVal, X[i][j]);
        }
        MinMax[j] = {MinVal, MaxVal};
    }
    Matrix X_scaled(NSamples, NFeatures);
    for (size_t i = 0; i < NSamples; ++i)
        for (size_t j = 0; j < NFeatures; ++j) {
            double Range = MinMax[j].second - MinMax[j].first;
            X_scaled[i][j] = (Range > 0) ? (X[i][j] - MinMax[j].first) / Range : 0.5;
        }
    return {X_scaled, MinMax};
}

double UnscaleValue(double ScaledValue, double MinVal, double MaxVal) {
    double Range = MaxVal - MinVal;
    return (Range > 0) ? ScaledValue * Range + MinVal : MinVal;
}

double ScaleValue(double value, double MinVal, double MaxVal) {
    double Range = MaxVal - MinVal;
    return (Range > 0) ? (value - MinVal) / Range : 0.5;
}

std::pair<Matrix, std::vector<std::pair<double, double>>> StandardizeFeatures(const Matrix& X) {
    size_t NSamples = X.NumRows(), NFeatures = X.NumCols();
    std::vector<std::pair<double, double>> MeanStd(NFeatures);
    for (size_t j = 0; j < NFeatures; ++j) {
        Vector Feature = X.GetCol(j);
        double Mean = Feature.Mean();
        double StdDev = Feature.StdDev();
        MeanStd[j] = {Mean, StdDev};
    }
    Matrix X_std(NSamples, NFeatures);
    for (size_t i = 0; i < NSamples; ++i)
        for (size_t j = 0; j < NFeatures; ++j)
            X_std[i][j] = (MeanStd[j].second > 0) ? (X[i][j] - MeanStd[j].first) / MeanStd[j].second : 0.0;
    return {X_std, MeanStd};
}

double UnstandardizeValue(double StdValue, double Mean, double StdDev) {
    return (StdDev > 0) ? StdValue * StdDev + Mean : Mean;
}

Matrix OneHotEncode(const Vector& CategoricalFeature, size_t NumCategories = 0) {
    if (NumCategories == 0) {
        std::set<double> UniqueValues;
        for (size_t i = 0; i < CategoricalFeature.size(); ++i)
            UniqueValues.insert(CategoricalFeature[i]);
        NumCategories = UniqueValues.size();
    }
    std::unordered_map<double, size_t> ValueToIndex;
    size_t NextIndex = 0;
    for (size_t i = 0; i < CategoricalFeature.size(); ++i) {
        if (ValueToIndex.find(CategoricalFeature[i]) == ValueToIndex.end()) {
            ValueToIndex[CategoricalFeature[i]] = NextIndex++;
            if (NextIndex > NumCategories)
                throw std::invalid_argument("More unique values than specified number of categories");
        }
    }
    Matrix Encoded(CategoricalFeature.size(), NumCategories, 0.0);
    for (size_t i = 0; i < CategoricalFeature.size(); ++i) {
        size_t Index = ValueToIndex[CategoricalFeature[i]];
        Encoded[i][Index] = 1.0;
    }
    return Encoded;
}

std::tuple<Matrix, Matrix, Vector, Vector> TrainTestSplit(const Matrix& X, const Vector& y,
                                                             double TestSize = 0.2, unsigned int RandomSeed = 42) {
    if (X.NumRows() != y.size() || X.NumRows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    size_t NSamples = X.NumRows();
    size_t NTest = static_cast<size_t>(NSamples * TestSize);
    NTest = std::max(size_t(1), NTest);
    size_t NTrain = NSamples - NTest;
    NTrain = std::max(size_t(1), NTrain);
    NTest = NSamples - NTrain;
    std::vector<size_t> Indices(NSamples);
    for (size_t i = 0; i < NSamples; ++i)
        Indices[i] = i;
    std::mt19937 g(RandomSeed);
    std::shuffle(Indices.begin(), Indices.end(), g);
    Matrix X_train(NTrain, X.NumCols());
    Matrix X_test(NTest, X.NumCols());
    Vector YTrain(NTrain);
    Vector YTest(NTest);
    for (size_t i = 0; i < NTrain; ++i) {
        size_t Idx = Indices[i];
        for (size_t j = 0; j < X.NumCols(); ++j)
            X_train[i][j] = X[Idx][j];
        YTrain[i] = y[Idx];
    }
    for (size_t i = 0; i < NTest; ++i) {
        size_t Idx = Indices[i + NTrain];
        for (size_t j = 0; j < X.NumCols(); ++j)
            X_test[i][j] = X[Idx][j];
        YTest[i] = y[Idx];
    }
    return {X_train, X_test, YTrain, YTest};
}

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> KFoldIndices(size_t NSamples, size_t k, unsigned int RandomSeed = 42) {
    std::vector<size_t> Indices(NSamples);
    for (size_t i = 0; i < NSamples; ++i)
        Indices[i] = i;
    std::mt19937 g(RandomSeed);
    std::shuffle(Indices.begin(), Indices.end(), g);
    std::vector<size_t> FoldSizes(k, NSamples / k);
    for (size_t i = 0; i < NSamples % k; ++i)
        FoldSizes[i]++;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> Splits;
    size_t Start = 0;
    for (size_t i = 0; i < k; ++i) {
        size_t FoldSize = FoldSizes[i];
        std::vector<size_t> ValIndices(Indices.begin() + Start, Indices.begin() + Start + FoldSize);
        std::vector<size_t> TrainIndices;
        for (size_t j = 0; j < NSamples; ++j)
            if (j < Start || j >= Start + FoldSize)
                TrainIndices.push_back(Indices[j]);
        Splits.push_back({TrainIndices, ValIndices});
        Start += FoldSize;
    }
    return Splits;
}

template<typename ModelType>
std::vector<double> CrossValidate(const Matrix& X, const Vector& y, size_t k,
                                   const std::function<ModelType()>& ModelFactory,
                                   const std::function<double(const ModelType&, const Matrix&, const Vector&)>& ScoreFunc,
                                   unsigned int RandomSeed = 42) {
    if (X.NumRows() != y.size() || X.NumRows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    size_t NSamples = X.NumRows();
    auto FoldIndices = KFoldIndices(NSamples, k, RandomSeed);
    std::vector<double> Scores;
    for (size_t i = 0; i < k; ++i) {
        const auto& TrainIndices = FoldIndices[i].first;
        const auto& ValIndices = FoldIndices[i].second;
        Matrix X_train(TrainIndices.size(), X.NumCols());
        Vector YTrain(TrainIndices.size());
        Matrix X_val(ValIndices.size(), X.NumCols());
        Vector YVal(ValIndices.size());
        for (size_t j = 0; j < TrainIndices.size(); ++j) {
            size_t Idx = TrainIndices[j];
            for (size_t c = 0; c < X.NumCols(); ++c)
                X_train[j][c] = X[Idx][c];
            YTrain[j] = y[Idx];
        }
        for (size_t j = 0; j < ValIndices.size(); ++j) {
            size_t Idx = ValIndices[j];
            for (size_t c = 0; c < X.NumCols(); ++c)
                X_val[j][c] = X[Idx][c];
            YVal[j] = y[Idx];
        }
        ModelType Model = ModelFactory();
        Model.Fit(X_train, YTrain);
        double Score = ScoreFunc(Model, X_val, YVal);
        Scores.push_back(Score);
    }
    return Scores;
}

template<typename ModelType, typename ParamType>
std::pair<ParamType, double> GridSearchCv(const Matrix& X, const Vector& y,
                                            const std::vector<ParamType>& ParamGrid,
                                            const std::function<ModelType(const ParamType&)>& ModelFactory,
                                            const std::function<double(const ModelType&, const Matrix&, const Vector&)>& ScoreFunc,
                                            size_t k = 5, unsigned int RandomSeed = 42) {
    if (X.NumRows() != y.size() || X.NumRows() == 0)
        throw std::invalid_argument("Invalid input data dimensions");
    if (ParamGrid.empty())
        throw std::invalid_argument("Parameter grid cannot be empty");
    double BestScore = -std::numeric_limits<double>::max();
    ParamType BestParams = ParamGrid[0];
    for (const auto& Params : ParamGrid) {
        auto FixedModelFactory = [&]() { return ModelFactory(Params); };
        auto Scores = CrossValidate<ModelType>(X, y, k, FixedModelFactory, ScoreFunc, RandomSeed);
        double MeanScore = 0.0;
        for (const auto& Score : Scores)
            MeanScore += Score;
        MeanScore /= Scores.size();
        if (MeanScore > BestScore) {
            BestScore = MeanScore;
            BestParams = Params;
        }
    }
    return {BestParams, BestScore};
}

// ----- Metrics -----
double MeanSquaredError(const Vector& YTrue, const Vector& YPred) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double Sum = 0.0;
    for (size_t i = 0; i < YTrue.size(); ++i) {
        double Diff = YTrue[i] - YPred[i];
        Sum += Diff * Diff;
    }
    return Sum / YTrue.size();
}

double MeanAbsoluteError(const Vector& YTrue, const Vector& YPred) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double Sum = 0.0;
    for (size_t i = 0; i < YTrue.size(); ++i)
        Sum += std::abs(YTrue[i] - YPred[i]);
    return Sum / YTrue.size();
}

double RSquared(const Vector& YTrue, const Vector& YPred) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    double YMean = YTrue.Mean();
    double SsTotal = 0.0, SsResidual = 0.0;
    for (size_t i = 0; i < YTrue.size(); ++i) {
        double DiffTotal = YTrue[i] - YMean;
        double DiffResidual = YTrue[i] - YPred[i];
        SsTotal += DiffTotal * DiffTotal;
        SsResidual += DiffResidual * DiffResidual;
    }
    return (SsTotal < 1e-10) ? 0.0 : 1.0 - (SsResidual / SsTotal);
}

double Accuracy(const Vector& YTrue, const Vector& YPred, double threshold = 0.5) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0)
        throw std::invalid_argument("Vectors must have the same non-zero size");
    size_t Correct = 0;
    for (size_t i = 0; i < YTrue.size(); ++i) {
        bool TrueClass = YTrue[i] > threshold;
        bool PredClass = YPred[i] > threshold;
        if (TrueClass == PredClass)
            Correct++;
    }
    return static_cast<double>(Correct) / YTrue.size();
}

// ----- Neural Network Layer -----
class Layer {
public:
    virtual ~Layer() = default;
    virtual Vector forward(const Vector& Input) = 0;
    virtual Vector Backward(const Vector& GradOutput) = 0;
    virtual void Update(Optimizer& Optimizer) = 0;
    virtual void Reset() = 0;
};

class DenseLayer : public Layer {
private:
    size_t InputSize;
    size_t OutputSize;
    Matrix Weights;
    Vector Biases;
    Vector Input;
    std::shared_ptr<ActivationFunction> Activation;
    Vector OutputBeforeActivation;
    Vector WeightsGradient;
public:
    DenseLayer(size_t InputSize, size_t OutputSize, 
               std::shared_ptr<ActivationFunction> Activation = std::make_shared<Identity>())
        : InputSize(InputSize), OutputSize(OutputSize),
          Weights(OutputSize, InputSize), Biases(OutputSize, 0.0),
          Activation(Activation), WeightsGradient(OutputSize * InputSize) {
        std::random_device Rd;
        std::mt19937 Gen(Rd());
        double Limit = std::sqrt(6.0 / (InputSize + OutputSize));
        std::uniform_real_distribution<double> Dis(-Limit, Limit);
        for (size_t i = 0; i < OutputSize; ++i)
            for (size_t j = 0; j < InputSize; ++j)
                Weights[i][j] = Dis(Gen) * 0.1;
    }
    
    Vector forward(const Vector& Input) override {
        if (Input.HasNan())
            throw std::runtime_error("NaN detected in input to dense layer");
        this->Input = Input;
        OutputBeforeActivation = Vector(OutputSize, 0.0);
        for (size_t i = 0; i < OutputSize; ++i) {
            double Sum = Biases[i];
            for (size_t j = 0; j < InputSize; ++j)
                Sum += Weights[i][j] * Input[j];
            OutputBeforeActivation[i] = Sum;
        }
        Vector Activated = Activation->Compute(OutputBeforeActivation);
        if (Activated.HasNan())
            throw std::runtime_error("NaN detected in dense layer output after activation");
        return Activated;
    }
    
    Vector Backward(const Vector& GradOutput) override {
        if (GradOutput.HasNan())
            throw std::runtime_error("NaN detected in gradient output");
        Vector GradOutputBeforeActivation = GradOutput;
        if (Activation->Name() != "identity") {
            Vector ActivationDerivative = Activation->Derivative(OutputBeforeActivation);
            for (size_t i = 0; i < OutputSize; ++i)
                GradOutputBeforeActivation[i] *= ActivationDerivative[i];
        }
        Vector GradInput(InputSize, 0.0);
        for (size_t i = 0; i < InputSize; ++i)
            for (size_t j = 0; j < OutputSize; ++j)
                GradInput[i] += Weights[j][i] * GradOutputBeforeActivation[j];
        size_t Idx = 0;
        for (size_t i = 0; i < OutputSize; ++i)
            for (size_t j = 0; j < InputSize; ++j)
                WeightsGradient[Idx++] = GradOutputBeforeActivation[i] * Input[j];
        for (size_t i = 0; i < OutputSize; ++i)
            Biases[i] = GradOutputBeforeActivation[i];
        if (GradInput.HasNan())
            throw std::runtime_error("NaN detected in gradient input");
        return GradInput;
    }
    
    void Update(Optimizer& Optimizer) override {
        Vector FlatWeights(OutputSize * InputSize);
        size_t Idx = 0;
        for (size_t i = 0; i < OutputSize; ++i)
            for (size_t j = 0; j < InputSize; ++j)
                FlatWeights[Idx++] = Weights[i][j];
        if (FlatWeights.HasNan() || WeightsGradient.HasNan())
            throw std::runtime_error("NaN detected in weights or gradients");
        Optimizer.Update(FlatWeights, WeightsGradient);
        Idx = 0;
        for (size_t i = 0; i < OutputSize; ++i)
            for (size_t j = 0; j < InputSize; ++j)
                Weights[i][j] = FlatWeights[Idx++];
        Vector BiasGradients = Biases;
        for (size_t i = 0; i < OutputSize; ++i)
            Optimizer.UpdateBias(Biases[i], BiasGradients[i]);
    }
    
    void Reset() override {
        std::random_device Rd;
        std::mt19937 Gen(Rd());
        double Limit = std::sqrt(6.0 / (InputSize + OutputSize));
        std::uniform_real_distribution<double> Dis(-Limit, Limit);
        for (size_t i = 0; i < OutputSize; ++i) {
            for (size_t j = 0; j < InputSize; ++j)
                Weights[i][j] = Dis(Gen) * 0.1;
            Biases[i] = 0.0;
        }
    }
};

// ----- Neural Network Model -----
class NeuralNetwork {
private:
    std::vector<std::shared_ptr<Layer>> Layers;
    std::shared_ptr<LossFunction> LossFunc_;
    std::shared_ptr<Optimizer> Optimizer_;
    bool Verbose;
    double YMin;
    double YMax;
public:
    NeuralNetwork(std::shared_ptr<LossFunction> LossF = std::make_shared<MSELoss>(),
                  std::shared_ptr<Optimizer> Opt = std::make_shared<SGDMomentum>(),
                  bool Verbose = false)
        : LossFunc_(LossF), Optimizer_(Opt), Verbose(Verbose), YMin(0.0), YMax(1.0) {}
    
    void AddLayer(std::shared_ptr<Layer> Layer) {
        Layers.push_back(Layer);
    }
    
    Vector forward(const Vector& Input) const {
        Vector Output = Input;
        for (const auto& Layer : Layers) {
            try {
                Output = Layer->forward(Output);
            } catch (const std::exception& e) {
                std::cerr << "Error in forward pass: " << e.what() << std::endl;
                throw;
            }
        }
        return Output;
    }
    
    void Backward(const Vector& Input, const Vector& Target) {
        try {
            Vector Output = forward(Input);
            Vector GradOutput = LossFunc_->Gradient(Target, Output);
            for (auto It = Layers.rbegin(); It != Layers.rend(); ++It)
                GradOutput = (*It)->Backward(GradOutput);
        } catch (const std::exception& e) {
            std::cerr << "Error in backward pass: " << e.what() << std::endl;
            throw;
        }
    }
    
    void Update() {
        try {
            for (auto& Layer : Layers)
                Layer->Update(*Optimizer_);
        } catch (const std::exception& e) {
            std::cerr << "Error in update: " << e.what() << std::endl;
            throw;
        }
    }
    
    void Reset() {
        for (auto& Layer : Layers)
            Layer->reset();
        Optimizer_->reset();
    }
    
    void Fit(const Matrix& X, const Vector& y, size_t BatchSize = 32, size_t Epochs = 100) {
        size_t NSamples = X.NumRows();
        if (NSamples != y.size())
            throw std::invalid_argument("Number of samples must match number of targets");
        std::vector<size_t> Indices(NSamples);
        std::iota(Indices.begin(), Indices.end(), 0);
        YMin = y.min();
        YMax = y.max();
        double YRange = YMax - YMin;
        Vector YNormalized(NSamples);
        for (size_t i = 0; i < NSamples; ++i)
            YNormalized[i] = (YRange > 0) ? (y[i] - YMin) / YRange : 0.5;
        for (size_t Epoch = 0; Epoch < Epochs; ++Epoch) {
            std::random_device Rd;
            std::mt19937 g(Rd());
            std::shuffle(Indices.begin(), Indices.end(), g);
            double TotalLoss = 0.0;
            for (size_t i = 0; i < NSamples; i += BatchSize) {
                size_t BatchEnd = std::min(i + BatchSize, NSamples);
                for (size_t j = i; j < BatchEnd; ++j) {
                    size_t Idx = Indices[j];
                    Vector Input = X.GetRow(Idx);
                    Vector Target(1, YNormalized[Idx]);
                    try {
                        Backward(Input, Target);
                        Vector Output = forward(Input);
                        TotalLoss += LossFunc_->Compute(Target, Output);
                    } catch (const std::exception& e) {
                        std::cerr << "Error during training: " << e.what() << std::endl;
                        reset();
                        break;
                    }
                }
                try {
                    Update();
                } catch (const std::exception& e) {
                    std::cerr << "Error updating parameters: " << e.what() << std::endl;
                    reset();
                    break;
                }
            }
            double AvgLoss = TotalLoss / NSamples;
            if (Verbose && (Epoch % 10 == 0 || Epoch == Epochs - 1))
                std::cout << "Epoch " << Epoch << ": loss = " << AvgLoss << std::endl;
            if (AvgLoss < 1e-5) {
                if (Verbose)
                    std::cout << "Early stopping at epoch " << Epoch << " with loss " << AvgLoss << std::endl;
                break;
            }
            if (std::isnan(AvgLoss) || std::isinf(AvgLoss))
                reset();
        }
    }
    
    Vector Predict(const Matrix& X) const {
        size_t NSamples = X.NumRows();
        Vector Predictions(NSamples);
        try {
            for (size_t i = 0; i < NSamples; ++i) {
                Vector Input = X.GetRow(i);
                Vector Output = forward(Input);
                Predictions[i] = (YMax > YMin) ? Output[0] * (YMax - YMin) + YMin : Output[0];
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
            throw;
        }
        return Predictions;
    }
    
    double Evaluate(const Matrix& X, const Vector& y) const {
        Vector Predictions = Predict(X);
        return LossFunc_->Compute(y, Predictions);
    }
    
    double RSquared(const Matrix& X, const Vector& y) const {
        Vector Predictions = Predict(X);
        double YMean = y.Mean();
        double SsTotal = 0.0, SsResidual = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double DiffTotal = y[i] - YMean;
            double DiffResidual = y[i] - Predictions[i];
            SsTotal += DiffTotal * DiffTotal;
            SsResidual += DiffResidual * DiffResidual;
        }
        return (SsTotal < 1e-10) ? 0.0 : 1.0 - (SsResidual / SsTotal);
    }
};

class LinearRegression {
private:
    double LearningRate;
    int MaxIterations;
    double Tol;
    bool Verbose;
    double Bias;
    std::vector<std::pair<double, double>> FeatureMinMax;
    double TargetMin;
    double TargetMax;
    bool UseScaling;
    std::shared_ptr<Optimizer> Optimizer_;
    Vector Weights;
public:
    LinearRegression(double LearningRate = 0.001, int MaxIterations = 1000000, 
                     double Tol = 1e-6, bool Verbose = false, bool UseScaling = true,
                     std::shared_ptr<Optimizer> Opt = std::make_shared<SGDMomentum>())
        : LearningRate(LearningRate), MaxIterations(MaxIterations), Tol(Tol), Verbose(Verbose),
          Bias(0.0), TargetMin(0.0), TargetMax(1.0), UseScaling(UseScaling), Optimizer_(Opt) {}
    
    void Fit(const Matrix& X_orig, const Vector& YOrig) {
        if (X_orig.NumRows() != YOrig.size() || X_orig.NumRows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Matrix X;
        Vector y;
        if (UseScaling) {
            auto [X_scaled, MinMax] = ScaleFeatures(X_orig);
            X = X_scaled;
            FeatureMinMax = MinMax;
            TargetMin = *std::min_element(YOrig.GetData().begin(), YOrig.GetData().end());
            TargetMax = *std::max_element(YOrig.GetData().begin(), YOrig.GetData().end());
            double TargetRange = TargetMax - TargetMin;
            y = Vector(YOrig.size());
            for (size_t i = 0; i < YOrig.size(); ++i)
                y[i] = (TargetRange > 0) ? (YOrig[i] - TargetMin) / TargetRange : 0.5;
        } else {
            X = X_orig;
            y = YOrig;
        }
        size_t NSamples = X.NumRows();
        size_t NFeatures = X.NumCols();
        Weights = Vector(NFeatures, 0.0);
        Bias = 0.0;
        Optimizer_->reset();
        double PrevLoss = std::numeric_limits<double>::max();
        for (int Iter = 0; Iter < MaxIterations; ++Iter) {
            Vector YPred(NSamples);
            for (size_t i = 0; i < NSamples; ++i) {
                double Pred = Bias;
                for (size_t j = 0; j < NFeatures; ++j)
                    Pred += X[i][j] * Weights[j];
                YPred[i] = Pred;
            }
            double Loss = 0.0;
            for (size_t i = 0; i < NSamples; ++i) {
                double Error = YPred[i] - y[i];
                Loss += Error * Error;
            }
            Loss /= NSamples;
            if (std::abs(Loss - PrevLoss) < Tol) {
                if (Verbose)
                    std::cout << "Converged at iteration " << Iter << " with loss " << Loss << std::endl;
                break;
            }
            PrevLoss = Loss;
            Vector GradW(NFeatures, 0.0);
            double GradB = 0.0;
            for (size_t i = 0; i < NSamples; ++i) {
                double Error = YPred[i] - y[i];
                for (size_t j = 0; j < NFeatures; ++j)
                    GradW[j] += Error * X[i][j];
                GradB += Error;
            }
            for (size_t j = 0; j < NFeatures; ++j)
                GradW[j] /= NSamples;
            GradB /= NSamples;
            Optimizer_->Update(Weights, GradW);
            Optimizer_->UpdateBias(Bias, GradB);
            if (Verbose && (Iter % 1000 == 0 || Iter == MaxIterations - 1)) {
                std::cout << "Iteration " << Iter << ": loss = " << Loss << std::endl;
                std::cout << "  weights = [";
                for (size_t j = 0; j < NFeatures; ++j) {
                    std::cout << Weights[j];
                    if (j < NFeatures - 1)
                        std::cout << ", ";
                }
                std::cout << "], bias = " << Bias << std::endl;
            }
        }
    }
    
    Vector Predict(const Matrix& X_orig) const {
        if (X_orig.NumCols() != Weights.size())
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        Matrix X;
        if (UseScaling && !FeatureMinMax.empty()) {
            size_t NSamples = X_orig.NumRows();
            size_t NFeatures = X_orig.NumCols();
            X = Matrix(NSamples, NFeatures);
            for (size_t i = 0; i < NSamples; ++i)
                for (size_t j = 0; j < NFeatures; ++j) {
                    double Range = FeatureMinMax[j].second - FeatureMinMax[j].first;
                    X[i][j] = (Range > 0) ? (X_orig[i][j] - FeatureMinMax[j].first) / Range : 0.5;
                }
        } else {
            X = X_orig;
        }
        size_t NSamples = X.NumRows();
        Vector YPred(NSamples);
        for (size_t i = 0; i < NSamples; ++i) {
            double Pred = Bias;
            for (size_t j = 0; j < X.NumCols(); ++j)
                Pred += X[i][j] * Weights[j];
            YPred[i] = Pred;
        }
        if (UseScaling) {
            double TargetRange = TargetMax - TargetMin;
            for (size_t i = 0; i < NSamples; ++i)
                YPred[i] = (TargetRange > 0) ? YPred[i] * TargetRange + TargetMin : TargetMin;
        }
        return YPred;
    }
    
    Vector GetWeights() const {
        if (!UseScaling || FeatureMinMax.empty())
            return Weights;
        Vector OrigWeights(Weights.size());
        double TargetRange = TargetMax - TargetMin;
        for (size_t j = 0; j < Weights.size(); ++j) {
            double FeatureRange = FeatureMinMax[j].second - FeatureMinMax[j].first;
            OrigWeights[j] = (FeatureRange > 0 && TargetRange > 0) ? Weights[j] * TargetRange / FeatureRange : Weights[j];
        }
        return OrigWeights;
    }
    
    double GetBias() const {
        if (!UseScaling)
            return Bias;
        double TargetRange = TargetMax - TargetMin;
        double UnscaledBias = Bias * TargetRange + TargetMin;
        for (size_t j = 0; j < Weights.size(); ++j) {
            double FeatureMin = FeatureMinMax[j].first;
            double FeatureRange = FeatureMinMax[j].second - FeatureMin;
            if (FeatureRange > 0)
                UnscaledBias -= Weights[j] * TargetRange * FeatureMin / FeatureRange;
        }
        return UnscaledBias;
    }
    
    double RSquared(const Matrix& X, const Vector& y) const {
        if (X.NumRows() != y.size() || X.NumRows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Vector YPred = Predict(X);
        double YMean = y.Mean();
        double SsTotal = 0.0, SsResidual = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double DiffTotal = y[i] - YMean;
            double DiffResidual = y[i] - YPred[i];
            SsTotal += DiffTotal * DiffTotal;
            SsResidual += DiffResidual * DiffResidual;
        }
        return (SsTotal == 0.0) ? 0.0 : 1.0 - (SsResidual / SsTotal);
    }
    
    std::shared_ptr<Optimizer> GetOptimizer() const { return Optimizer_; }
    void SetOptimizer(std::shared_ptr<Optimizer> NewOptimizer) { Optimizer_ = NewOptimizer; }
};

typedef LinearRegression MultipleLinearRegression;

class LogisticRegression {
private:
    double LearningRate;
    int MaxIterations;
    double Tol;
    bool Verbose;
    double Bias;
    Vector Weights;
    std::vector<std::pair<double, double>> FeatureMinMax;
    bool UseScaling;
    std::shared_ptr<Optimizer> Optimizer_;
    Sigmoid Sigmoid_;
public:
    LogisticRegression(double LearningRate = 0.001, int MaxIterations = 1000000, 
                       double Tol = 1e-6, bool Verbose = false, bool UseScaling = true,
                       std::shared_ptr<Optimizer> Opt = std::make_shared<SGDMomentum>())
        : LearningRate(LearningRate), MaxIterations(MaxIterations), Tol(Tol), Verbose(Verbose),
          Bias(0.0), UseScaling(UseScaling), Optimizer_(Opt) {}
    
    void Fit(const Matrix& X_orig, const Vector& YOrig) {
        if (X_orig.NumRows() != YOrig.size() || X_orig.NumRows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Matrix X;
        if (UseScaling) {
            auto [X_scaled, MinMax] = ScaleFeatures(X_orig);
            X = X_scaled;
            FeatureMinMax = MinMax;
        } else {
            X = X_orig;
        }
        size_t NSamples = X.NumRows();
        size_t NFeatures = X.NumCols();
        Weights = Vector(NFeatures, 0.0);
        Bias = 0.0;
        Optimizer_->reset();
        double PrevLoss = std::numeric_limits<double>::max();
        for (int Iter = 0; Iter < MaxIterations; ++Iter) {
            Vector z(NSamples);
            Vector YPred(NSamples);
            for (size_t i = 0; i < NSamples; ++i) {
                double Pred = Bias;
                for (size_t j = 0; j < NFeatures; ++j)
                    Pred += X[i][j] * Weights[j];
                z[i] = Pred;
                YPred[i] = Sigmoid_.Compute(Pred);
            }
            double Loss = 0.0;
            for (size_t i = 0; i < NSamples; ++i) {
                double y = YOrig[i];
                double p = std::max(std::min(YPred[i], 1.0 - 1e-15), 1e-15);
                Loss += y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
            }
            Loss = -Loss / NSamples;
            if (std::abs(Loss - PrevLoss) < Tol) {
                if (Verbose)
                    std::cout << "Converged at iteration " << Iter << " with loss " << Loss << std::endl;
                break;
            }
            PrevLoss = Loss;
            Vector GradW(NFeatures, 0.0);
            double GradB = 0.0;
            for (size_t i = 0; i < NSamples; ++i) {
                double Error = YPred[i] - YOrig[i];
                for (size_t j = 0; j < NFeatures; ++j)
                    GradW[j] += Error * X[i][j];
                GradB += Error;
            }
            for (size_t j = 0; j < NFeatures; ++j)
                GradW[j] /= NSamples;
            GradB /= NSamples;
            Optimizer_->Update(Weights, GradW);
            Optimizer_->UpdateBias(Bias, GradB);
            if (Verbose && (Iter % 1000 == 0 || Iter == MaxIterations - 1))
                std::cout << "Iteration " << Iter << ": loss = " << Loss << std::endl;
        }
    }
    
    Vector PredictProba(const Matrix& X_orig) const {
        if (X_orig.NumCols() != Weights.size())
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        Matrix X;
        if (UseScaling && !FeatureMinMax.empty()) {
            size_t NSamples = X_orig.NumRows();
            size_t NFeatures = X_orig.NumCols();
            X = Matrix(NSamples, NFeatures);
            for (size_t i = 0; i < NSamples; ++i)
                for (size_t j = 0; j < NFeatures; ++j) {
                    double Range = FeatureMinMax[j].second - FeatureMinMax[j].first;
                    X[i][j] = (Range > 0) ? (X_orig[i][j] - FeatureMinMax[j].first) / Range : 0.5;
                }
        } else {
            X = X_orig;
        }
        size_t NSamples = X.NumRows();
        Vector Probas(NSamples);
        for (size_t i = 0; i < NSamples; ++i) {
            double z = Bias;
            for (size_t j = 0; j < X.NumCols(); ++j)
                z += X[i][j] * Weights[j];
            Probas[i] = Sigmoid_.Compute(z);
        }
        return Probas;
    }
    
    Vector Predict(const Matrix& X, double threshold = 0.5) const {
        Vector Probas = PredictProba(X);
        size_t NSamples = Probas.size();
        Vector Predictions(NSamples);
        for (size_t i = 0; i < NSamples; ++i)
            Predictions[i] = (Probas[i] >= threshold) ? 1.0 : 0.0;
        return Predictions;
    }
    
    Vector GetWeights() const { return Weights; }
    double GetBias() const { return Bias; }
    
    double Accuracy(const Matrix& X, const Vector& y, double threshold = 0.5) const {
        if (X.NumRows() != y.size() || X.NumRows() == 0)
            throw std::invalid_argument("Invalid input data dimensions");
        Vector YPred = Predict(X, threshold);
        size_t Correct = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if ((y[i] >= threshold && YPred[i] >= threshold) || (y[i] < threshold && YPred[i] < threshold))
                Correct++;
        return static_cast<double>(Correct) / y.size();
    }
    
    std::shared_ptr<Optimizer> GetOptimizer() const { return Optimizer_; }
    void SetOptimizer(std::shared_ptr<Optimizer> NewOptimizer) { Optimizer_ = NewOptimizer; }
};

// ----- Main Function -----
int main() {
    try {
        std::cout << "===== Extended Machine Learning Library Demo =====" << std::endl;
        
        // Sample data: features = [IQ, Study Time], target = exam score
        Matrix X(10, 2);
        Vector y(10);
        X[0][0] = 105; X[0][1] = 7.5; y[0] = 85;
        X[1][0] = 120; X[1][1] = 9.0; y[1] = 94;
        X[2][0] = 95;  X[2][1] = 3.5; y[2] = 70;
        X[3][0] = 110; X[3][1] = 5.0; y[3] = 88;
        X[4][0] = 130; X[4][1] = 8.0; y[4] = 96;
        X[5][0] = 115; X[5][1] = 6.5; y[5] = 87;
        X[6][0] = 98;  X[6][1] = 4.0; y[6] = 72;
        X[7][0] = 125; X[7][1] = 7.0; y[7] = 91;
        X[8][0] = 100; X[8][1] = 3.0; y[8] = 68;
        X[9][0] = 118; X[9][1] = 8.5; y[9] = 89;
        
        std::cout << "\n===== Linear Regression with Different Optimizers =====" << std::endl;
        
        // Train using SGD with Momentum
        std::cout << "\nSGD with Momentum:" << std::endl;
        auto SgdOptimizer = std::make_shared<SGDMomentum>(0.01, 0.9);
        LinearRegression SgdModel(0.01, 1000, 1e-6, true, true, SgdOptimizer);
        SgdModel.Fit(X, y);
        Vector YPredSgd = SgdModel.Predict(X);
        double R2Sgd = SgdModel.RSquared(X, y);
        std::cout << "R-squared: " << R2Sgd << std::endl;
        
        // Train using Adam
        std::cout << "\nAdam:" << std::endl;
        auto AdamOptimizer = std::make_shared<Adam>(0.01);
        LinearRegression AdamModel(0.01, 1000, 1e-6, true, true, AdamOptimizer);
        AdamModel.Fit(X, y);
        Vector YPredAdam = AdamModel.Predict(X);
        double R2Adam = AdamModel.RSquared(X, y);
        std::cout << "R-squared: " << R2Adam << std::endl;
        
        // Train using RMSProp
        std::cout << "\nRMSProp:" << std::endl;
        auto RmspropOptimizer = std::make_shared<RMSProp>(0.01);
        LinearRegression RmspropModel(0.01, 1000, 1e-6, true, true, RmspropOptimizer);
        RmspropModel.Fit(X, y);
        Vector YPredRmsprop = RmspropModel.Predict(X);
        double R2Rmsprop = RmspropModel.RSquared(X, y);
        std::cout << "R-squared: " << R2Rmsprop << std::endl;
        
        // ----- Inference with Sample Input for Each Model -----
        // New sample: IQ = 115, Study Time = 8
        Matrix Sample(1, 2);
        Sample[0][0] = 115;
        Sample[0][1] = 8.0;
        std::cout << "\n===== Inference on Sample Input (IQ = 115, Study Time = 8) =====" << std::endl;
        Vector SamplePredSgd = SgdModel.Predict(Sample);
        std::cout << "SGD Model Prediction: " << SamplePredSgd[0] << std::endl;
        Vector SamplePredAdam = AdamModel.Predict(Sample);
        std::cout << "Adam Model Prediction: " << SamplePredAdam[0] << std::endl;
        Vector SamplePredRmsprop = RmspropModel.Predict(Sample);
        std::cout << "RMSProp Model Prediction: " << SamplePredRmsprop[0] << std::endl;
        
        std::cout << "\n===== Cross-Validation =====" << std::endl;
        auto ModelFactory = []() {
            auto Optimizer = std::make_shared<Adam>(0.01);
            return LinearRegression(0.01, 500, 1e-6, false, true, Optimizer);
        };
        auto ScoreFunc = [](const LinearRegression& Model, const Matrix& X, const Vector& y) {
            return Model.RSquared(X, y);
        };
        auto CvScores = CrossValidate<LinearRegression>(X, y, 5, ModelFactory, ScoreFunc);
        std::cout << "5-fold cross-validation R-squared scores:" << std::endl;
        double MeanScore = 0.0;
        for (size_t i = 0; i < CvScores.size(); ++i) {
            std::cout << "  Fold " << i + 1 << ": " << CvScores[i] << std::endl;
            MeanScore += CvScores[i];
        }
        MeanScore /= CvScores.size();
        std::cout << "Mean R-squared: " << MeanScore << std::endl;
        
        std::cout << "\n===== Grid Search =====" << std::endl;
        struct HyperParams {
            double LearningRate;
            std::string OptimizerType;
            HyperParams() : LearningRate(0.01), OptimizerType("sgd") {}
            HyperParams(double Lr, const std::string& Opt) : LearningRate(Lr), OptimizerType(Opt) {}
        };
        std::vector<HyperParams> ParamGrid = {
            HyperParams(0.001, "sgd"),
            HyperParams(0.01, "sgd"),
            HyperParams(0.001, "adam"),
            HyperParams(0.01, "adam"),
            HyperParams(0.001, "rmsprop"),
            HyperParams(0.01, "rmsprop")
        };
        auto ParamsModelFactory = [](const HyperParams& Params) {
            std::shared_ptr<Optimizer> Opt;
            if (Params.OptimizerType == "sgd")
                Opt = std::make_shared<SGDMomentum>(Params.LearningRate);
            else if (Params.OptimizerType == "adam")
                Opt = std::make_shared<Adam>(Params.LearningRate);
            else
                Opt = std::make_shared<RMSProp>(Params.LearningRate);
            return LinearRegression(Params.LearningRate, 500, 1e-6, false, true, Opt);
        };
        auto BestResult = GridSearchCv<LinearRegression, HyperParams>(X, y, ParamGrid, ParamsModelFactory, ScoreFunc, 5);
        std::cout << "Best parameters:" << std::endl;
        std::cout << "  Learning rate: " << BestResult.first.LearningRate << std::endl;
        std::cout << "  Optimizer: " << BestResult.first.OptimizerType << std::endl;
        std::cout << "Best R-squared: " << BestResult.second << std::endl;
        
        // Create final best model
        std::shared_ptr<Optimizer> BestOptimizer;
        if (BestResult.first.OptimizerType == "sgd")
            BestOptimizer = std::make_shared<SGDMomentum>(BestResult.first.LearningRate);
        else if (BestResult.first.OptimizerType == "adam")
            BestOptimizer = std::make_shared<Adam>(BestResult.first.LearningRate);
        else
            BestOptimizer = std::make_shared<RMSProp>(BestResult.first.LearningRate);
        
        LinearRegression BestModel(BestResult.first.LearningRate, 500, 1e-6, true, true, BestOptimizer);
        std::cout << "\n===== Final Model Training with Best Hyperparameters =====" << std::endl;
        BestModel.Fit(X, y);
        Vector YPredBest = BestModel.Predict(X);
        double FinalR2 = BestModel.RSquared(X, y);
        std::cout << "Final Model R-squared: " << FinalR2 << std::endl;
        
        // Inference with the final best model
        Vector SamplePredBest = BestModel.Predict(Sample);
        std::cout << "\nFinal Best Model Prediction on Sample Input (IQ = 115, Study Time = 8): " << SamplePredBest[0] << std::endl;
        
        std::cout << "\nDemo Completed Successfully." << std::endl;
    } catch (const std::exception& Ex) {
        std::cerr << "An exception occurred: " << Ex.what() << std::endl;
        return 1;
    }
    return 0;
}
