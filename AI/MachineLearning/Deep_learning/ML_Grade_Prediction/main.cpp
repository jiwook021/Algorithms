#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>

// Simple Vector class
class Vector {
private:
    std::vector<double> data;

public:
    // Default constructor
    Vector() : data() {}
    
    Vector(size_t size, double value = 0.0) : data(size, value) {}
    Vector(const std::vector<double>& Vec) : data(Vec) {}

    double& operator[](size_t Index) { return data[Index]; }
    const double& operator[](size_t Index) const { return data[Index]; }
    size_t size() const { return data.size(); }

    Vector operator+(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for addition");
        }
        
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] + Other[i];
        }
        return Result;
    }

    Vector operator-(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for subtraction");
        }
        
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] - Other[i];
        }
        return Result;
    }

    Vector operator*(double Scalar) const {
        Vector Result(size());
        for (size_t i = 0; i < size(); ++i) {
            Result[i] = data[i] * Scalar;
        }
        return Result;
    }

    double Dot(const Vector& Other) const {
        if (size() != Other.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        
        double Result = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            Result += data[i] * Other[i];
        }
        return Result;
    }

    double Mean() const {
        if (size() == 0) return 0.0;
        double Sum = 0.0;
        for (const auto& Val : data) {
            Sum += Val;
        }
        return Sum / size();
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

    // Pearson correlation coefficient with another vector
    double Correlation(const Vector& Other) const {
        if (size() != Other.size() || size() == 0) {
            throw std::invalid_argument("Vectors must have the same non-zero size");
        }

        double MeanX = Mean();
        double MeanY = Other.Mean();
        double SumXy = 0.0, SumX2 = 0.0, SumY2 = 0.0;

        for (size_t i = 0; i < size(); ++i) {
            double XDiff = data[i] - MeanX;
            double YDiff = Other[i] - MeanY;
            SumXy += XDiff * YDiff;
            SumX2 += XDiff * XDiff;
            SumY2 += YDiff * YDiff;
        }

        if (SumX2 == 0.0 || SumY2 == 0.0) {
            return 0.0;  // Avoid division by zero
        }

        return SumXy / std::sqrt(SumX2 * SumY2);
    }

    const std::vector<double>& GetData() const { return data; }
};

// Simple Matrix class
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t Rows;
    size_t Cols;

public:
    // Default constructor
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

    Vector GetCol(size_t Col) const {
        if (Col >= Cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        Vector Result(Rows);
        for (size_t i = 0; i < Rows; ++i) {
            Result[i] = data[i][Col];
        }
        return Result;
    }

    // Get a specific row as a Vector
    Vector GetRow(size_t Row) const {
        if (Row >= Rows) {
            throw std::out_of_range("Row index out of range");
        }
        
        return Vector(data[Row]);
    }
};

// Feature scaling (min-max scaling)
std::pair<Matrix, std::vector<std::pair<double, double>>> ScaleFeatures(const Matrix& X) {
    size_t NSamples = X.NumRows();
    size_t NFeatures = X.NumCols();
    
    // Find min and max values for each feature
    std::vector<std::pair<double, double>> MinMax(NFeatures);  // (min, max) pairs
    for (size_t j = 0; j < NFeatures; ++j) {
        double MinVal = std::numeric_limits<double>::max();
        double MaxVal = std::numeric_limits<double>::lowest();
        
        for (size_t i = 0; i < NSamples; ++i) {
            MinVal = std::min(MinVal, X[i][j]);
            MaxVal = std::max(MaxVal, X[i][j]);
        }
        
        MinMax[j] = {MinVal, MaxVal};
    }
    
    // Scale features
    Matrix X_scaled(NSamples, NFeatures);
    for (size_t i = 0; i < NSamples; ++i) {
        for (size_t j = 0; j < NFeatures; ++j) {
            double Range = MinMax[j].second - MinMax[j].first;
            if (Range > 0) {
                X_scaled[i][j] = (X[i][j] - MinMax[j].first) / Range;
            } else {
                X_scaled[i][j] = 0.5;  // Default value if min == max
            }
        }
    }
    
    return {X_scaled, MinMax};
}

// Unscale a value using the min-max values
double UnscaleValue(double ScaledValue, double MinVal, double MaxVal) {
    double Range = MaxVal - MinVal;
    if (Range > 0) {
        return ScaledValue * Range + MinVal;
    } else {
        return MinVal;
    }
}

// Scale a value using the min-max values
double ScaleValue(double value, double MinVal, double MaxVal) {
    double Range = MaxVal - MinVal;
    if (Range > 0) {
        return (value - MinVal) / Range;
    } else {
        return 0.5;  // Default value if min == max
    }
}

// Linear Regression with gradient descent
class LinearRegression {
private:
    Vector Weights;
    double Bias;
    double LearningRate;
    int MaxIterations;
    double Tol;
    bool Verbose;
    
    // For feature scaling
    std::vector<std::pair<double, double>> FeatureMinMax;
    double TargetMin;
    double TargetMax;
    bool UseScaling;

public:
    LinearRegression(double LearningRate = 0.001, int MaxIterations = 1000000, 
                     double Tol = 1e-6, bool Verbose = false, bool UseScaling = true)
        : LearningRate(LearningRate), MaxIterations(MaxIterations), 
          Tol(Tol), Verbose(Verbose), Bias(0.0), UseScaling(UseScaling),
          TargetMin(0.0), TargetMax(1.0) {}

    void Fit(const Matrix& X_orig, const Vector& YOrig) {
        if (X_orig.NumRows() != YOrig.size() || X_orig.NumRows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        // Feature scaling if enabled
        Matrix X;
        Vector y;
        
        if (UseScaling) {
            // Scale features
            auto [X_scaled, MinMax] = ScaleFeatures(X_orig);
            X = X_scaled;
            FeatureMinMax = MinMax;
            
            // Scale target
            TargetMin = *std::min_element(YOrig.GetData().begin(), YOrig.GetData().end());
            TargetMax = *std::max_element(YOrig.GetData().begin(), YOrig.GetData().end());
            
            double TargetRange = TargetMax - TargetMin;
            y = Vector(YOrig.size());
            for (size_t i = 0; i < YOrig.size(); ++i) {
                if (TargetRange > 0) {
                    y[i] = (YOrig[i] - TargetMin) / TargetRange;
                } else {
                    y[i] = 0.5;
                }
            }
        } else {
            X = X_orig;
            y = YOrig;
        }
        
        size_t NSamples = X.NumRows();
        size_t NFeatures = X.NumCols();

        // Initialize weights and bias
        Weights = Vector(NFeatures, 0.0);
        Bias = 0.0;

        double PrevLoss = std::numeric_limits<double>::max();
        double CurrLearningRate = LearningRate;
        
        // Adaptive momentum
        Vector Momentum(NFeatures, 0.0);
        double BiasMomentum = 0.0;
        double Beta = 0.9;  // Momentum factor

        for (int Iter = 0; Iter < MaxIterations; ++Iter) {
            // Compute predictions
            Vector YPred(NSamples);
            for (size_t i = 0; i < NSamples; ++i) {
                double Pred = Bias;
                for (size_t j = 0; j < NFeatures; ++j) {
                    Pred += X[i][j] * Weights[j];
                }
                YPred[i] = Pred;
            }

            // Compute loss (MSE)
            double Loss = 0.0;
            for (size_t i = 0; i < NSamples; ++i) {
                double Error = YPred[i] - y[i];
                Loss += Error * Error;
            }
            Loss /= NSamples;

            // Check for convergence
            if (std::abs(Loss - PrevLoss) < Tol) {
                if (Verbose) {
                    std::cout << "Converged at iteration " << Iter << " with loss " << Loss << std::endl;
                }
                break;
            }
            
            // Learning rate scheduling
            // Reduce learning rate if loss increases
            if (Loss > PrevLoss) {
                CurrLearningRate *= 0.5;
                if (Verbose) {
                    std::cout << "Reducing learning rate to " << CurrLearningRate << std::endl;
                }
            }
            
            PrevLoss = Loss;

            // Compute gradients
            Vector GradW(NFeatures, 0.0);
            double GradB = 0.0;

            for (size_t i = 0; i < NSamples; ++i) {
                double Error = YPred[i] - y[i];
                for (size_t j = 0; j < NFeatures; ++j) {
                    GradW[j] += Error * X[i][j];
                }
                GradB += Error;
            }

            // Scale gradients by number of samples
            for (size_t j = 0; j < NFeatures; ++j) {
                GradW[j] /= NSamples;
            }
            GradB /= NSamples;
            
            // Update with momentum
            for (size_t j = 0; j < NFeatures; ++j) {
                Momentum[j] = Beta * Momentum[j] + (1.0 - Beta) * GradW[j];
                Weights[j] -= CurrLearningRate * Momentum[j];
            }
            BiasMomentum = Beta * BiasMomentum + (1.0 - Beta) * GradB;
            Bias -= CurrLearningRate * BiasMomentum;
            
            // Debug output
            if (Verbose && (Iter % 1000 == 0 || Iter == MaxIterations - 1)) {
                std::cout << "Iteration " << Iter << ": loss = " << Loss << std::endl;
                std::cout << "  weights = [";
                for (size_t j = 0; j < NFeatures; ++j) {
                    std::cout << Weights[j];
                    if (j < NFeatures - 1) std::cout << ", ";
                }
                std::cout << "], bias = " << Bias << std::endl;
            }
        }
    }

    Vector Predict(const Matrix& X_orig) const {
        if (X_orig.NumCols() != Weights.size()) {
            throw std::invalid_argument("Input feature dimensions don't match model parameters");
        }
        
        // Scale input features if needed
        Matrix X;
        if (UseScaling && !FeatureMinMax.empty()) {
            size_t NSamples = X_orig.NumRows();
            size_t NFeatures = X_orig.NumCols();
            
            X = Matrix(NSamples, NFeatures);
            for (size_t i = 0; i < NSamples; ++i) {
                for (size_t j = 0; j < NFeatures; ++j) {
                    double Range = FeatureMinMax[j].second - FeatureMinMax[j].first;
                    if (Range > 0) {
                        X[i][j] = (X_orig[i][j] - FeatureMinMax[j].first) / Range;
                    } else {
                        X[i][j] = 0.5;
                    }
                }
            }
        } else {
            X = X_orig;
        }
        
        size_t NSamples = X.NumRows();
        Vector YPred(NSamples);

        // Make predictions
        for (size_t i = 0; i < NSamples; ++i) {
            double Pred = Bias;
            for (size_t j = 0; j < X.NumCols(); ++j) {
                Pred += X[i][j] * Weights[j];
            }
            YPred[i] = Pred;
        }
        
        // Unscale predictions if needed
        if (UseScaling) {
            double TargetRange = TargetMax - TargetMin;
            for (size_t i = 0; i < NSamples; ++i) {
                if (TargetRange > 0) {
                    YPred[i] = YPred[i] * TargetRange + TargetMin;
                } else {
                    YPred[i] = TargetMin;
                }
            }
        }

        return YPred;
    }

    Vector GetWeights() const {
        if (!UseScaling || FeatureMinMax.empty()) {
            return Weights;
        }
        
        // Transform weights to original scale
        Vector OrigWeights(Weights.size());
        double TargetRange = TargetMax - TargetMin;
        
        for (size_t j = 0; j < Weights.size(); ++j) {
            double FeatureRange = FeatureMinMax[j].second - FeatureMinMax[j].first;
            if (FeatureRange > 0 && TargetRange > 0) {
                OrigWeights[j] = Weights[j] * TargetRange / FeatureRange;
            } else {
                OrigWeights[j] = Weights[j];
            }
        }
        
        return OrigWeights;
    }

    double GetBias() const {
        if (!UseScaling) {
            return Bias;
        }
        
        // Transform bias to original scale
        double TargetRange = TargetMax - TargetMin;
        double UnscaledBias = Bias * TargetRange + TargetMin;
        
        // Adjust for feature scaling in the weights
        for (size_t j = 0; j < Weights.size(); ++j) {
            double FeatureMin = FeatureMinMax[j].first;
            double FeatureRange = FeatureMinMax[j].second - FeatureMin;
            
            if (FeatureRange > 0) {
                UnscaledBias -= Weights[j] * TargetRange * FeatureMin / FeatureRange;
            }
        }
        
        return UnscaledBias;
    }

    // Calculate R-squared
    double RSquared(const Matrix& X, const Vector& y) const {
        if (X.NumRows() != y.size() || X.NumRows() == 0) {
            throw std::invalid_argument("Invalid input data dimensions");
        }
        
        Vector YPred = Predict(X);
        double YMean = y.Mean();
        
        double SsTotal = 0.0;
        double SsResidual = 0.0;
        
        for (size_t i = 0; i < y.size(); ++i) {
            double DiffTotal = y[i] - YMean;
            double DiffResidual = y[i] - YPred[i];
            
            SsTotal += DiffTotal * DiffTotal;
            SsResidual += DiffResidual * DiffResidual;
        }
        
        if (SsTotal == 0.0) {
            return 0.0;  // Avoid division by zero
        }
        
        return 1.0 - (SsResidual / SsTotal);
    }
};

// Multiple Linear Regression model (identical interface to LinearRegression)
typedef LinearRegression MultipleLinearRegression;

// Standardize features (z-score normalization)
Matrix Standardize(const Matrix& X) {
    size_t NSamples = X.NumRows();
    size_t NFeatures = X.NumCols();
    
    Matrix X_std(NSamples, NFeatures);
    
    for (size_t j = 0; j < NFeatures; ++j) {
        Vector Feature = X.GetCol(j);
        double Mean = Feature.Mean();
        double StdDev = Feature.StdDev();
        
        for (size_t i = 0; i < NSamples; ++i) {
            if (StdDev > 0) {
                X_std[i][j] = (X[i][j] - Mean) / StdDev;
            } else {
                X_std[i][j] = 0.0;
            }
        }
    }
    
    return X_std;
}

// Split data into training and testing sets
std::tuple<Matrix, Matrix, Vector, Vector> TrainTestSplit(
    const Matrix& X, const Vector& y, double TestSize = 0.2) {
    
    if (X.NumRows() != y.size() || X.NumRows() == 0) {
        throw std::invalid_argument("Invalid input data dimensions");
    }
    
    size_t NSamples = X.NumRows();
    size_t NTest = static_cast<size_t>(NSamples * TestSize);
    
    // Ensure at least one test sample
    NTest = std::max(size_t(1), NTest);
    
    // Ensure at least one training sample
    size_t NTrain = NSamples - NTest;
    NTrain = std::max(size_t(1), NTrain);
    
    // Adjust n_test if necessary
    NTest = NSamples - NTrain;
    
    // Create indices and shuffle them
    std::vector<size_t> Indices(NSamples);
    for (size_t i = 0; i < NSamples; ++i) {
        Indices[i] = i;
    }
    
    std::random_device Rd;
    std::mt19937 g(Rd());
    std::shuffle(Indices.begin(), Indices.end(), g);
    
    // Split data
    Matrix X_train(NTrain, X.NumCols());
    Matrix X_test(NTest, X.NumCols());
    Vector YTrain(NTrain);
    Vector YTest(NTest);
    
    for (size_t i = 0; i < NTrain; ++i) {
        size_t Idx = Indices[i];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_train[i][j] = X[Idx][j];
        }
        YTrain[i] = y[Idx];
    }
    
    for (size_t i = 0; i < NTest; ++i) {
        size_t Idx = Indices[i + NTrain];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_test[i][j] = X[Idx][j];
        }
        YTest[i] = y[Idx];
    }
    
    return {X_train, X_test, YTrain, YTest};
}

// Calculate Mean Squared Error
double MeanSquaredError(const Vector& YTrue, const Vector& YPred) {
    if (YTrue.size() != YPred.size() || YTrue.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    double Sum = 0.0;
    for (size_t i = 0; i < YTrue.size(); ++i) {
        double Diff = YTrue[i] - YPred[i];
        Sum += Diff * Diff;
    }
    return Sum / YTrue.size();
}

int main() {
    std::cout << "===== IQ and Study Time Analysis with Machine Learning =====" << std::endl;
    int lowest_IQ = 70; 
    // Define the data directly in the code
    std::vector<std::vector<double>> StudentData = {
        // IQ,   StudyTime, Grade
        {105.0-lowest_IQ,  7.5,      85.0},
        {120.0-lowest_IQ,  9.0,      94.0},
        {95.0-lowest_IQ,   3.5,      70.0},
        {110.0-lowest_IQ,  5.0,      88.0},
        {130.0-lowest_IQ,  8.0,      96.0},
        {115.0-lowest_IQ,  6.5,      87.0},
        {98.0-lowest_IQ,   4.0,      72.0},
        {125.0-lowest_IQ,  7.0,      91.0},
        {100.0-lowest_IQ,  3.0,      68.0},
        {118.0-lowest_IQ,  8.5,      89.0},
        {90.0-lowest_IQ,   2.5,      65.0},
        {135.0-lowest_IQ,  9.5,      98.0},
        {122.0-lowest_IQ,  7.8,      90.0},
        {88.0-lowest_IQ,   2.0,      60.0},
        {103.0-lowest_IQ,  5.5,      79.0},
        {112.0-lowest_IQ,  6.8,      84.0},
        {96.0-lowest_IQ,   3.8,      73.0},
        {116.0-lowest_IQ,  7.2,      88.0}
    };
    
    // Extract features (X) and target (y) from the data
    size_t NSamples = StudentData.size();
    Matrix X(NSamples, 2);  // 2 features: IQ and Study Time
    Vector y(NSamples);     // Target: Grade
    
    for (size_t i = 0; i < NSamples; ++i) {
        X[i][0] = StudentData[i][0];  // IQ
        X[i][1] = StudentData[i][1];  // Study Time
        y[i] = StudentData[i][2];     // Grade
    }
    
    // Extract feature columns
    Vector IqValues = X.GetCol(0);
    Vector StudyTimes = X.GetCol(1);
    
    // Basic statistics
    std::cout << "\n===== Basic Statistics =====" << std::endl;
    std::cout << "Number of students: " << NSamples << std::endl;
    
    std::cout << "\nIQ Statistics:" << std::endl;
    std::cout << "  Mean: " << IqValues.Mean() + lowest_IQ<< std::endl;
    std::cout << "  Standard Deviation: " << IqValues.StdDev() << std::endl;
    
    std::cout << "\nStudy Time Statistics (hours/day):" << std::endl;
    std::cout << "  Mean: " << StudyTimes.Mean() << std::endl;
    std::cout << "  Standard Deviation: " << StudyTimes.StdDev() << std::endl;
    
    std::cout << "\nGrade Statistics:" << std::endl;
    std::cout << "  Mean: " << y.Mean() << std::endl;
    std::cout << "  Standard Deviation: " << y.StdDev() << std::endl;
    
    // Correlation analysis
    std::cout << "\n===== Correlation Analysis =====" << std::endl;
    double CorrIqGrade = IqValues.Correlation(y);
    double CorrStudyGrade = StudyTimes.Correlation(y);
    double CorrIqStudy = IqValues.Correlation(StudyTimes);
    
    std::cout << "Correlation between IQ and Grades: " << CorrIqGrade << std::endl;
    std::cout << "Correlation between Study Time and Grades: " << CorrStudyGrade << std::endl;
    std::cout << "Correlation between IQ and Study Time: " << CorrIqStudy << std::endl;
    
    // Split data into training and testing sets
    // Using a fixed seed for reproducibility
    std::random_device Rd;
    std::mt19937 g(42);  // Fixed seed
    
    // Create indices and shuffle them
    std::vector<size_t> Indices(NSamples);
    for (size_t i = 0; i < NSamples; ++i) {
        Indices[i] = i;
    }
    
    std::shuffle(Indices.begin(), Indices.end(), g);
    
    // Using 70% training, 30% testing
    size_t NTrain = static_cast<size_t>(NSamples * 0.7);
    size_t NTest = NSamples - NTrain;
    
    Matrix X_train(NTrain, X.NumCols());
    Matrix X_test(NTest, X.NumCols());
    Vector YTrain(NTrain);
    Vector YTest(NTest);
    
    for (size_t i = 0; i < NTrain; ++i) {
        size_t Idx = Indices[i];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_train[i][j] = X[Idx][j];
        }
        YTrain[i] = y[Idx];
    }
    
    for (size_t i = 0; i < NTest; ++i) {
        size_t Idx = Indices[i + NTrain];
        for (size_t j = 0; j < X.NumCols(); ++j) {
            X_test[i][j] = X[Idx][j];
        }
        YTest[i] = y[Idx];
    }
    
    std::cout << "\n===== Linear Regression Models =====" << std::endl;
    std::cout << "Training set size: " << X_train.NumRows() << " samples" << std::endl;
    std::cout << "Test set size: " << X_test.NumRows() << " samples" << std::endl;
    
    // Model 1: Grade = f(IQ)
    std::cout << "\nModel 1: Grade as a function of IQ" << std::endl;
    
    Matrix X_iq_train(X_train.NumRows(), 1);
    Matrix X_iq_test(X_test.NumRows(), 1);
    
    for (size_t i = 0; i < X_train.NumRows(); ++i) {
        X_iq_train[i][0] = X_train[i][0];  // IQ
    }
    
    for (size_t i = 0; i < X_test.NumRows(); ++i) {
        X_iq_test[i][0] = X_test[i][0];  // IQ
    }
    
    try {
        // Use a smaller learning rate and enable feature scaling
        LinearRegression Model1(0.05, 20000, 1e-6, false, true);
        Model1.Fit(X_iq_train, YTrain);
        
        Vector YPred1 = Model1.Predict(X_iq_test);
        double Mse1 = MeanSquaredError(YTest, YPred1);
        double R21 = Model1.RSquared(X_iq_test, YTest);
        
        std::cout << "  Coefficient (IQ): " << Model1.GetWeights()[0] << std::endl;
        std::cout << "  Intercept: " << Model1.GetBias() << std::endl;
        std::cout << "  Mean Squared Error: " << Mse1 << std::endl;
        std::cout << "  R-squared: " << R21 << std::endl;
        std::cout << "  Formula: Grade = " << Model1.GetWeights()[0] << " * IQ + " << Model1.GetBias() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 1: " << e.what() << std::endl;
    }
    
    // Model 2: Grade = f(Study Time)
    std::cout << "\nModel 2: Grade as a function of Study Time" << std::endl;
    
    Matrix X_study_train(X_train.NumRows(), 1);
    Matrix X_study_test(X_test.NumRows(), 1);
    
    for (size_t i = 0; i < X_train.NumRows(); ++i) {
        X_study_train[i][0] = X_train[i][1];  // Study Time
    }
    
    for (size_t i = 0; i < X_test.NumRows(); ++i) {
        X_study_test[i][0] = X_test[i][1];  // Study Time
    }
    
    try {
        LinearRegression Model2(0.05, 20000, 1e-6, false, true);
        Model2.Fit(X_study_train, YTrain);
        
        Vector YPred2 = Model2.Predict(X_study_test);
        double Mse2 = MeanSquaredError(YTest, YPred2);
        double R22 = Model2.RSquared(X_study_test, YTest);
        
        std::cout << "  Coefficient (Study Time): " << Model2.GetWeights()[0] << std::endl;
        std::cout << "  Intercept: " << Model2.GetBias() << std::endl;
        std::cout << "  Mean Squared Error: " << Mse2 << std::endl;
        std::cout << "  R-squared: " << R22 << std::endl;
        std::cout << "  Formula: Grade = " << Model2.GetWeights()[0] << " * StudyTime + " << Model2.GetBias() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 2: " << e.what() << std::endl;
    }
    
    // Model 3: Grade = f(IQ, Study Time)
    std::cout << "\nModel 3: Grade as a function of both IQ and Study Time" << std::endl;
    
    try {
        LinearRegression Model3(0.05, 20000, 1e-6, false, true);
        Model3.Fit(X_train, YTrain);
        
        Vector YPred3 = Model3.Predict(X_test);
        double Mse3 = MeanSquaredError(YTest, YPred3);
        double R23 = Model3.RSquared(X_test, YTest);
        
        Vector Coeffs = Model3.GetWeights();
        std::cout << "  Coefficient (IQ): " << Coeffs[0] << std::endl;
        std::cout << "  Coefficient (Study Time): " << Coeffs[1] << std::endl;
        std::cout << "  Intercept: " << Model3.GetBias() << std::endl;
        std::cout << "  Mean Squared Error: " << Mse3 << std::endl;
        std::cout << "  R-squared: " << R23 << std::endl;
        std::cout << "  Formula: Grade = " << Coeffs[0] << " * IQ + " 
                  << Coeffs[1] << " * StudyTime + " << Model3.GetBias() << std::endl;
        
        // Analysis of model performance
        std::cout << "\n===== Model Comparison =====" << std::endl;
        std::cout << "Model 1 (IQ only) - R-squared: ";
        try {
            LinearRegression Model1(0.05, 20000, 1e-6, false, true);
            Model1.Fit(X_iq_train, YTrain);
            std::cout << Model1.RSquared(X_iq_test, YTest) << std::endl;
        } catch(...) {
            std::cout << "Could not compute" << std::endl;
        }
        
        std::cout << "Model 2 (Study Time only) - R-squared: ";
        try {
            LinearRegression Model2(0.05, 20000, 1e-6, false, true);
            Model2.Fit(X_study_train, YTrain);
            std::cout << Model2.RSquared(X_study_test, YTest) << std::endl;
        } catch(...) {
            std::cout << "Could not compute" << std::endl;
        }
        
        std::cout << "Model 3 (IQ and Study Time) - R-squared: " << R23 << std::endl;
        
        // Standardized coefficients to compare feature importance
        Matrix X_std = Standardize(X);
        MultipleLinearRegression StdModel(0.05, 20000, 1e-6, false, true);
        StdModel.Fit(X_std, y);
        
        Vector StdCoeffs = StdModel.GetWeights();
        std::cout << "\n===== Feature Importance Analysis =====" << std::endl;
        std::cout << "Standardized Coefficients:" << std::endl;
        std::cout << "  IQ: " << StdCoeffs[0] << std::endl;
        std::cout << "  Study Time: " << StdCoeffs[1] << std::endl;
        
        // Calculate predicted grades for different IQ and study time combinations
        std::cout << "\n===== Grade Predictions for Different Student Profiles =====" << std::endl;
        
        struct StudentProfile {
            double Iq;
            double StudyTime;
            std::string Description;
        };
        
        std::vector<StudentProfile> Profiles = {
            {90.0-lowest_IQ, 2.0, "Low IQ, Low Study Time"},
            {90.0-lowest_IQ, 8.0, "Low IQ, High Study Time"},
            {110.0-lowest_IQ, 5.0, "Average IQ, Average Study Time"},
            {130.0-lowest_IQ, 2.0, "High IQ, Low Study Time"},
            {130.0-lowest_IQ, 8.0, "High IQ, High Study Time"}
        };
        
        Matrix ProfileMatrix(Profiles.size(), 2);
        for (size_t i = 0; i < Profiles.size(); ++i) {
            ProfileMatrix[i][0] = Profiles[i].Iq;
            ProfileMatrix[i][1] = Profiles[i].StudyTime;
        }
        
        Vector Predictions = Model3.Predict(ProfileMatrix);
        
        std::cout << std::setw(30) << std::left << "Student Profile" 
                  << std::setw(10) << std::left << "IQ" 
                  << std::setw(15) << std::left << "Study Time (h)" 
                  << std::setw(10) << std::left << "Grade" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (size_t i = 0; i < Profiles.size(); ++i) {
            std::cout << std::setw(30) << std::left << Profiles[i].Description 
                      << std::setw(10) << std::left << Profiles[i].Iq 
                      << std::setw(15) << std::left << Profiles[i].StudyTime 
                      << std::setw(10) << std::left << Predictions[i] << std::endl;
        }
        double CoeffIq = Coeffs[0];
        double CoeffStudy = Coeffs[1];
        double Intercept = Model3.GetBias();
        
       
        
        // Study time effect at different IQ levels
        std::cout << "\n===== Effect of Increasing Study Time by 1 Hour =====" << std::endl;
        
        double StudyTimeCoeff = Coeffs[1];
        std::cout << "Grade increase per additional hour of study: " << StudyTimeCoeff << " points" << std::endl;
        
        // Calculate "IQ equivalent" of study time
        std::cout << "\n===== 'IQ Equivalent' of Study Time =====" << std::endl;
        
        if (std::abs(CoeffIq) < 1e-10 || std::abs(CoeffStudy) < 1e-10) {
            std::cout << "Cannot calculate IQ equivalent (one or both coefficients are too small)" << std::endl;
        } else {
            double HoursPerIqPoint = CoeffIq / CoeffStudy;
            
            std::cout << "One IQ point is equivalent to " << HoursPerIqPoint << " hours of study time" << std::endl;
            std::cout << "5 hours of additional study time is equivalent to " << (5.0 * CoeffStudy / CoeffIq) << " IQ points" << std::endl;
        }
        
        // Interactive Grade Prediction Tool
        std::cout << "\n===== Interactive Grade Prediction Tool =====" << std::endl;
        std::cout << "Enter IQ and study time to predict a student's grade." << std::endl;
        
        char ContinuePrediction = 'y';
        while (ContinuePrediction == 'y' || ContinuePrediction == 'Y') {
            double InputIq, InputStudyTime;
            
            std::cout << "\nEnter student's IQ: ";
            std::cin >> InputIq;
            
            std::cout << "Enter daily study time (hours): ";
            std::cin >> InputStudyTime;
            
            // Create a single-row matrix for the input
            Matrix InputData(1, 2);
            InputData[0][0] = InputIq - lowest_IQ;  // Apply the same IQ adjustment used in training
            InputData[0][1] = InputStudyTime;
            
            // Predict using the trained model
            Vector PredictedGrade = Model3.Predict(InputData);
            
            std::cout << "\nStudent Profile:" << std::endl;
            std::cout << "  IQ: " << InputIq << std::endl;
            std::cout << "  Study Time: " << InputStudyTime << " hours/day" << std::endl;
            std::cout << "  Predicted Grade: " << PredictedGrade[0] << std::endl;
            
            // Optional: Show contribution of each factor
            double IqContribution = CoeffIq * (InputIq - lowest_IQ);
            double StudyContribution = CoeffStudy * InputStudyTime;
            
            std::cout << "\nContribution to grade:" << std::endl;
            std::cout << "  From IQ: " << IqContribution << " points" << std::endl;
            std::cout << "  From Study Time: " << StudyContribution << " points" << std::endl;
            std::cout << "  Base value: " << Intercept << " points" << std::endl;
            
            // Suggest improvement
            std::cout << "\nFor each additional hour of study, grade could improve by " 
                      << CoeffStudy << " points." << std::endl;
            
            std::cout << "\nPredict another grade? (y/n): ";
            std::cin >> ContinuePrediction;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Model 3: " << e.what() << std::endl;
    }
    
    return 0;
}

//https://claude.ai/chat/cc1104fe-952e-4899-90f8-c9bc8679fa56