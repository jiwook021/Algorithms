#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <limits>

// Using the Vector and Matrix classes from the original code
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
        return Weights;
    }

    double GetBias() const {
        return Bias;
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
    std::cout << "===== Employee Performance Analysis with Machine Learning =====" << std::endl;
    
    // Define the employee performance data
    // Years of experience, Education level (1=Bachelor, 2=Master, 3=PhD), 
    // Number of completed projects, Average project delivery time (days),
    // Hours worked per week, Attendance rate (%), Score on aptitude test (0-100),
    // Annual performance evaluation score (0-100)
    std::vector<std::vector<double>> EmployeeData = {
        // YrsExp, EduLvl, Projects, DeliveryTime, HrsPerWeek, Attendance, AptitudeScore, PerfScore
        {2.0,     1.0,    4.0,      28.0,         38.0,       92.0,       78.0,          72.0},
        {5.0,     2.0,    8.0,      25.0,         42.0,       95.0,       82.0,          81.0},
        {1.0,     1.0,    2.0,      35.0,         35.0,       88.0,       70.0,          65.0},
        {8.0,     2.0,    12.0,     22.0,         45.0,       97.0,       85.0,          87.0},
        {3.0,     1.0,    6.0,      26.0,         40.0,       93.0,       75.0,          76.0},
        {10.0,    3.0,    15.0,     20.0,         48.0,       98.0,       90.0,          92.0},
        {4.0,     2.0,    7.0,      24.0,         41.0,       94.0,       80.0,          79.0},
        {0.5,     1.0,    1.0,      40.0,         35.0,       85.0,       65.0,          60.0},
        {6.0,     2.0,    10.0,     23.0,         44.0,       96.0,       84.0,          83.0},
        {12.0,    3.0,    18.0,     18.0,         50.0,       99.0,       95.0,          95.0},
        {7.0,     2.0,    11.0,     24.0,         43.0,       95.0,       83.0,          84.0},
        {2.5,     1.0,    5.0,      27.0,         39.0,       91.0,       76.0,          74.0},
        {9.0,     3.0,    14.0,     21.0,         47.0,       97.0,       88.0,          90.0},
        {1.5,     1.0,    3.0,      32.0,         37.0,       90.0,       72.0,          68.0},
        {4.5,     2.0,    8.0,      25.0,         42.0,       94.0,       81.0,          80.0},
        {11.0,    3.0,    17.0,     19.0,         49.0,       98.0,       92.0,          94.0},
        {3.5,     1.0,    6.0,      26.0,         40.0,       93.0,       77.0,          75.0},
        {8.5,     2.0,    13.0,     22.0,         46.0,       96.0,       86.0,          88.0},
        {6.5,     2.0,    10.0,     23.0,         44.0,       95.0,       84.0,          85.0},
        {2.0,     1.0,    3.0,      30.0,         38.0,       89.0,       74.0,          70.0}
    };
    
    // Extract features (X) and target (y) from the data
    size_t NSamples = EmployeeData.size();
    size_t NFeatures = 7;  // Number of features
    
    Matrix X(NSamples, NFeatures);
    Vector y(NSamples);
    
    for (size_t i = 0; i < NSamples; ++i) {
        for (size_t j = 0; j < NFeatures; ++j) {
            X[i][j] = EmployeeData[i][j];  // All features
        }
        y[i] = EmployeeData[i][7];  // Performance score
    }
    
    // Feature names for reference
    std::vector<std::string> FeatureNames = {
        "Years of Experience", 
        "Education Level", 
        "Completed Projects", 
        "Avg Delivery Time (days)", 
        "Hours per Week", 
        "Attendance Rate (%)", 
        "Aptitude Test Score"
    };
    
    // Basic statistics
    std::cout << "\n===== Basic Statistics =====" << std::endl;
    std::cout << "Number of employees in dataset: " << NSamples << std::endl;
    
    for (size_t j = 0; j < NFeatures; ++j) {
        Vector Feature = X.GetCol(j);
        std::cout << "\n" << FeatureNames[j] << " Statistics:" << std::endl;
        std::cout << "  Mean: " << Feature.Mean() << std::endl;
        std::cout << "  Standard Deviation: " << Feature.StdDev() << std::endl;
    }
    
    std::cout << "\nPerformance Score Statistics:" << std::endl;
    std::cout << "  Mean: " << y.Mean() << std::endl;
    std::cout << "  Standard Deviation: " << y.StdDev() << std::endl;
    
    // Correlation analysis
    std::cout << "\n===== Correlation with Performance Score =====" << std::endl;
    std::cout << std::setw(30) << std::left << "Feature" 
              << std::setw(12) << std::right << "Correlation" << std::endl;
    std::cout << std::string(42, '-') << std::endl;
    
    std::vector<double> Correlations;
    
    for (size_t j = 0; j < NFeatures; ++j) {
        Vector Feature = X.GetCol(j);
        double Corr = Feature.Correlation(y);
        Correlations.push_back(Corr);
        
        std::cout << std::setw(30) << std::left << FeatureNames[j]
                  << std::setw(12) << std::right << std::fixed << std::setprecision(4) << Corr << std::endl;
    }
    
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
    
    // Train a multiple linear regression model with all features
    std::cout << "\nModel: Performance Score as a function of all features" << std::endl;
    
    try {
        MultipleLinearRegression Model(0.05, 20000, 1e-6, false, true);
        Model.Fit(X_train, YTrain);
        
        Vector YPred = Model.Predict(X_test);
        double Mse = MeanSquaredError(YTest, YPred);
        double R2 = Model.RSquared(X_test, YTest);
        
        std::cout << "Model Performance:" << std::endl;
        std::cout << "  Mean Squared Error: " << Mse << std::endl;
        std::cout << "  R-squared: " << R2 << std::endl;
        
        // Feature importance (based on standardized coefficients)
        Matrix X_std = Standardize(X);
        MultipleLinearRegression StdModel(0.05, 20000, 1e-6, false, true);
        StdModel.Fit(X_std, y);
        
        Vector StdCoeffs = StdModel.GetWeights();
        
        std::cout << "\n===== Feature Importance Analysis =====" << std::endl;
        std::cout << std::setw(30) << std::left << "Feature" 
                  << std::setw(15) << std::right << "Coefficient" 
                  << std::setw(15) << std::right << "Std. Coefficient" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        Vector Coeffs = Model.GetWeights();
        for (size_t j = 0; j < NFeatures; ++j) {
            std::cout << std::setw(30) << std::left << FeatureNames[j]
                      << std::setw(15) << std::right << std::fixed << std::setprecision(4) << Coeffs[j]
                      << std::setw(15) << std::right << std::fixed << std::setprecision(4) << StdCoeffs[j] << std::endl;
        }
        std::cout << std::setw(30) << std::left << "Intercept"
                  << std::setw(15) << std::right << std::fixed << std::setprecision(4) << Model.GetBias() << std::endl;
        
        // Performance prediction for different profiles
        std::cout << "\n===== Performance Predictions for Employee Profiles =====" << std::endl;
        
        struct EmployeeProfile {
            std::vector<double> Features;
            std::string Description;
        };
        
        std::vector<EmployeeProfile> Profiles = {
            {{1.0, 1.0, 2.0, 30.0, 38.0, 90.0, 70.0}, "Junior, Bachelor's, Few Projects"},
            {{3.0, 1.0, 5.0, 25.0, 40.0, 93.0, 75.0}, "Mid-level, Bachelor's, Average Projects"},
            {{5.0, 2.0, 9.0, 22.0, 43.0, 95.0, 85.0}, "Experienced, Master's, Many Projects"},
            {{10.0, 3.0, 15.0, 20.0, 48.0, 98.0, 95.0}, "Senior, PhD, Numerous Projects"},
            {{2.0, 1.0, 3.0, 28.0, 45.0, 95.0, 80.0}, "Junior, Bachelor's, High Hours"},
            {{6.0, 2.0, 10.0, 21.0, 40.0, 97.0, 90.0}, "Experienced, Master's, High Aptitude"}
        };
        
        Matrix ProfileMatrix(Profiles.size(), NFeatures);
        for (size_t i = 0; i < Profiles.size(); ++i) {
            for (size_t j = 0; j < NFeatures; ++j) {
                ProfileMatrix[i][j] = Profiles[i].Features[j];
            }
        }
        
        Vector Predictions = Model.Predict(ProfileMatrix);
        
        std::cout << std::setw(40) << std::left << "Employee Profile" 
                  << std::setw(20) << std::right << "Predicted Score" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < Profiles.size(); ++i) {
            std::cout << std::setw(40) << std::left << Profiles[i].Description
                      << std::setw(20) << std::right << std::fixed << std::setprecision(2) 
                      << Predictions[i] << std::endl;
        }
        
        // Find the most influential positive and negative factors
        std::vector<std::pair<double, size_t>> FeatureImportance;
        for (size_t j = 0; j < NFeatures; ++j) {
            FeatureImportance.push_back({std::abs(StdCoeffs[j]), j});
        }
        
        // Sort by absolute value (descending)
        std::sort(FeatureImportance.begin(), FeatureImportance.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "\n===== Most Influential Factors for Performance =====" << std::endl;
        std::cout << "Factors listed in order of importance:" << std::endl;
        
        for (size_t i = 0; i < FeatureImportance.size(); ++i) {
            size_t j = FeatureImportance[i].second;
            std::string Effect = StdCoeffs[j] > 0 ? "Positive" : "Negative";
            std::string Explanation;
            
            // Custom explanations based on coefficient sign
            if (j == 0) { // Years of Experience
                Explanation = StdCoeffs[j] > 0 ? 
                    "More experienced employees tend to perform better" : 
                    "Years of experience may not translate to better performance";
            } else if (j == 1) { // Education Level
                Explanation = StdCoeffs[j] > 0 ? 
                    "Higher education correlates with better performance" : 
                    "Higher education doesn't necessarily lead to better performance";
            } else if (j == 2) { // Completed Projects
                Explanation = StdCoeffs[j] > 0 ? 
                    "Employees who complete more projects tend to score higher" : 
                    "Project quantity may be prioritized over quality";
            } else if (j == 3) { // Avg Delivery Time
                Explanation = StdCoeffs[j] > 0 ? 
                    "Longer delivery times correlate with better performance" : 
                    "Faster project completion correlates with better performance";
            } else if (j == 4) { // Hours per Week
                Explanation = StdCoeffs[j] > 0 ? 
                    "Employees who work more hours score higher" : 
                    "Working longer hours doesn't improve performance";
            } else if (j == 5) { // Attendance Rate
                Explanation = StdCoeffs[j] > 0 ? 
                    "Higher attendance correlates with better performance" : 
                    "Attendance doesn't significantly impact performance";
            } else if (j == 6) { // Aptitude Test Score
                Explanation = StdCoeffs[j] > 0 ? 
                    "Aptitude score strongly predicts job performance" : 
                    "Aptitude tests may not be relevant to job performance";
            }
            
            std::cout << i + 1 << ". " << FeatureNames[j] 
                      << " (Impact: " << Effect << ")" << std::endl;
            std::cout << "   Coefficient: " << StdCoeffs[j] << std::endl;
            std::cout << "   Interpretation: " << Explanation << std::endl;
        }
        
        // Recommendations based on analysis
        std::cout << "\n===== Recommendations for Performance Improvement =====" << std::endl;
        
        // Look at top positive factors
        std::vector<size_t> TopPositive;
        for (size_t j = 0; j < NFeatures; ++j) {
            if (StdCoeffs[j] > 0) {
                TopPositive.push_back(j);
            }
        }
        
        // Sort by coefficient value (descending)
        std::sort(TopPositive.begin(), TopPositive.end(),
            [&StdCoeffs](size_t a, size_t b) { return StdCoeffs[a] > StdCoeffs[b]; });
        
        if (!TopPositive.empty()) {
            std::cout << "Focus on improving these factors:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(3), TopPositive.size()); ++i) {
                size_t j = TopPositive[i];
                std::cout << "- " << FeatureNames[j] << std::endl;
            }
        }
        
        // Interactive prediction tool
        std::cout << "\n===== Interactive Performance Prediction Tool =====" << std::endl;
        std::cout << "Enter employee attributes to predict performance score." << std::endl;
        
        char ContinuePrediction = 'y';
        while (ContinuePrediction == 'y' || ContinuePrediction == 'Y') {
            Matrix Input(1, NFeatures);
            
            std::cout << "\nEnter years of experience: ";
            std::cin >> Input[0][0];
            
            std::cout << "Enter education level (1=Bachelor, 2=Master, 3=PhD): ";
            std::cin >> Input[0][1];
            
            std::cout << "Enter number of completed projects: ";
            std::cin >> Input[0][2];
            
            std::cout << "Enter average project delivery time (days): ";
            std::cin >> Input[0][3];
            
            std::cout << "Enter hours worked per week: ";
            std::cin >> Input[0][4];
            
            std::cout << "Enter attendance rate (%): ";
            std::cin >> Input[0][5];
            
            std::cout << "Enter aptitude test score (0-100): ";
            std::cin >> Input[0][6];
            
            Vector Pred = Model.Predict(Input);
            
            std::cout << "\nPredicted Performance Score: " << Pred[0] << std::endl;
            
            // Show contribution of each feature
            std::cout << "\nContribution breakdown:" << std::endl;
            double Base = Model.GetBias();
            std::cout << "Base value: " << Base << " points" << std::endl;
            
            for (size_t j = 0; j < NFeatures; ++j) {
                double Contrib = Coeffs[j] * Input[0][j];
                std::cout << FeatureNames[j] << ": " << Contrib << " points" << std::endl;
            }
            
            std::cout << "\nMake another prediction? (y/n): ";
            std::cin >> ContinuePrediction;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in model: " << e.what() << std::endl;
    }
    
    return 0;
}