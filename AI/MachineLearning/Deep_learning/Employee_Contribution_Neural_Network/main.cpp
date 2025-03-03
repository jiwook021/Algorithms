// Compute statistics from a dataset
void ComputeStatistics(const std::vector<EmployeeMetrics>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute statistics from empty dataset");
    }
    
    // Reset statistics
    for (const auto& Feature : FeatureNames) {
        MinValues[Feature] = std::numeric_limits<double>::max();
        MaxValues[Feature] = std::numeric_limits<double>::lowest();
        MeanValues[Feature] = 0.0;
        StdDevValues[Feature] = 0.0;
    }
    
    // Compute min, max, and mean
    for (const auto& Employee : data) {
        UpdateStatisticsFromEmployee(Employee, true);
    }
    
    // Compute standard deviation
    if (NormMethod == NormMethod::ZScore) {
        // Divide sums by count to get means
        for (const auto& Feature : FeatureNames) {
            MeanValues[Feature] /= data.size();
        }
        
        // Compute squared differences
        for (const auto& Employee : data) {
            std::vector<double> Values = EmployeeToVector(Employee);
            for (size_t i = 0; i < FeatureNames.size() && i < Values.size(); ++i) {
                double Diff = Values[i] - MeanValues[FeatureNames[i]];
                StdDevValues[FeatureNames[i]] += Diff * Diff;
            }
        }
        
        // Finalize standard deviations
        for (const auto& Feature : FeatureNames) {
            StdDevValues[Feature] = std::sqrt(StdDevValues[Feature] / data.size());
            // Avoid division by zero
            if (StdDevValues[Feature] < 1e-10) {
                StdDevValues[Feature] = 1.0;
            }
        }
    }
}

// Update statistics from a single employee record
void UpdateStatisticsFromEmployee(const EmployeeMetrics& Employee, bool ComputingMean = false) {
    std::vector<double> Values = EmployeeToVector(Employee);
    
    for (size_t i = 0; i < FeatureNames.size() && i < Values.size(); ++i) {
        const std::string& Feature = FeatureNames[i];
        double value = Values[i];
        
        // Update min and max
        MinValues[Feature] = std::min(MinValues[Feature], value);
        MaxValues[Feature] = std::max(MaxValues[Feature], value);
        
        // Update sum for mean calculation
        if (ComputingMean) {
            MeanValues[Feature] += value;
        }
    }
}

// Convert employee metrics to vector
std::vector<double> EmployeeToVector(const EmployeeMetrics& Employee) const {
    return {
        Employee.CodeCommits,
        Employee.LinesOfCode,
        Employee.CodeReviews,
        Employee.BugsFixed,
        Employee.DocumentationEdits,
        Employee.MeetingAttendance,
        Employee.TeamCollaboration,
        Employee.TechnicalDifficulty,
        Employee.ContributionScore
    };
}

// Normalize a single value
double NormalizeValue(double value, const std::string& Feature) const {
    if (NormMethod == NormMethod::MinMax) {
        // Min-Max normalization: (value - min) / (max - min)
        double min = MinValues.at(Feature);
        double max = MaxValues.at(Feature);
        if (std::abs(max - min) < 1e-10) {
            return 0.5; // Default to mid-range if no variation
        }
        return (value - min) / (max - min);
    } else {
        // Z-score normalization: (value - mean) / stdDev
        double Mean = MeanValues.at(Feature);
        double StdDev = StdDevValues.at(Feature);
        return (value - Mean) / StdDev;
    }
}

// Denormalize a value
double DenormalizeValue(double NormalizedValue, const std::string& Feature) const {
    if (NormMethod == NormMethod::MinMax) {
        // Min-Max denormalization: normalizedValue * (max - min) + min
        double min = MinValues.at(Feature);
        double max = MaxValues.at(Feature);
        return NormalizedValue * (max - min) + min;
    } else {
        // Z-score denormalization: normalizedValue * stdDev + mean
        double Mean = MeanValues.at(Feature);
        double StdDev = StdDevValues.at(Feature);
        return NormalizedValue * StdDev + Mean;
    }
}

// Normalize an employee metrics object
EmployeeMetrics NormalizeMetrics(const EmployeeMetrics& Employee) const {
    EmployeeMetrics Normalized = Employee;
    Normalized.CodeCommits = NormalizeValue(Employee.CodeCommits, "codeCommits");
    Normalized.LinesOfCode = NormalizeValue(Employee.LinesOfCode, "linesOfCode");
    Normalized.CodeReviews = NormalizeValue(Employee.CodeReviews, "codeReviews");
    Normalized.BugsFixed = NormalizeValue(Employee.BugsFixed, "bugsFixed");
    Normalized.DocumentationEdits = NormalizeValue(Employee.DocumentationEdits, "documentationEdits");
    Normalized.MeetingAttendance = NormalizeValue(Employee.MeetingAttendance, "meetingAttendance");
    Normalized.TeamCollaboration = NormalizeValue(Employee.TeamCollaboration, "teamCollaboration");
    Normalized.TechnicalDifficulty = NormalizeValue(Employee.TechnicalDifficulty, "technicalDifficulty");
    Normalized.ContributionScore = NormalizeValue(Employee.ContributionScore, "contributionScore");
    return Normalized;
}

// Denormalize an employee metrics object
EmployeeMetrics DenormalizeMetrics(const EmployeeMetrics& NormalizedEmployee) const {
    EmployeeMetrics Denormalized = NormalizedEmployee;
    Denormalized.CodeCommits = DenormalizeValue(NormalizedEmployee.CodeCommits, "codeCommits");
    Denormalized.LinesOfCode = DenormalizeValue(NormalizedEmployee.LinesOfCode, "linesOfCode");
    Denormalized.CodeReviews = DenormalizeValue(NormalizedEmployee.CodeReviews, "codeReviews");
    Denormalized.BugsFixed = DenormalizeValue(NormalizedEmployee.BugsFixed, "bugsFixed");
    Denormalized.DocumentationEdits = DenormalizeValue(NormalizedEmployee.DocumentationEdits, "documentationEdits");
    Denormalized.MeetingAttendance = DenormalizeValue(NormalizedEmployee.MeetingAttendance, "meetingAttendance");
    Denormalized.TeamCollaboration = DenormalizeValue(NormalizedEmployee.TeamCollaboration, "teamCollaboration");
    Denormalized.TechnicalDifficulty = DenormalizeValue(NormalizedEmployee.TechnicalDifficulty, "technicalDifficulty");
    Denormalized.ContributionScore = DenormalizeValue(NormalizedEmployee.ContributionScore, "contributionScore");
    return Denormalized;
}

// Prepare data for neural network training
void PrepareTrainingData(const std::vector<EmployeeMetrics>& data,
                        std::vector<std::vector<double>>& Inputs,
                        std::vector<std::vector<double>>& Targets) {
    // Compute statistics first
    ComputeStatistics(data);
    
    // Clear output vectors
    Inputs.clear();
    Targets.clear();
    
    // Normalize and separate inputs and targets
    for (const auto& Employee : data) {
        EmployeeMetrics Normalized = NormalizeMetrics(Employee);
        
        // Extract inputs
        Inputs.push_back(Normalized.ToInputVector());
        
        // Extract target (contribution score)
        Targets.push_back({Normalized.ContributionScore});
    }
}

// Split data into training and validation sets
void SplitData(const std::vector<EmployeeMetrics>& data, 
               std::vector<EmployeeMetrics>& TrainingData,
               std::vector<EmployeeMetrics>& ValidationData,
               double ValidationRatio = 0.2) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot split empty dataset");
    }
    
    // Clear output vectors
    TrainingData.clear();
    ValidationData.clear();
    
    // Create a copy and shuffle
    std::vector<EmployeeMetrics> ShuffledData = data;
    std::random_device Rd;
    std::mt19937 g(Rd());
    std::shuffle(ShuffledData.begin(), ShuffledData.end(), g);
    
    // Calculate split index
    size_t ValidationSize = static_cast<size_t>(std::round(data.size() * ValidationRatio));
    size_t TrainingSize = data.size() - ValidationSize;
    
    // Split data
    TrainingData.insert(TrainingData.end(), ShuffledData.begin(), ShuffledData.begin() + TrainingSize);
    ValidationData.insert(ValidationData.end(), ShuffledData.begin() + TrainingSize, ShuffledData.end());
}

// Save normalization parameters to a file
bool SaveParameters(const std::string& Filename) const {
    try {
        std::ofstream File(Filename);
        if (!File.IsOpen()) {
            return false;
        }
        
        // Write normalization method
        File << (NormMethod == NormMethod::MinMax ? "MinMax" : "ZScore") << "\n";
        
        // Write feature names
        for (const auto& Feature : FeatureNames) {
            File << Feature << ",";
        }
        File << "\n";
        
        // Write min values
        for (const auto& Feature : FeatureNames) {
            File << MinValues.at(Feature) << ",";
        }
        File << "\n";
        
        // Write max values
        for (const auto& Feature : FeatureNames) {
            File << MaxValues.at(Feature) << ",";
        }
        File << "\n";
        
        // Write mean values
        for (const auto& Feature : FeatureNames) {
            File << MeanValues.at(Feature) << ",";
        }
        File << "\n";
        
        // Write stdDev values
        for (const auto& Feature : FeatureNames) {
            File << StdDevValues.at(Feature) << ",";
        }
        File << "\n";
        
        File.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving normalization parameters: " << e.What() << std::endl;
        return false;
    }
}

// Load normalization parameters from a file
bool LoadParameters(const std::string& Filename) {
    try {
        std::ifstream File(Filename);
        if (!File.IsOpen()) {
            return false;
        }
        
        std::string Line;
        
        // Read normalization method
        std::getline(File, Line);
        NormMethod = (Line == "MinMax") ? NormMethod::MinMax : NormMethod::ZScore;
        
        // Read feature names
        std::getline(File, Line);
        std::istringstream FeatureStream(Line);
        std::string Feature;
        FeatureNames.clear();
        while (std::getline(FeatureStream, Feature, ',')) {
            if (!Feature.empty()) {
                FeatureNames.push_back(Feature);
            }
        }
        
        // Read min values
        std::getline(File, Line);
        std::istringstream MinStream(Line);
        std::string ValueStr;
        size_t FeatureIndex = 0;
        while (std::getline(MinStream, ValueStr, ',') && FeatureIndex < FeatureNames.size()) {
            if (!ValueStr.empty()) {
                MinValues[FeatureNames[FeatureIndex++]] = std::stod(ValueStr);
            }
        }
        
        // Read max values
        std::getline(File, Line);
        std::istringstream MaxStream(Line);
        FeatureIndex = 0;
        while (std::getline(MaxStream, ValueStr, ',') && FeatureIndex < FeatureNames.size()) {
            if (!ValueStr.empty()) {
                MaxValues[FeatureNames[FeatureIndex++]] = std::stod(ValueStr);
            }
        }
        
        // Read mean values
        std::getline(File, Line);
        std::istringstream MeanStream(Line);
        FeatureIndex = 0;
        while (std::getline(MeanStream, ValueStr, ',') && FeatureIndex < FeatureNames.size()) {
            if (!ValueStr.empty()) {
                MeanValues[FeatureNames[FeatureIndex++]] = std::stod(ValueStr);
            }
        }
        
        // Read stdDev values
        std::getline(File, Line);
        std::istringstream StdDevStream(Line);
        FeatureIndex = 0;
        while (std::getline(StdDevStream, ValueStr, ',') && FeatureIndex < FeatureNames.size()) {
            if (!ValueStr.empty()) {
                StdDevValues[FeatureNames[FeatureIndex++]] = std::stod(ValueStr);
            }
        }
        
        File.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading normalization parameters: " << e.What() << std::endl;
        return false;
    }
}
};

/**
* @brief Employee Contribution Predictor class that combines data processing and neural network
*/
class EmployeeContributionPredictor {
private:
std::unique_ptr<NeuralNetwork> NeuralNetwork;
std::unique_ptr<DataProcessor> DataProcessor;
bool IsModelTrained;

public:
// Constructor
EmployeeContributionPredictor() 
    : IsModelTrained(false) {
    // Create data processor with default settings
    DataProcessor = std::make_unique<DataProcessor>();
    
    // Create neural network with default architecture
    NeuralNetConfig Config;
    NeuralNetwork = std::make_unique<NeuralNetwork>(Config);
}

// Configure the neural network
void ConfigureNetwork(const NeuralNetConfig& Config) {
    NeuralNetwork = std::make_unique<NeuralNetwork>(Config);
    IsModelTrained = false;
}

// Train the predictor
double Train(const std::vector<EmployeeMetrics>& data, double ValidationRatio = 0.2) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot train with empty dataset");
    }
    
    // Split data into training and validation sets
    std::vector<EmployeeMetrics> TrainingData, ValidationData;
    DataProcessor->SplitData(data, TrainingData, ValidationData, ValidationRatio);
    
    // Prepare training data
    std::vector<std::vector<double>> TrainingInputs, TrainingTargets;
    DataProcessor->PrepareTrainingData(TrainingData, TrainingInputs, TrainingTargets);
    
    // Train the neural network
    NeuralNetwork->Train(TrainingInputs, TrainingTargets);
    
    // Prepare validation data
    std::vector<std::vector<double>> ValidationInputs, ValidationTargets;
    DataProcessor->PrepareTrainingData(ValidationData, ValidationInputs, ValidationTargets);
    
    // Validate the neural network
    double ValidationError = NeuralNetwork->Validate(ValidationInputs, ValidationTargets);
    
    IsModelTrained = true;
    return ValidationError;
}

// Predict contribution for a single employee
double PredictContribution(const EmployeeMetrics& Employee) const {
    if (!IsModelTrained) {
        throw std::runtime_error("Model not trained yet");
    }
    
    // Normalize metrics
    EmployeeMetrics NormalizedEmployee = DataProcessor->NormalizeMetrics(Employee);
    
    // Predict using normalized input vector
    std::vector<double> Inputs = NormalizedEmployee.ToInputVector();
    std::vector<double> Outputs = NeuralNetwork->FeedForward(Inputs);
    
    if (Outputs.empty()) {
        throw std::runtime_error("Neural network produced no output");
    }
    
    // Denormalize the prediction
    return DataProcessor->DenormalizeValue(Outputs[0], "contributionScore");
}

// Predict contributions for multiple employees
std::vector<double> PredictContributions(const std::vector<EmployeeMetrics>& Employees) const {
    std::vector<double> Predictions;
    Predictions.reserve(Employees.size());
    
    for (const auto& Employee : Employees) {
        Predictions.push_back(PredictContribution(Employee));
    }
    
    return Predictions;
}

// Save the model and normalization parameters
bool SaveModel(const std::string& ModelFile, const std::string& NormFile) const {
    return NeuralNetwork->SaveModel(ModelFile) && 
           DataProcessor->SaveParameters(NormFile);
}

// Load the model and normalization parameters
bool LoadModel(const std::string& ModelFile, const std::string& NormFile) {
    bool Success = NeuralNetwork->LoadModel(ModelFile) && 
                   DataProcessor->LoadParameters(NormFile);
    IsModelTrained = Success;
    return Success;
}

// Get training statistics
double GetTrainingError() const { return NeuralNetwork->GetTotalError(); }
size_t GetTrainingEpochs() const { return NeuralNetwork->GetTrainingEpochs(); }
double GetValidationAccuracy() const { return NeuralNetwork->GetValidationAccuracy(); }
};

} // namespace ML

/**
* @brief Enhanced CSV handling class for employee data
* This class specializes in reading and processing CSV files containing employee metrics
*/
class CSVHandler {
private:
bool HasHeader;
char Delimiter;
std::string DateFormat;

public:
// Constructor with options
CSVHandler(bool Header = true, char Delim = ',', std::string Format = "%Y-%m-%d")
    : HasHeader(Header), Delimiter(Delim), DateFormat(std::move(Format)) {}

// Read employee metrics from CSV file
std::vector<ML::EmployeeMetrics> ReadEmployeeMetrics(const std::string& Filename) {
    std::vector<ML::EmployeeMetrics> Employees;
    std::ifstream File(Filename);
    
    if (!File.IsOpen()) {
        throw std::runtime_error("Failed to open file: " + Filename);
    }
    
    std::string Line;
    size_t LineNum = 0;
    
    // Skip header if needed
    if (HasHeader && std::getline(File, Line)) {
        LineNum++;
        // Optionally validate headers here
    }
    
    // Read data lines
    while (std::getline(File, Line)) {
        LineNum++;
        
        try {
            // Parse CSV line
            std::vector<std::string> Fields = ParseCSVLine(Line);
            
            // Ensure we have enough fields
            if (Fields.size() < 11) {
                std::cerr << "Warning: Line " << LineNum << " has insufficient fields (" 
                          << Fields.size() << "), expected at least 11. Skipping..." << std::endl;
                continue;
            }
            
            // Extract and convert values
            std::string EmployeeId = Fields[0];
            std::string Name = Fields[1];
            
            double CodeCommits = std::stod(Fields[2]);
            double LinesOfCode = std::stod(Fields[3]);
            double CodeReviews = std::stod(Fields[4]);
            double BugsFixed = std::stod(Fields[5]);
            double DocumentationEdits = std::stod(Fields[6]);
            double MeetingAttendance = std::stod(Fields[7]);
            double TeamCollaboration = std::stod(Fields[8]);
            double TechnicalDifficulty = std::stod(Fields[9]);
            double ContributionScore = std::stod(Fields[10]);
            
            // Create employee metrics object
            ML::EmployeeMetrics Metrics(
                EmployeeId, Name, CodeCommits, LinesOfCode, CodeReviews,
                BugsFixed, DocumentationEdits, MeetingAttendance,
                TeamCollaboration, TechnicalDifficulty, ContributionScore
            );
            
            Employees.push_back(Metrics);
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line " << LineNum << ": " << e.What() << std::endl;
            std::cerr << "Line content: " << Line << std::endl;
            // Continue processing other lines
        }
    }
    
    File.close();
    return Employees;
}

// Write employee metrics to CSV file
bool WriteEmployeeMetrics(const std::string& Filename, 
                         const std::vector<ML::EmployeeMetrics>& Employees,
                         bool IncludePredictions = false,
                         const std::vector<double>& Predictions = {}) {
    
    std::ofstream File(Filename);
    if (!File.IsOpen()) {
        std::cerr << "Failed to open file for writing: " << Filename << std::endl;
        return false;
    }
    
    // Write header
    if (HasHeader) {
        File << "EmployeeID" << Delimiter 
             << "Name" << Delimiter 
             << "CodeCommits" << Delimiter 
             << "LinesOfCode" << Delimiter 
             << "CodeReviews" << Delimiter 
             << "BugsFixed" << Delimiter 
             << "DocumentationEdits" << Delimiter 
             << "MeetingAttendance" << Delimiter 
             << "TeamCollaboration" << Delimiter 
             << "TechnicalDifficulty" << Delimiter 
             << "ContributionScore";
        
        if (IncludePredictions) {
            File << Delimiter << "PredictedContribution";
        }
        
        File << std::endl;
    }
    
    // Write data
    for (size_t i = 0; i < Employees.size(); ++i) {
        const auto& Emp = Employees[i];
        
        File << EscapeCSV(Emp.EmployeeId) << Delimiter 
             << EscapeCSV(Emp.Name) << Delimiter 
             << std::fixed << std::setprecision(2) << Emp.CodeCommits << Delimiter 
             << Emp.LinesOfCode << Delimiter 
             << Emp.CodeReviews << Delimiter 
             << Emp.BugsFixed << Delimiter 
             << Emp.DocumentationEdits << Delimiter 
             << Emp.MeetingAttendance << Delimiter 
             << Emp.TeamCollaboration << Delimiter 
             << Emp.TechnicalDifficulty << Delimiter 
             << std::setprecision(4) << Emp.ContributionScore;
        
        if (IncludePredictions && i < Predictions.size()) {
            File << Delimiter << std::setprecision(4) << Predictions[i];
        }
        
        File << std::endl;
    }
    
    File.close();
    return true;
}

// Generate a CSV template file with headers and example data
bool GenerateTemplateFile(const std::string& Filename, int NumExamples = 5) {
    std::ofstream File(Filename);
    if (!File.IsOpen()) {
        std::cerr << "Failed to open file for writing: " << Filename << std::endl;
        return false;
    }
    
    // Write header
    File << "EmployeeID,Name,CodeCommits,LinesOfCode,CodeReviews,BugsFixed,"
         << "DocumentationEdits,MeetingAttendance,TeamCollaboration,"
         << "TechnicalDifficulty,ContributionScore" << std::endl;
    
    // Generate example rows
    std::vector<std::string> Names = {
        "John Smith", "Jane Doe", "Michael Johnson", "Emily Davis", "David Wilson"
    };
    
    for (int i = 0; i < NumExamples && i < static_cast<int>(Names.size()); ++i) {
        File << "EMP" << (i + 1001) << ","
             << Names[i] << ","
             << (10 + i * 5) << "," // Code commits
             << (500 + i * 300) << "," // Lines of code
             << (5 + i * 2) << "," // Code reviews
             << (3 + i * 2) << "," // Bugs fixed
             << (20 + i * 10) << "," // Documentation edits
             << (80 + i * 2) << "," // Meeting attendance
             << (7.0 + i * 0.5) << "," // Team collaboration
             << (6.0 + i * 0.5) << "," // Technical difficulty
             << (0.65 + i * 0.05) << std::endl; // Contribution score
    }
    
    File.close();
    
    std::cout << "Template file generated: " << Filename << std::endl;
    std::cout << "Fill this file with your employee metrics data." << std::endl;
    return true;
}

// Function to append predictions to an existing CSV file
bool AppendPredictions(const std::string& InputFile, const std::string& OutputFile,
                       const std::vector<double>& Predictions) {
    // Read the original file
    std::vector<ML::EmployeeMetrics> Employees = ReadEmployeeMetrics(InputFile);
    
    if (Employees.size() != Predictions.size()) {
        std::cerr << "Error: Number of employees (" << Employees.size() 
                  << ") doesn't match number of predictions (" 
                  << Predictions.size() << ")" << std::endl;
        return false;
    }
    
    // Write with predictions
    return WriteEmployeeMetrics(OutputFile, Employees, true, Predictions);
}

private:
// Parse a single CSV line into fields
std::vector<std::string> ParseCSVLine(const std::string& Line) {
    std::vector<std::string> Fields;
    std::string Field;
    bool InQuotes = false;
    
    for (char c : Line) {
        if (c == '"') {
            InQuotes = !InQuotes;
        } else if (c == Delimiter && !InQuotes) {
            // End of field
            Fields.push_back(Field);
            Field.clear();
        } else {
            Field += c;
        }
    }
    
    // Add the last field
    Fields.push_back(Field);
    
    return Fields;
}

// Escape a string for CSV output
std::string EscapeCSV(const std::string& str) {
    // If the string contains delimiter, quotes, or newlines, it needs to be quoted
    if (str.find(Delimiter) != std::string::npos || 
        str.find('"') != std::string::npos || 
        str.find('\n') != std::string::npos) {
        
        // Double up any quotes in the string
        std::string Escaped = str;
        size_t Pos = 0;
        while ((Pos = Escaped.find('"', Pos)) != std::string::npos) {
            Escaped.insert(Pos, 1, '"');
            Pos += 2;
        }
        
        // Wrap in quotes
        return "\"" + Escaped + "\"";
    }
    
    return str;
}
};

/**
* @brief Batch Processor for processing multiple employees in batches
*/
class BatchProcessor {
private:
ML::EmployeeContributionPredictor& Predictor;

public:
// Constructor
BatchProcessor(ML::EmployeeContributionPredictor& Pred) : Predictor(Pred) {}

// Process a batch of employees from input CSV to output CSV
bool ProcessBatch(const std::string& InputFile, const std::string& OutputFile) {
    try {
        // Read employee data
        CSVHandler CsvHandler;
        std::vector<ML::EmployeeMetrics> Employees = CsvHandler.ReadEmployeeMetrics(InputFile);
        
        if (Employees.empty()) {
            std::cerr << "No employees loaded from file: " << InputFile << std::endl;
            return false;
        }
        
        std::cout << "Processing " << Employees.size() << " employees..." << std::endl;
        
        // Generate predictions
        std::vector<double> Predictions = Predictor.PredictContributions(Employees);
        
        // Write predictions to output file
        return CsvHandler.WriteEmployeeMetrics(OutputFile, Employees, true, Predictions);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during batch processing: " << e.What() << std::endl;
        return false;
    }
}

// Process a directory of CSV files
bool ProcessDirectory(const std::string& InputDir, const std::string& OutputDir) {
    try {
        // Create output directory if it doesn't exist
        if (!std::filesystem::exists(OutputDir)) {
            std::filesystem::create_directories(OutputDir);
        }
        
        size_t FilesProcessed = 0;
        
        // Process each CSV file in the directory
        for (const auto& Entry : std::filesystem::directory_iterator(InputDir)) {
            if (Entry.Path().Extension() == ".csv") {
                std::string InputFile = Entry.Path().string();
                std::string Filename = Entry.Path().Filename().string();
                std::string OutputFile = OutputDir + "/" + "processed_" + Filename;
                
                std::cout << "Processing file: " << Filename << std::endl;
                
                if (ProcessBatch(InputFile, OutputFile)) {
                    FilesProcessed++;
                }
            }
        }
        
        std::cout << "Successfully processed " << FilesProcessed << " files." << std::endl;
        return FilesProcessed > 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during directory processing: " << e.What() << std::endl;
        return false;
    }
}

// Analyze directory of employee data and output statistical report
bool AnalyzeDirectory(const std::string& InputDir, const std::string& ReportFile) {
    try {
        std::vector<ML::EmployeeMetrics> AllEmployees;
        
        // Load all employees from all CSV files
        for (const auto& Entry : std::filesystem::directory_iterator(InputDir)) {
            if (Entry.Path().Extension() == ".csv") {
                CSVHandler CsvHandler;
                auto Employees = CsvHandler.ReadEmployeeMetrics(Entry.Path().string());
                AllEmployees.insert(AllEmployees.end(), Employees.begin(), Employees.end());
            }
        }
        
        if (AllEmployees.empty()) {
            std::cerr << "No employee data found in directory: " << InputDir << std::endl;
            return false;
        }
        
        // Generate predictions for all employees
        std::vector<double> Predictions = Predictor.PredictContributions(AllEmployees);
        
        // Calculate statistics
        struct TeamStats {
            double AvgActualContribution = 0.0;
            double AvgPredictedContribution = 0.0;
            double MinContribution = 1.0;
            double MaxContribution = 0.0;
            size_t count = 0;
            
            void AddEmployee(double Actual, double Predicted) {
                AvgActualContribution = (AvgActualContribution * count + Actual) / (count + 1);
                AvgPredictedContribution = (AvgPredictedContribution * count + Predicted) / (count + 1);
                MinContribution = std::min(MinContribution, Actual);
                MaxContribution = std::max(MaxContribution, Actual);
                count++;
            }
        };
        
        std::unordered_map<std::string, TeamStats> TeamStats;
        
        // Extract team from employee ID (assuming format TEAM-XXX)
        for (size_t i = 0; i < AllEmployees.size(); ++i) {
            std::string Team = "Unknown";
            size_t DashPos = AllEmployees[i].EmployeeId.find('-');
            if (DashPos != std::string::npos) {
                Team = AllEmployees[i].EmployeeId.substr(0, DashPos);
            }
            
            TeamStats[Team].AddEmployee(AllEmployees[i].ContributionScore, Predictions[i]);
        }
        
        // Generate report
        std::ofstream Report(ReportFile);
        if (!Report.IsOpen()) {
            std::cerr << "Failed to open report file for writing: " << ReportFile << std::endl;
            return false;
        }
        
        Report << "Employee Contribution Analysis Report" << std::endl;
        Report << "=====================================" << std::endl << std::endl;
        
        Report << "Overall Statistics:" << std::endl;
        Report << "Total Employees: " << AllEmployees.size() << std::endl;
        
        // Calculate overall averages
        double TotalActual = 0.0, TotalPredicted = 0.0;
        double MinActual = 1.0, MaxActual = 0.0;
        
        for (size_t i = 0; i < AllEmployees.size(); ++i) {
            TotalActual += AllEmployees[i].ContributionScore;
            TotalPredicted += Predictions[i];
            MinActual = std::min(MinActual, AllEmployees[i].ContributionScore);
            MaxActual = std::max(MaxActual, AllEmployees[i].ContributionScore);
        }
        
        double AvgActual = TotalActual / AllEmployees.size();
        double AvgPredicted = TotalPredicted / AllEmployees.size();
        
        Report << "Average Actual Contribution: " << std::fixed << std::setprecision(2)
               << (AvgActual * 100.0) << "%" << std::endl;
        Report << "Average Predicted Contribution: " << std::fixed << std::setprecision(2)
               << (AvgPredicted * 100.0) << "%" << std::endl;
        Report << "Minimum Contribution: " << std::fixed << std::setprecision(2)
               << (MinActual * 100.0) << "%" << std::endl;
        Report << "Maximum Contribution: " << std::fixed << std::setprecision(2)
               << (MaxActual * 100.0) << "%" << std::endl << std::endl;
        
        Report << "Team Statistics:" << std::endl;
        Report << "---------------" << std::endl;
        
        for (const auto& [Team, Stats] : TeamStats) {
            Report << "Team: " << Team << std::endl;
            Report << "  Number of Employees: " << Stats.count << std::endl;
            Report << "  Average Actual Contribution: " << std::fixed << std::setprecision(2)
                   << (Stats.AvgActualContribution * 100.0) << "%" << std::endl;
            Report << "  Average Predicted Contribution: " << std::fixed << std::setprecision(2)
                   << (Stats.AvgPredictedContribution * 100.0) << "%" << std::endl;
            Report << "  Range: " << std::fixed << std::setprecision(2)
                   << (Stats.MinContribution * 100.0) << "% - "
                   << (Stats.MaxContribution * 100.0) << "%" << std::endl << std::endl;
        }
        
        Report.close();
        
        std::cout << "Analysis report generated: " << ReportFile << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during analysis: " << e.What() << std::endl;
        return false;
    }
}
};

/**
* @brief Main function to demonstrate the neural network for employee contribution prediction
*/
int main(int argc, char* argv[]) {
try {
    std::cout << "Employee Software Contribution Neural Network Predictor" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    // Command line argument parsing for different modes
    std::string Mode = "train"; // Default mode
    std::string InputFile = "";
    std::string OutputFile = "";
    
    if (argc > 1) {
        Mode = argv[1];
    }
    
    if (argc > 2) {
        InputFile = argv[2];
    }
    
    if (argc > 3) {
        OutputFile = argv[3];
    }
    
    // Create CSV handler
    CSVHandler CsvHandler;
    
    // If no input file is specified, generate a template
    if (Mode != "template" && InputFile.empty()) {
        std::cout << "No input file specified. Generating template CSV..." << std::endl;
        CsvHandler.GenerateTemplateFile("employee_template.csv");
        std::cout << "Please fill in the template and rerun with: " << std::endl;
        std::cout << "./employee_nn train employee_template.csv" << std::endl;
        return 0;
    }
    
    // Handle different operation modes
    if (Mode == "template") {
        // Generate template file
        std::string TemplateFile = InputFile.empty() ? "employee_template.csv" : InputFile;
        CsvHandler.GenerateTemplateFile(TemplateFile);
        return 0;
    }
    else if (Mode == "train" || Mode == "train_and_predict") {
        // Load data from CSV
        std::vector<ML::EmployeeMetrics> data;
        
        if (std::filesystem::exists(InputFile)) {
            std::cout << "Loading data from: " << InputFile << std::endl;
            data = CsvHandler.ReadEmployeeMetrics(InputFile);
            std::cout << "Loaded " << data.size() << " employee records." << std::endl;
        } else {
            // If input file doesn't exist, generate synthetic data
            std::cout << "Input file not found, generating synthetic data..." << std::endl;
            
            // Generate synthetic data
            std::random_device Rd;
            std::mt19937 Gen(Rd());
            std::uniform_int_distribution<> IdDist(10000, 99999);
            std::uniform_int_distribution<> CommitsDist(0, 100);
            std::uniform_int_distribution<> LocDist(0, 5000);
            std::uniform_int_distribution<> ReviewsDist(0, 50);
            std::uniform_int_distribution<> BugsDist(0, 30);
            std::uniform_int_distribution<> DocsDist(0, 200);
            std::uniform_real_distribution<> MeetingsDist(50.0, 100.0);
            std::uniform_real_distribution<> CollabDist(1.0, 10.0);
            std::uniform_real_distribution<> DifficultyDist(1.0, 10.0);
            
            // Generate 500 synthetic employees
            for (int i = 0; i < 500; ++i) {
                std::string Id = "EMP" + std::to_string(IdDist(Gen));
                std::string Name = "Employee" + std::to_string(i);
                
                double Commits = CommitsDist(Gen);
                double Loc = LocDist(Gen);
                double Reviews = ReviewsDist(Gen);
                double Bugs = BugsDist(Gen);
                double Docs = DocsDist(Gen);
                double Meetings = MeetingsDist(Gen);
                double Collab = CollabDist(Gen);
                double Difficulty = DifficultyDist(Gen);
                
                // Calculate a contribution score based on metrics (simulated formula)
                double ContributionScore = 0.0;
                ContributionScore += 0.15 * (Commits / 100.0);
                ContributionScore += 0.10 * (Loc / 5000.0);
                ContributionScore += 0.15 * (Reviews / 50.0);
                ContributionScore += 0.20 * (Bugs / 30.0);
                ContributionScore += 0.05 * (Docs / 200.0);
                ContributionScore += 0.10 * (Meetings / 100.0);
                ContributionScore += 0.15 * (Collab / 10.0);
                ContributionScore += 0.10 * (Difficulty / 10.0);
                
                // Add some randomness (noise)
                std::normal_distribution<> Noise(0.0, 0.05);
                ContributionScore = std::min(1.0, std::max(0.0, ContributionScore + Noise(Gen)));
                
                ML::EmployeeMetrics Metrics(
                    Id, Name, Commits, Loc, Reviews, Bugs, Docs, 
                    Meetings, Collab, Difficulty, ContributionScore
                );
                
                data.push_back(Metrics);
            }
            
            // Save synthetic data to CSV
            std::string SyntheticFile = "synthetic_employees.csv";
            CsvHandler.WriteEmployeeMetrics(SyntheticFile, data);
            std::cout << "Generated " << data.size() << " synthetic employee records and saved to "
                      << SyntheticFile << std::endl;
        }
        
        if (data.empty()) {
            std::cerr << "No data to train on. Exiting." << std::endl;
            return 1;
        }
        
        // Split data for training and testing
        std::vector<ML::EmployeeMetrics> TrainingData, TestData;
        size_t SplitIndex = static_cast<size_t>(data.size() * 0.8); // 80% training, 20% testing
        TrainingData.insert(TrainingData.end(), data.begin(), data.begin() + SplitIndex);
        TestData.insert(TestData.end(), data.begin() + SplitIndex, data.end());
        
        std::cout << "Training set size: " << TrainingData.size() << std::endl;
        std::cout << "Test set size: " << TestData.size() << std::endl;
        
        // Configure neural network
        ML::NeuralNetConfig Config;
        Config.LayerSizes = {8, 16, 8, 1};  // 8 input features, 2 hidden layers, 1 output
        Config.Activations = {
            std::make_shared<ML::ReLUActivation>(),      // First hidden layer
            std::make_shared<ML::ReLUActivation>(),      // Second hidden layer
            std::make_shared<ML::SigmoidActivation>()    // Output layer (sigmoid for 0-1 output)
        };
        Config.LearningRate = 0.01;
        Config.MaxEpochs = 2000;
        Config.ErrorThreshold = 0.0001;
        Config.Verbose = true;
        
        // Create and train predictor
        std::cout << "Creating neural network model..." << std::endl;
        ML::EmployeeContributionPredictor Predictor;
        Predictor.ConfigureNetwork(Config);
        
        std::cout << "Training neural network..." << std::endl;
        auto StartTime = std::chrono::high_resolution_clock::now();
        double ValidationError = Predictor.Train(TrainingData);
        auto EndTime = std::chrono::high_resolution_clock::now();
        auto TrainingTime = std::chrono::duration_cast<std::chrono::seconds>(EndTime - StartTime).count();
        
        std::cout << "Training completed in " << TrainingTime << " seconds." << std::endl;
        std::cout << "Validation error: " << ValidationError << std::endl;
        
        // Save the model
        std::cout << "Saving model..." << std::endl;
        if (Predictor.SaveModel("employee_model.bin", "employee_norm.csv")) {
            std::cout << "Model saved successfully." << std::endl;
        } else {
            std::cout << "Failed to save model." << std::endl;
        }
        
        // Evaluate model performance
        std::cout << "Evaluating model on test data..." << std::endl;
        
        std::vector<double> Predictions = Predictor.PredictContributions(TestData);
        
        // Calculate error metrics
        double Mse = 0.0, Mae = 0.0;
        for (size_t i = 0; i < TestData.size(); ++i) {
            double Error = Predictions[i] - TestData[i].ContributionScore;
            Mse += Error * Error;
            Mae += std::abs(Error);
        }
        
        Mse /= TestData.size();
        Mae /= TestData.size();
        double Rmse = std::sqrt(Mse);
        
        std::cout << "Test set metrics:" << std::endl;
        std::cout << "Mean Squared Error (MSE): " << Mse << std::endl;
        std::cout << "Root Mean Squared Error (RMSE): " << Rmse << std::endl;
        std::cout << "Mean Absolute Error (MAE): " << Mae << std::endl;
        
        // If train_and_predict mode, also make predictions on a new file
        if (Mode == "train_and_predict" && !OutputFile.empty()) {
            std::cout << "Making predictions on: " << OutputFile << std::endl;
            
            // Create batch processor
            BatchProcessor BatchProcessor(Predictor);
            BatchProcessor.ProcessBatch(OutputFile, "predictions_" + OutputFile);
        }
    }
    else if (Mode == "predict") {
        // Load the model
        ML::EmployeeContributionPredictor Predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!Predictor.LoadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Make predictions
        if (std::filesystem::is_directory(InputFile)) {
            // Process an entire directory
            std::cout << "Processing directory: " << InputFile << std::endl;
            
            std::string OutputDir = OutputFile.empty() ? "predictions" : OutputFile;
            
            BatchProcessor BatchProcessor(Predictor);
            BatchProcessor.ProcessDirectory(InputFile, OutputDir);
        } else {
            // Process a single file
            std::cout << "Processing file: " << InputFile << std::endl;
            
            std::string OutputFilename = OutputFile.empty() ? 
                "predictions_" + InputFile : OutputFile;
            
            BatchProcessor BatchProcessor(Predictor);
            BatchProcessor.ProcessBatch(InputFile, OutputFilename);
        }
    }
    else if (Mode == "analyze") {
        // Load the model
        ML::EmployeeContributionPredictor Predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!Predictor.LoadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Analyze data
        std::string ReportFile = OutputFile.empty() ? "contribution_analysis.txt" : OutputFile;
        
        if (std::filesystem::is_directory(InputFile)) {
            // Analyze an entire directory
            std::cout << "Analyzing directory: " << InputFile << std::endl;
            
            BatchProcessor BatchProcessor(Predictor);
            BatchProcessor.AnalyzeDirectory(InputFile, ReportFile);
        } else {
            // Analyze a single file
            std::cout << "Analyzing file: " << InputFile << std::endl;
            
            // Load employee data
            std::vector<ML::EmployeeMetrics> Employees = CsvHandler.ReadEmployeeMetrics(InputFile);
            
            if (Employees.empty()) {
                std::cerr << "No employees loaded from file: " << InputFile << std::endl;
                return 1;
            }
            
            // Make predictions
            std::vector<double> Predictions = Predictor.PredictContributions(Employees);
            
            // Print detailed analysis of each employee
            for (size_t i = 0; i < Employees.size(); ++i) {
                std::cout << "Employee: " << Employees[i].Name << " (ID: " << Employees[i].EmployeeId << ")" << std::endl;
                std::cout << "  Metrics:" << std::endl;
                std::cout << "    Code Commits: " << Employees[i].CodeCommits << std::endl;
                std::cout << "    Lines of Code: " << Employees[i].LinesOfCode << std::endl;
                std::cout << "    Code Reviews: " << Employees[i].CodeReviews << std::endl;
                std::cout << "    Bugs Fixed: " << Employees[i].BugsFixed << std::endl;
                std::cout << "    Documentation Edits: " << Employees[i].DocumentationEdits << std::endl;
                std::cout << "    Meeting Attendance: " << Employees[i].MeetingAttendance << "%" << std::endl;
                std::cout << "    Team Collaboration: " << Employees[i].TeamCollaboration << "/10" << std::endl;
                std::cout << "    Technical Difficulty: " << Employees[i].TechnicalDifficulty << "/10" << std::endl;
                std::cout << "  Actual Contribution: " << std::fixed << std::setprecision(2) 
                          << (Employees[i].ContributionScore * 100.0) << "%" << std::endl;
                std::cout << "  Predicted Contribution: " << std::fixed << std::setprecision(2) 
                          << (Predictions[i] * 100.0) << "%" << std::endl;
                std::cout << "  Difference: " << std::fixed << std::setprecision(2) 
                          << ((Predictions[i] - Employees[i].ContributionScore) * 100.0) << "%" << std::endl;
                std::cout << "------------------------------------" << std::endl;
            }
            
            // Save results to a report file
            std::ofstream Report(ReportFile);
            Report << "Employee Contribution Analysis" << std::endl;
            Report << "=============================" << std::endl << std::endl;
            
            for (size_t i = 0; i < Employees.size(); ++i) {
                Report << "Employee: " << Employees[i].Name << " (ID: " << Employees[i].EmployeeId << ")" << std::endl;
                Report << "  Actual Contribution: " << std::fixed << std::setprecision(2) 
                       << (Employees[i].ContributionScore * 100.0) << "%" << std::endl;
                Report << "  Predicted Contribution: " << std::fixed << std::setprecision(2) 
                       << (Predictions[i] * 100.0) << "%" << std::endl;
                Report << "  Difference: " << std::fixed << std::setprecision(2) 
                       << ((Predictions[i] - Employees[i].ContributionScore) * 100.0) << "%" << std::endl;
                Report << "------------------------------------" << std::endl;
            }
            
            Report.close();
            std::cout << "Analysis saved to: " << ReportFile << std::endl;
        }
    }
    else if (Mode == "interactive") {
        // Load the model
        ML::EmployeeContributionPredictor Predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!Predictor.LoadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Interactive mode for entering employee metrics
        std::cout << "Interactive Mode - Enter employee metrics:" << std::endl;
        
        std::string EmployeeId, Name;
        double CodeCommits, LinesOfCode, CodeReviews, BugsFixed;
        double DocumentationEdits, MeetingAttendance, TeamCollaboration, TechnicalDifficulty;
        
        std::cout << "Employee ID: ";
        std::cin >> EmployeeId;
        std::cin.ignore(); // Clear newline
        
        std::cout << "Name: ";
        std::getline(std::cin, Name);
        
        std::cout << "Code Commits: ";
        std::cin >> CodeCommits;
        
        std::cout << "Lines of Code: ";
        std::cin >> LinesOfCode;
        
        std::cout << "Code Reviews: ";
        std::cin >> CodeReviews;
        
        std::cout << "Bugs Fixed: ";
        std::cin >> BugsFixed;
        
        std::cout << "Documentation Edits: ";
        std::cin >> DocumentationEdits;
        
        std::cout << "Meeting Attendance (%): ";
        std::cin >> MeetingAttendance;
        
        std::cout << "Team Collaboration (0-10): ";
        std::cin >> TeamCollaboration;
        
        std::cout << "Technical Difficulty (0-10): ";
        std::cin >> TechnicalDifficulty;
        
        // Create employee object
        ML::EmployeeMetrics Employee(
            EmployeeId, Name, CodeCommits, LinesOfCode, CodeReviews,
            BugsFixed, DocumentationEdits, MeetingAttendance,
            TeamCollaboration, TechnicalDifficulty
        );
        
        // Make prediction
        double Contribution = Predictor.PredictContribution(Employee);
        
        std::cout << "\nPrediction Results:" << std::endl;
        std::cout << "Employee: " << Name << " (ID: " << EmployeeId << ")" << std::endl;
        std::cout << "Predicted Contribution: " << std::fixed << std::setprecision(2) 
                 << (Contribution * 100.0) << "%" << std::endl;
        
        // Provide some analysis
        std::cout << "\nContribution Analysis:" << std::endl;
        
        if (Contribution >= 0.9) {
            std::cout << "Exceptional performer - consider for leadership roles or mentoring opportunities." << std::endl;
        } else if (Contribution >= 0.75) {
            std::cout << "Strong performer - valuable team member with consistent high output." << std::endl;
        } else if (Contribution >= 0.6) {
            std::cout << "Good performer - reliable contributor with room for growth." << std::endl;
        } else if (Contribution >= 0.4) {
            std::cout << "Average performer - may benefit from targeted coaching in specific areas." << std::endl;
        } else if (Contribution >= 0.25) {
            std::cout << "Below average performer - consider performance improvement plan." << std::endl;
        } else {
            std::cout << "Struggling performer - needs immediate attention and support." << std::endl;
        }
    }
    else {
        std::cerr << "Unknown mode: " << Mode << std::endl;
        std::cout << "Available modes:" << std::endl;
        std::cout << "  template [filename]                   - Generate a template CSV file" << std::endl;
        std::cout << "  train [inputfile]                     - Train the model using data from CSV" << std::endl;
        std::cout << "  predict [inputfile] [outputfile]      - Make predictions using a trained model" << std::endl;
        std::cout << "  train_and_predict [train] [predict]   - Train and then predict on different data" << std::endl;
        std::cout << "  analyze [inputfile] [reportfile]      - Analyze employee data and generate a report" << std::endl;
        std::cout << "  interactive                          - Enter employee metrics interactively" << std::endl;
        return 1;
    }
    
    std::cout << "Done." << std::endl;
    
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.What() << std::endl;
    return 1;
}

return 0;
