// Compute statistics from a dataset
void computeStatistics(const std::vector<EmployeeMetrics>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute statistics from empty dataset");
    }
    
    // Reset statistics
    for (const auto& feature : featureNames) {
        minValues[feature] = std::numeric_limits<double>::max();
        maxValues[feature] = std::numeric_limits<double>::lowest();
        meanValues[feature] = 0.0;
        stdDevValues[feature] = 0.0;
    }
    
    // Compute min, max, and mean
    for (const auto& employee : data) {
        updateStatisticsFromEmployee(employee, true);
    }
    
    // Compute standard deviation
    if (normMethod == NormMethod::ZScore) {
        // Divide sums by count to get means
        for (const auto& feature : featureNames) {
            meanValues[feature] /= data.size();
        }
        
        // Compute squared differences
        for (const auto& employee : data) {
            std::vector<double> values = employeeToVector(employee);
            for (size_t i = 0; i < featureNames.size() && i < values.size(); ++i) {
                double diff = values[i] - meanValues[featureNames[i]];
                stdDevValues[featureNames[i]] += diff * diff;
            }
        }
        
        // Finalize standard deviations
        for (const auto& feature : featureNames) {
            stdDevValues[feature] = std::sqrt(stdDevValues[feature] / data.size());
            // Avoid division by zero
            if (stdDevValues[feature] < 1e-10) {
                stdDevValues[feature] = 1.0;
            }
        }
    }
}

// Update statistics from a single employee record
void updateStatisticsFromEmployee(const EmployeeMetrics& employee, bool computingMean = false) {
    std::vector<double> values = employeeToVector(employee);
    
    for (size_t i = 0; i < featureNames.size() && i < values.size(); ++i) {
        const std::string& feature = featureNames[i];
        double value = values[i];
        
        // Update min and max
        minValues[feature] = std::min(minValues[feature], value);
        maxValues[feature] = std::max(maxValues[feature], value);
        
        // Update sum for mean calculation
        if (computingMean) {
            meanValues[feature] += value;
        }
    }
}

// Convert employee metrics to vector
std::vector<double> employeeToVector(const EmployeeMetrics& employee) const {
    return {
        employee.codeCommits,
        employee.linesOfCode,
        employee.codeReviews,
        employee.bugsFixed,
        employee.documentationEdits,
        employee.meetingAttendance,
        employee.teamCollaboration,
        employee.technicalDifficulty,
        employee.contributionScore
    };
}

// Normalize a single value
double normalizeValue(double value, const std::string& feature) const {
    if (normMethod == NormMethod::MinMax) {
        // Min-Max normalization: (value - min) / (max - min)
        double min = minValues.at(feature);
        double max = maxValues.at(feature);
        if (std::abs(max - min) < 1e-10) {
            return 0.5; // Default to mid-range if no variation
        }
        return (value - min) / (max - min);
    } else {
        // Z-score normalization: (value - mean) / stdDev
        double mean = meanValues.at(feature);
        double stdDev = stdDevValues.at(feature);
        return (value - mean) / stdDev;
    }
}

// Denormalize a value
double denormalizeValue(double normalizedValue, const std::string& feature) const {
    if (normMethod == NormMethod::MinMax) {
        // Min-Max denormalization: normalizedValue * (max - min) + min
        double min = minValues.at(feature);
        double max = maxValues.at(feature);
        return normalizedValue * (max - min) + min;
    } else {
        // Z-score denormalization: normalizedValue * stdDev + mean
        double mean = meanValues.at(feature);
        double stdDev = stdDevValues.at(feature);
        return normalizedValue * stdDev + mean;
    }
}

// Normalize an employee metrics object
EmployeeMetrics normalizeMetrics(const EmployeeMetrics& employee) const {
    EmployeeMetrics normalized = employee;
    normalized.codeCommits = normalizeValue(employee.codeCommits, "codeCommits");
    normalized.linesOfCode = normalizeValue(employee.linesOfCode, "linesOfCode");
    normalized.codeReviews = normalizeValue(employee.codeReviews, "codeReviews");
    normalized.bugsFixed = normalizeValue(employee.bugsFixed, "bugsFixed");
    normalized.documentationEdits = normalizeValue(employee.documentationEdits, "documentationEdits");
    normalized.meetingAttendance = normalizeValue(employee.meetingAttendance, "meetingAttendance");
    normalized.teamCollaboration = normalizeValue(employee.teamCollaboration, "teamCollaboration");
    normalized.technicalDifficulty = normalizeValue(employee.technicalDifficulty, "technicalDifficulty");
    normalized.contributionScore = normalizeValue(employee.contributionScore, "contributionScore");
    return normalized;
}

// Denormalize an employee metrics object
EmployeeMetrics denormalizeMetrics(const EmployeeMetrics& normalizedEmployee) const {
    EmployeeMetrics denormalized = normalizedEmployee;
    denormalized.codeCommits = denormalizeValue(normalizedEmployee.codeCommits, "codeCommits");
    denormalized.linesOfCode = denormalizeValue(normalizedEmployee.linesOfCode, "linesOfCode");
    denormalized.codeReviews = denormalizeValue(normalizedEmployee.codeReviews, "codeReviews");
    denormalized.bugsFixed = denormalizeValue(normalizedEmployee.bugsFixed, "bugsFixed");
    denormalized.documentationEdits = denormalizeValue(normalizedEmployee.documentationEdits, "documentationEdits");
    denormalized.meetingAttendance = denormalizeValue(normalizedEmployee.meetingAttendance, "meetingAttendance");
    denormalized.teamCollaboration = denormalizeValue(normalizedEmployee.teamCollaboration, "teamCollaboration");
    denormalized.technicalDifficulty = denormalizeValue(normalizedEmployee.technicalDifficulty, "technicalDifficulty");
    denormalized.contributionScore = denormalizeValue(normalizedEmployee.contributionScore, "contributionScore");
    return denormalized;
}

// Prepare data for neural network training
void prepareTrainingData(const std::vector<EmployeeMetrics>& data,
                        std::vector<std::vector<double>>& inputs,
                        std::vector<std::vector<double>>& targets) {
    // Compute statistics first
    computeStatistics(data);
    
    // Clear output vectors
    inputs.clear();
    targets.clear();
    
    // Normalize and separate inputs and targets
    for (const auto& employee : data) {
        EmployeeMetrics normalized = normalizeMetrics(employee);
        
        // Extract inputs
        inputs.push_back(normalized.toInputVector());
        
        // Extract target (contribution score)
        targets.push_back({normalized.contributionScore});
    }
}

// Split data into training and validation sets
void splitData(const std::vector<EmployeeMetrics>& data, 
               std::vector<EmployeeMetrics>& trainingData,
               std::vector<EmployeeMetrics>& validationData,
               double validationRatio = 0.2) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot split empty dataset");
    }
    
    // Clear output vectors
    trainingData.clear();
    validationData.clear();
    
    // Create a copy and shuffle
    std::vector<EmployeeMetrics> shuffledData = data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffledData.begin(), shuffledData.end(), g);
    
    // Calculate split index
    size_t validationSize = static_cast<size_t>(std::round(data.size() * validationRatio));
    size_t trainingSize = data.size() - validationSize;
    
    // Split data
    trainingData.insert(trainingData.end(), shuffledData.begin(), shuffledData.begin() + trainingSize);
    validationData.insert(validationData.end(), shuffledData.begin() + trainingSize, shuffledData.end());
}

// Save normalization parameters to a file
bool saveParameters(const std::string& filename) const {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        // Write normalization method
        file << (normMethod == NormMethod::MinMax ? "MinMax" : "ZScore") << "\n";
        
        // Write feature names
        for (const auto& feature : featureNames) {
            file << feature << ",";
        }
        file << "\n";
        
        // Write min values
        for (const auto& feature : featureNames) {
            file << minValues.at(feature) << ",";
        }
        file << "\n";
        
        // Write max values
        for (const auto& feature : featureNames) {
            file << maxValues.at(feature) << ",";
        }
        file << "\n";
        
        // Write mean values
        for (const auto& feature : featureNames) {
            file << meanValues.at(feature) << ",";
        }
        file << "\n";
        
        // Write stdDev values
        for (const auto& feature : featureNames) {
            file << stdDevValues.at(feature) << ",";
        }
        file << "\n";
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving normalization parameters: " << e.what() << std::endl;
        return false;
    }
}

// Load normalization parameters from a file
bool loadParameters(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        
        // Read normalization method
        std::getline(file, line);
        normMethod = (line == "MinMax") ? NormMethod::MinMax : NormMethod::ZScore;
        
        // Read feature names
        std::getline(file, line);
        std::istringstream featureStream(line);
        std::string feature;
        featureNames.clear();
        while (std::getline(featureStream, feature, ',')) {
            if (!feature.empty()) {
                featureNames.push_back(feature);
            }
        }
        
        // Read min values
        std::getline(file, line);
        std::istringstream minStream(line);
        std::string valueStr;
        size_t featureIndex = 0;
        while (std::getline(minStream, valueStr, ',') && featureIndex < featureNames.size()) {
            if (!valueStr.empty()) {
                minValues[featureNames[featureIndex++]] = std::stod(valueStr);
            }
        }
        
        // Read max values
        std::getline(file, line);
        std::istringstream maxStream(line);
        featureIndex = 0;
        while (std::getline(maxStream, valueStr, ',') && featureIndex < featureNames.size()) {
            if (!valueStr.empty()) {
                maxValues[featureNames[featureIndex++]] = std::stod(valueStr);
            }
        }
        
        // Read mean values
        std::getline(file, line);
        std::istringstream meanStream(line);
        featureIndex = 0;
        while (std::getline(meanStream, valueStr, ',') && featureIndex < featureNames.size()) {
            if (!valueStr.empty()) {
                meanValues[featureNames[featureIndex++]] = std::stod(valueStr);
            }
        }
        
        // Read stdDev values
        std::getline(file, line);
        std::istringstream stdDevStream(line);
        featureIndex = 0;
        while (std::getline(stdDevStream, valueStr, ',') && featureIndex < featureNames.size()) {
            if (!valueStr.empty()) {
                stdDevValues[featureNames[featureIndex++]] = std::stod(valueStr);
            }
        }
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading normalization parameters: " << e.what() << std::endl;
        return false;
    }
}
};

/**
* @brief Employee Contribution Predictor class that combines data processing and neural network
*/
class EmployeeContributionPredictor {
private:
std::unique_ptr<NeuralNetwork> neuralNetwork;
std::unique_ptr<DataProcessor> dataProcessor;
bool isModelTrained;

public:
// Constructor
EmployeeContributionPredictor() 
    : isModelTrained(false) {
    // Create data processor with default settings
    dataProcessor = std::make_unique<DataProcessor>();
    
    // Create neural network with default architecture
    NeuralNetConfig config;
    neuralNetwork = std::make_unique<NeuralNetwork>(config);
}

// Configure the neural network
void configureNetwork(const NeuralNetConfig& config) {
    neuralNetwork = std::make_unique<NeuralNetwork>(config);
    isModelTrained = false;
}

// Train the predictor
double train(const std::vector<EmployeeMetrics>& data, double validationRatio = 0.2) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot train with empty dataset");
    }
    
    // Split data into training and validation sets
    std::vector<EmployeeMetrics> trainingData, validationData;
    dataProcessor->splitData(data, trainingData, validationData, validationRatio);
    
    // Prepare training data
    std::vector<std::vector<double>> trainingInputs, trainingTargets;
    dataProcessor->prepareTrainingData(trainingData, trainingInputs, trainingTargets);
    
    // Train the neural network
    neuralNetwork->train(trainingInputs, trainingTargets);
    
    // Prepare validation data
    std::vector<std::vector<double>> validationInputs, validationTargets;
    dataProcessor->prepareTrainingData(validationData, validationInputs, validationTargets);
    
    // Validate the neural network
    double validationError = neuralNetwork->validate(validationInputs, validationTargets);
    
    isModelTrained = true;
    return validationError;
}

// Predict contribution for a single employee
double predictContribution(const EmployeeMetrics& employee) const {
    if (!isModelTrained) {
        throw std::runtime_error("Model not trained yet");
    }
    
    // Normalize metrics
    EmployeeMetrics normalizedEmployee = dataProcessor->normalizeMetrics(employee);
    
    // Predict using normalized input vector
    std::vector<double> inputs = normalizedEmployee.toInputVector();
    std::vector<double> outputs = neuralNetwork->feedForward(inputs);
    
    if (outputs.empty()) {
        throw std::runtime_error("Neural network produced no output");
    }
    
    // Denormalize the prediction
    return dataProcessor->denormalizeValue(outputs[0], "contributionScore");
}

// Predict contributions for multiple employees
std::vector<double> predictContributions(const std::vector<EmployeeMetrics>& employees) const {
    std::vector<double> predictions;
    predictions.reserve(employees.size());
    
    for (const auto& employee : employees) {
        predictions.push_back(predictContribution(employee));
    }
    
    return predictions;
}

// Save the model and normalization parameters
bool saveModel(const std::string& modelFile, const std::string& normFile) const {
    return neuralNetwork->saveModel(modelFile) && 
           dataProcessor->saveParameters(normFile);
}

// Load the model and normalization parameters
bool loadModel(const std::string& modelFile, const std::string& normFile) {
    bool success = neuralNetwork->loadModel(modelFile) && 
                   dataProcessor->loadParameters(normFile);
    isModelTrained = success;
    return success;
}

// Get training statistics
double getTrainingError() const { return neuralNetwork->getTotalError(); }
size_t getTrainingEpochs() const { return neuralNetwork->getTrainingEpochs(); }
double getValidationAccuracy() const { return neuralNetwork->getValidationAccuracy(); }
};

} // namespace ML

/**
* @brief Enhanced CSV handling class for employee data
* This class specializes in reading and processing CSV files containing employee metrics
*/
class CSVHandler {
private:
bool hasHeader;
char delimiter;
std::string dateFormat;

public:
// Constructor with options
CSVHandler(bool header = true, char delim = ',', std::string format = "%Y-%m-%d")
    : hasHeader(header), delimiter(delim), dateFormat(std::move(format)) {}

// Read employee metrics from CSV file
std::vector<ML::EmployeeMetrics> readEmployeeMetrics(const std::string& filename) {
    std::vector<ML::EmployeeMetrics> employees;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::string line;
    size_t lineNum = 0;
    
    // Skip header if needed
    if (hasHeader && std::getline(file, line)) {
        lineNum++;
        // Optionally validate headers here
    }
    
    // Read data lines
    while (std::getline(file, line)) {
        lineNum++;
        
        try {
            // Parse CSV line
            std::vector<std::string> fields = parseCSVLine(line);
            
            // Ensure we have enough fields
            if (fields.size() < 11) {
                std::cerr << "Warning: Line " << lineNum << " has insufficient fields (" 
                          << fields.size() << "), expected at least 11. Skipping..." << std::endl;
                continue;
            }
            
            // Extract and convert values
            std::string employeeId = fields[0];
            std::string name = fields[1];
            
            double codeCommits = std::stod(fields[2]);
            double linesOfCode = std::stod(fields[3]);
            double codeReviews = std::stod(fields[4]);
            double bugsFixed = std::stod(fields[5]);
            double documentationEdits = std::stod(fields[6]);
            double meetingAttendance = std::stod(fields[7]);
            double teamCollaboration = std::stod(fields[8]);
            double technicalDifficulty = std::stod(fields[9]);
            double contributionScore = std::stod(fields[10]);
            
            // Create employee metrics object
            ML::EmployeeMetrics metrics(
                employeeId, name, codeCommits, linesOfCode, codeReviews,
                bugsFixed, documentationEdits, meetingAttendance,
                teamCollaboration, technicalDifficulty, contributionScore
            );
            
            employees.push_back(metrics);
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line " << lineNum << ": " << e.what() << std::endl;
            std::cerr << "Line content: " << line << std::endl;
            // Continue processing other lines
        }
    }
    
    file.close();
    return employees;
}

// Write employee metrics to CSV file
bool writeEmployeeMetrics(const std::string& filename, 
                         const std::vector<ML::EmployeeMetrics>& employees,
                         bool includePredictions = false,
                         const std::vector<double>& predictions = {}) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write header
    if (hasHeader) {
        file << "EmployeeID" << delimiter 
             << "Name" << delimiter 
             << "CodeCommits" << delimiter 
             << "LinesOfCode" << delimiter 
             << "CodeReviews" << delimiter 
             << "BugsFixed" << delimiter 
             << "DocumentationEdits" << delimiter 
             << "MeetingAttendance" << delimiter 
             << "TeamCollaboration" << delimiter 
             << "TechnicalDifficulty" << delimiter 
             << "ContributionScore";
        
        if (includePredictions) {
            file << delimiter << "PredictedContribution";
        }
        
        file << std::endl;
    }
    
    // Write data
    for (size_t i = 0; i < employees.size(); ++i) {
        const auto& emp = employees[i];
        
        file << escapeCSV(emp.employeeId) << delimiter 
             << escapeCSV(emp.name) << delimiter 
             << std::fixed << std::setprecision(2) << emp.codeCommits << delimiter 
             << emp.linesOfCode << delimiter 
             << emp.codeReviews << delimiter 
             << emp.bugsFixed << delimiter 
             << emp.documentationEdits << delimiter 
             << emp.meetingAttendance << delimiter 
             << emp.teamCollaboration << delimiter 
             << emp.technicalDifficulty << delimiter 
             << std::setprecision(4) << emp.contributionScore;
        
        if (includePredictions && i < predictions.size()) {
            file << delimiter << std::setprecision(4) << predictions[i];
        }
        
        file << std::endl;
    }
    
    file.close();
    return true;
}

// Generate a CSV template file with headers and example data
bool generateTemplateFile(const std::string& filename, int numExamples = 5) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "EmployeeID,Name,CodeCommits,LinesOfCode,CodeReviews,BugsFixed,"
         << "DocumentationEdits,MeetingAttendance,TeamCollaboration,"
         << "TechnicalDifficulty,ContributionScore" << std::endl;
    
    // Generate example rows
    std::vector<std::string> names = {
        "John Smith", "Jane Doe", "Michael Johnson", "Emily Davis", "David Wilson"
    };
    
    for (int i = 0; i < numExamples && i < static_cast<int>(names.size()); ++i) {
        file << "EMP" << (i + 1001) << ","
             << names[i] << ","
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
    
    file.close();
    
    std::cout << "Template file generated: " << filename << std::endl;
    std::cout << "Fill this file with your employee metrics data." << std::endl;
    return true;
}

// Function to append predictions to an existing CSV file
bool appendPredictions(const std::string& inputFile, const std::string& outputFile,
                       const std::vector<double>& predictions) {
    // Read the original file
    std::vector<ML::EmployeeMetrics> employees = readEmployeeMetrics(inputFile);
    
    if (employees.size() != predictions.size()) {
        std::cerr << "Error: Number of employees (" << employees.size() 
                  << ") doesn't match number of predictions (" 
                  << predictions.size() << ")" << std::endl;
        return false;
    }
    
    // Write with predictions
    return writeEmployeeMetrics(outputFile, employees, true, predictions);
}

private:
// Parse a single CSV line into fields
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    
    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == delimiter && !inQuotes) {
            // End of field
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    
    // Add the last field
    fields.push_back(field);
    
    return fields;
}

// Escape a string for CSV output
std::string escapeCSV(const std::string& str) {
    // If the string contains delimiter, quotes, or newlines, it needs to be quoted
    if (str.find(delimiter) != std::string::npos || 
        str.find('"') != std::string::npos || 
        str.find('\n') != std::string::npos) {
        
        // Double up any quotes in the string
        std::string escaped = str;
        size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos) {
            escaped.insert(pos, 1, '"');
            pos += 2;
        }
        
        // Wrap in quotes
        return "\"" + escaped + "\"";
    }
    
    return str;
}
};

/**
* @brief Batch Processor for processing multiple employees in batches
*/
class BatchProcessor {
private:
ML::EmployeeContributionPredictor& predictor;

public:
// Constructor
BatchProcessor(ML::EmployeeContributionPredictor& pred) : predictor(pred) {}

// Process a batch of employees from input CSV to output CSV
bool processBatch(const std::string& inputFile, const std::string& outputFile) {
    try {
        // Read employee data
        CSVHandler csvHandler;
        std::vector<ML::EmployeeMetrics> employees = csvHandler.readEmployeeMetrics(inputFile);
        
        if (employees.empty()) {
            std::cerr << "No employees loaded from file: " << inputFile << std::endl;
            return false;
        }
        
        std::cout << "Processing " << employees.size() << " employees..." << std::endl;
        
        // Generate predictions
        std::vector<double> predictions = predictor.predictContributions(employees);
        
        // Write predictions to output file
        return csvHandler.writeEmployeeMetrics(outputFile, employees, true, predictions);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during batch processing: " << e.what() << std::endl;
        return false;
    }
}

// Process a directory of CSV files
bool processDirectory(const std::string& inputDir, const std::string& outputDir) {
    try {
        // Create output directory if it doesn't exist
        if (!std::filesystem::exists(outputDir)) {
            std::filesystem::create_directories(outputDir);
        }
        
        size_t filesProcessed = 0;
        
        // Process each CSV file in the directory
        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".csv") {
                std::string inputFile = entry.path().string();
                std::string filename = entry.path().filename().string();
                std::string outputFile = outputDir + "/" + "processed_" + filename;
                
                std::cout << "Processing file: " << filename << std::endl;
                
                if (processBatch(inputFile, outputFile)) {
                    filesProcessed++;
                }
            }
        }
        
        std::cout << "Successfully processed " << filesProcessed << " files." << std::endl;
        return filesProcessed > 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during directory processing: " << e.what() << std::endl;
        return false;
    }
}

// Analyze directory of employee data and output statistical report
bool analyzeDirectory(const std::string& inputDir, const std::string& reportFile) {
    try {
        std::vector<ML::EmployeeMetrics> allEmployees;
        
        // Load all employees from all CSV files
        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".csv") {
                CSVHandler csvHandler;
                auto employees = csvHandler.readEmployeeMetrics(entry.path().string());
                allEmployees.insert(allEmployees.end(), employees.begin(), employees.end());
            }
        }
        
        if (allEmployees.empty()) {
            std::cerr << "No employee data found in directory: " << inputDir << std::endl;
            return false;
        }
        
        // Generate predictions for all employees
        std::vector<double> predictions = predictor.predictContributions(allEmployees);
        
        // Calculate statistics
        struct TeamStats {
            double avgActualContribution = 0.0;
            double avgPredictedContribution = 0.0;
            double minContribution = 1.0;
            double maxContribution = 0.0;
            size_t count = 0;
            
            void addEmployee(double actual, double predicted) {
                avgActualContribution = (avgActualContribution * count + actual) / (count + 1);
                avgPredictedContribution = (avgPredictedContribution * count + predicted) / (count + 1);
                minContribution = std::min(minContribution, actual);
                maxContribution = std::max(maxContribution, actual);
                count++;
            }
        };
        
        std::unordered_map<std::string, TeamStats> teamStats;
        
        // Extract team from employee ID (assuming format TEAM-XXX)
        for (size_t i = 0; i < allEmployees.size(); ++i) {
            std::string team = "Unknown";
            size_t dashPos = allEmployees[i].employeeId.find('-');
            if (dashPos != std::string::npos) {
                team = allEmployees[i].employeeId.substr(0, dashPos);
            }
            
            teamStats[team].addEmployee(allEmployees[i].contributionScore, predictions[i]);
        }
        
        // Generate report
        std::ofstream report(reportFile);
        if (!report.is_open()) {
            std::cerr << "Failed to open report file for writing: " << reportFile << std::endl;
            return false;
        }
        
        report << "Employee Contribution Analysis Report" << std::endl;
        report << "=====================================" << std::endl << std::endl;
        
        report << "Overall Statistics:" << std::endl;
        report << "Total Employees: " << allEmployees.size() << std::endl;
        
        // Calculate overall averages
        double totalActual = 0.0, totalPredicted = 0.0;
        double minActual = 1.0, maxActual = 0.0;
        
        for (size_t i = 0; i < allEmployees.size(); ++i) {
            totalActual += allEmployees[i].contributionScore;
            totalPredicted += predictions[i];
            minActual = std::min(minActual, allEmployees[i].contributionScore);
            maxActual = std::max(maxActual, allEmployees[i].contributionScore);
        }
        
        double avgActual = totalActual / allEmployees.size();
        double avgPredicted = totalPredicted / allEmployees.size();
        
        report << "Average Actual Contribution: " << std::fixed << std::setprecision(2)
               << (avgActual * 100.0) << "%" << std::endl;
        report << "Average Predicted Contribution: " << std::fixed << std::setprecision(2)
               << (avgPredicted * 100.0) << "%" << std::endl;
        report << "Minimum Contribution: " << std::fixed << std::setprecision(2)
               << (minActual * 100.0) << "%" << std::endl;
        report << "Maximum Contribution: " << std::fixed << std::setprecision(2)
               << (maxActual * 100.0) << "%" << std::endl << std::endl;
        
        report << "Team Statistics:" << std::endl;
        report << "---------------" << std::endl;
        
        for (const auto& [team, stats] : teamStats) {
            report << "Team: " << team << std::endl;
            report << "  Number of Employees: " << stats.count << std::endl;
            report << "  Average Actual Contribution: " << std::fixed << std::setprecision(2)
                   << (stats.avgActualContribution * 100.0) << "%" << std::endl;
            report << "  Average Predicted Contribution: " << std::fixed << std::setprecision(2)
                   << (stats.avgPredictedContribution * 100.0) << "%" << std::endl;
            report << "  Range: " << std::fixed << std::setprecision(2)
                   << (stats.minContribution * 100.0) << "% - "
                   << (stats.maxContribution * 100.0) << "%" << std::endl << std::endl;
        }
        
        report.close();
        
        std::cout << "Analysis report generated: " << reportFile << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during analysis: " << e.what() << std::endl;
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
    std::string mode = "train"; // Default mode
    std::string inputFile = "";
    std::string outputFile = "";
    
    if (argc > 1) {
        mode = argv[1];
    }
    
    if (argc > 2) {
        inputFile = argv[2];
    }
    
    if (argc > 3) {
        outputFile = argv[3];
    }
    
    // Create CSV handler
    CSVHandler csvHandler;
    
    // If no input file is specified, generate a template
    if (mode != "template" && inputFile.empty()) {
        std::cout << "No input file specified. Generating template CSV..." << std::endl;
        csvHandler.generateTemplateFile("employee_template.csv");
        std::cout << "Please fill in the template and rerun with: " << std::endl;
        std::cout << "./employee_nn train employee_template.csv" << std::endl;
        return 0;
    }
    
    // Handle different operation modes
    if (mode == "template") {
        // Generate template file
        std::string templateFile = inputFile.empty() ? "employee_template.csv" : inputFile;
        csvHandler.generateTemplateFile(templateFile);
        return 0;
    }
    else if (mode == "train" || mode == "train_and_predict") {
        // Load data from CSV
        std::vector<ML::EmployeeMetrics> data;
        
        if (std::filesystem::exists(inputFile)) {
            std::cout << "Loading data from: " << inputFile << std::endl;
            data = csvHandler.readEmployeeMetrics(inputFile);
            std::cout << "Loaded " << data.size() << " employee records." << std::endl;
        } else {
            // If input file doesn't exist, generate synthetic data
            std::cout << "Input file not found, generating synthetic data..." << std::endl;
            
            // Generate synthetic data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> idDist(10000, 99999);
            std::uniform_int_distribution<> commitsDist(0, 100);
            std::uniform_int_distribution<> locDist(0, 5000);
            std::uniform_int_distribution<> reviewsDist(0, 50);
            std::uniform_int_distribution<> bugsDist(0, 30);
            std::uniform_int_distribution<> docsDist(0, 200);
            std::uniform_real_distribution<> meetingsDist(50.0, 100.0);
            std::uniform_real_distribution<> collabDist(1.0, 10.0);
            std::uniform_real_distribution<> difficultyDist(1.0, 10.0);
            
            // Generate 500 synthetic employees
            for (int i = 0; i < 500; ++i) {
                std::string id = "EMP" + std::to_string(idDist(gen));
                std::string name = "Employee" + std::to_string(i);
                
                double commits = commitsDist(gen);
                double loc = locDist(gen);
                double reviews = reviewsDist(gen);
                double bugs = bugsDist(gen);
                double docs = docsDist(gen);
                double meetings = meetingsDist(gen);
                double collab = collabDist(gen);
                double difficulty = difficultyDist(gen);
                
                // Calculate a contribution score based on metrics (simulated formula)
                double contributionScore = 0.0;
                contributionScore += 0.15 * (commits / 100.0);
                contributionScore += 0.10 * (loc / 5000.0);
                contributionScore += 0.15 * (reviews / 50.0);
                contributionScore += 0.20 * (bugs / 30.0);
                contributionScore += 0.05 * (docs / 200.0);
                contributionScore += 0.10 * (meetings / 100.0);
                contributionScore += 0.15 * (collab / 10.0);
                contributionScore += 0.10 * (difficulty / 10.0);
                
                // Add some randomness (noise)
                std::normal_distribution<> noise(0.0, 0.05);
                contributionScore = std::min(1.0, std::max(0.0, contributionScore + noise(gen)));
                
                ML::EmployeeMetrics metrics(
                    id, name, commits, loc, reviews, bugs, docs, 
                    meetings, collab, difficulty, contributionScore
                );
                
                data.push_back(metrics);
            }
            
            // Save synthetic data to CSV
            std::string syntheticFile = "synthetic_employees.csv";
            csvHandler.writeEmployeeMetrics(syntheticFile, data);
            std::cout << "Generated " << data.size() << " synthetic employee records and saved to "
                      << syntheticFile << std::endl;
        }
        
        if (data.empty()) {
            std::cerr << "No data to train on. Exiting." << std::endl;
            return 1;
        }
        
        // Split data for training and testing
        std::vector<ML::EmployeeMetrics> trainingData, testData;
        size_t splitIndex = static_cast<size_t>(data.size() * 0.8); // 80% training, 20% testing
        trainingData.insert(trainingData.end(), data.begin(), data.begin() + splitIndex);
        testData.insert(testData.end(), data.begin() + splitIndex, data.end());
        
        std::cout << "Training set size: " << trainingData.size() << std::endl;
        std::cout << "Test set size: " << testData.size() << std::endl;
        
        // Configure neural network
        ML::NeuralNetConfig config;
        config.layerSizes = {8, 16, 8, 1};  // 8 input features, 2 hidden layers, 1 output
        config.activations = {
            std::make_shared<ML::ReLUActivation>(),      // First hidden layer
            std::make_shared<ML::ReLUActivation>(),      // Second hidden layer
            std::make_shared<ML::SigmoidActivation>()    // Output layer (sigmoid for 0-1 output)
        };
        config.learningRate = 0.01;
        config.maxEpochs = 2000;
        config.errorThreshold = 0.0001;
        config.verbose = true;
        
        // Create and train predictor
        std::cout << "Creating neural network model..." << std::endl;
        ML::EmployeeContributionPredictor predictor;
        predictor.configureNetwork(config);
        
        std::cout << "Training neural network..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        double validationError = predictor.train(trainingData);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto trainingTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        
        std::cout << "Training completed in " << trainingTime << " seconds." << std::endl;
        std::cout << "Validation error: " << validationError << std::endl;
        
        // Save the model
        std::cout << "Saving model..." << std::endl;
        if (predictor.saveModel("employee_model.bin", "employee_norm.csv")) {
            std::cout << "Model saved successfully." << std::endl;
        } else {
            std::cout << "Failed to save model." << std::endl;
        }
        
        // Evaluate model performance
        std::cout << "Evaluating model on test data..." << std::endl;
        
        std::vector<double> predictions = predictor.predictContributions(testData);
        
        // Calculate error metrics
        double mse = 0.0, mae = 0.0;
        for (size_t i = 0; i < testData.size(); ++i) {
            double error = predictions[i] - testData[i].contributionScore;
            mse += error * error;
            mae += std::abs(error);
        }
        
        mse /= testData.size();
        mae /= testData.size();
        double rmse = std::sqrt(mse);
        
        std::cout << "Test set metrics:" << std::endl;
        std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
        std::cout << "Root Mean Squared Error (RMSE): " << rmse << std::endl;
        std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
        
        // If train_and_predict mode, also make predictions on a new file
        if (mode == "train_and_predict" && !outputFile.empty()) {
            std::cout << "Making predictions on: " << outputFile << std::endl;
            
            // Create batch processor
            BatchProcessor batchProcessor(predictor);
            batchProcessor.processBatch(outputFile, "predictions_" + outputFile);
        }
    }
    else if (mode == "predict") {
        // Load the model
        ML::EmployeeContributionPredictor predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!predictor.loadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Make predictions
        if (std::filesystem::is_directory(inputFile)) {
            // Process an entire directory
            std::cout << "Processing directory: " << inputFile << std::endl;
            
            std::string outputDir = outputFile.empty() ? "predictions" : outputFile;
            
            BatchProcessor batchProcessor(predictor);
            batchProcessor.processDirectory(inputFile, outputDir);
        } else {
            // Process a single file
            std::cout << "Processing file: " << inputFile << std::endl;
            
            std::string outputFilename = outputFile.empty() ? 
                "predictions_" + inputFile : outputFile;
            
            BatchProcessor batchProcessor(predictor);
            batchProcessor.processBatch(inputFile, outputFilename);
        }
    }
    else if (mode == "analyze") {
        // Load the model
        ML::EmployeeContributionPredictor predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!predictor.loadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Analyze data
        std::string reportFile = outputFile.empty() ? "contribution_analysis.txt" : outputFile;
        
        if (std::filesystem::is_directory(inputFile)) {
            // Analyze an entire directory
            std::cout << "Analyzing directory: " << inputFile << std::endl;
            
            BatchProcessor batchProcessor(predictor);
            batchProcessor.analyzeDirectory(inputFile, reportFile);
        } else {
            // Analyze a single file
            std::cout << "Analyzing file: " << inputFile << std::endl;
            
            // Load employee data
            std::vector<ML::EmployeeMetrics> employees = csvHandler.readEmployeeMetrics(inputFile);
            
            if (employees.empty()) {
                std::cerr << "No employees loaded from file: " << inputFile << std::endl;
                return 1;
            }
            
            // Make predictions
            std::vector<double> predictions = predictor.predictContributions(employees);
            
            // Print detailed analysis of each employee
            for (size_t i = 0; i < employees.size(); ++i) {
                std::cout << "Employee: " << employees[i].name << " (ID: " << employees[i].employeeId << ")" << std::endl;
                std::cout << "  Metrics:" << std::endl;
                std::cout << "    Code Commits: " << employees[i].codeCommits << std::endl;
                std::cout << "    Lines of Code: " << employees[i].linesOfCode << std::endl;
                std::cout << "    Code Reviews: " << employees[i].codeReviews << std::endl;
                std::cout << "    Bugs Fixed: " << employees[i].bugsFixed << std::endl;
                std::cout << "    Documentation Edits: " << employees[i].documentationEdits << std::endl;
                std::cout << "    Meeting Attendance: " << employees[i].meetingAttendance << "%" << std::endl;
                std::cout << "    Team Collaboration: " << employees[i].teamCollaboration << "/10" << std::endl;
                std::cout << "    Technical Difficulty: " << employees[i].technicalDifficulty << "/10" << std::endl;
                std::cout << "  Actual Contribution: " << std::fixed << std::setprecision(2) 
                          << (employees[i].contributionScore * 100.0) << "%" << std::endl;
                std::cout << "  Predicted Contribution: " << std::fixed << std::setprecision(2) 
                          << (predictions[i] * 100.0) << "%" << std::endl;
                std::cout << "  Difference: " << std::fixed << std::setprecision(2) 
                          << ((predictions[i] - employees[i].contributionScore) * 100.0) << "%" << std::endl;
                std::cout << "------------------------------------" << std::endl;
            }
            
            // Save results to a report file
            std::ofstream report(reportFile);
            report << "Employee Contribution Analysis" << std::endl;
            report << "=============================" << std::endl << std::endl;
            
            for (size_t i = 0; i < employees.size(); ++i) {
                report << "Employee: " << employees[i].name << " (ID: " << employees[i].employeeId << ")" << std::endl;
                report << "  Actual Contribution: " << std::fixed << std::setprecision(2) 
                       << (employees[i].contributionScore * 100.0) << "%" << std::endl;
                report << "  Predicted Contribution: " << std::fixed << std::setprecision(2) 
                       << (predictions[i] * 100.0) << "%" << std::endl;
                report << "  Difference: " << std::fixed << std::setprecision(2) 
                       << ((predictions[i] - employees[i].contributionScore) * 100.0) << "%" << std::endl;
                report << "------------------------------------" << std::endl;
            }
            
            report.close();
            std::cout << "Analysis saved to: " << reportFile << std::endl;
        }
    }
    else if (mode == "interactive") {
        // Load the model
        ML::EmployeeContributionPredictor predictor;
        std::cout << "Loading model..." << std::endl;
        
        if (!predictor.loadModel("employee_model.bin", "employee_norm.csv")) {
            std::cerr << "Failed to load model. Please train the model first." << std::endl;
            return 1;
        }
        
        // Interactive mode for entering employee metrics
        std::cout << "Interactive Mode - Enter employee metrics:" << std::endl;
        
        std::string employeeId, name;
        double codeCommits, linesOfCode, codeReviews, bugsFixed;
        double documentationEdits, meetingAttendance, teamCollaboration, technicalDifficulty;
        
        std::cout << "Employee ID: ";
        std::cin >> employeeId;
        std::cin.ignore(); // Clear newline
        
        std::cout << "Name: ";
        std::getline(std::cin, name);
        
        std::cout << "Code Commits: ";
        std::cin >> codeCommits;
        
        std::cout << "Lines of Code: ";
        std::cin >> linesOfCode;
        
        std::cout << "Code Reviews: ";
        std::cin >> codeReviews;
        
        std::cout << "Bugs Fixed: ";
        std::cin >> bugsFixed;
        
        std::cout << "Documentation Edits: ";
        std::cin >> documentationEdits;
        
        std::cout << "Meeting Attendance (%): ";
        std::cin >> meetingAttendance;
        
        std::cout << "Team Collaboration (0-10): ";
        std::cin >> teamCollaboration;
        
        std::cout << "Technical Difficulty (0-10): ";
        std::cin >> technicalDifficulty;
        
        // Create employee object
        ML::EmployeeMetrics employee(
            employeeId, name, codeCommits, linesOfCode, codeReviews,
            bugsFixed, documentationEdits, meetingAttendance,
            teamCollaboration, technicalDifficulty
        );
        
        // Make prediction
        double contribution = predictor.predictContribution(employee);
        
        std::cout << "\nPrediction Results:" << std::endl;
        std::cout << "Employee: " << name << " (ID: " << employeeId << ")" << std::endl;
        std::cout << "Predicted Contribution: " << std::fixed << std::setprecision(2) 
                 << (contribution * 100.0) << "%" << std::endl;
        
        // Provide some analysis
        std::cout << "\nContribution Analysis:" << std::endl;
        
        if (contribution >= 0.9) {
            std::cout << "Exceptional performer - consider for leadership roles or mentoring opportunities." << std::endl;
        } else if (contribution >= 0.75) {
            std::cout << "Strong performer - valuable team member with consistent high output." << std::endl;
        } else if (contribution >= 0.6) {
            std::cout << "Good performer - reliable contributor with room for growth." << std::endl;
        } else if (contribution >= 0.4) {
            std::cout << "Average performer - may benefit from targeted coaching in specific areas." << std::endl;
        } else if (contribution >= 0.25) {
            std::cout << "Below average performer - consider performance improvement plan." << std::endl;
        } else {
            std::cout << "Struggling performer - needs immediate attention and support." << std::endl;
        }
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
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
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}

return 0;
}#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <optional>
#include <functional>

namespace ML {

/**
* @brief Structure to hold employee metrics data
*/
struct EmployeeMetrics {
// Employee identification
std::string employeeId;
std::string name;

// Input features for the neural network
double codeCommits;         // Number of code commits
double linesOfCode;         // Lines of code written
double codeReviews;         // Number of code reviews performed
double bugsFixed;           // Number of bugs fixed
double documentationEdits;  // Contribution to documentation
double meetingAttendance;   // Percentage of meetings attended
double teamCollaboration;   // Peer rating of collaboration (0-10)
double technicalDifficulty; // Average difficulty of assigned tasks (0-10)

// Target output (if known, for training)
double contributionScore;   // Overall contribution score (0-1)

// Constructor with validation
EmployeeMetrics(
    std::string id = "",
    std::string empName = "",
    double commits = 0.0,
    double loc = 0.0,
    double reviews = 0.0,
    double bugs = 0.0,
    double docs = 0.0,
    double meetings = 0.0,
    double collab = 0.0,
    double difficulty = 0.0,
    double score = 0.0
) : employeeId(std::move(id)),
    name(std::move(empName)),
    codeCommits(commits),
    linesOfCode(loc),
    codeReviews(reviews),
    bugsFixed(bugs),
    documentationEdits(docs),
    meetingAttendance(meetings),
    teamCollaboration(collab),
    technicalDifficulty(difficulty),
    contributionScore(score) {
    validate();
}

// Validate the metrics data
void validate() const {
    if (meetingAttendance < 0.0 || meetingAttendance > 100.0) {
        throw std::invalid_argument("Meeting attendance must be between 0 and 100 percent");
    }
    if (teamCollaboration < 0.0 || teamCollaboration > 10.0) {
        throw std::invalid_argument("Team collaboration must be between 0 and 10");
    }
    if (technicalDifficulty < 0.0 || technicalDifficulty > 10.0) {
        throw std::invalid_argument("Technical difficulty must be between 0 and 10");
    }
    if (contributionScore < 0.0 || contributionScore > 1.0) {
        throw std::invalid_argument("Contribution score must be between 0 and 1");
    }
}

// Convert metrics to input vector for the neural network
std::vector<double> toInputVector() const {
    return {
        codeCommits,
        linesOfCode,
        codeReviews,
        bugsFixed, 
        documentationEdits,
        meetingAttendance,
        teamCollaboration,
        technicalDifficulty
    };
}
};

/**
* @brief Activation functions for neurons
*/
class ActivationFunction {
public:
virtual ~ActivationFunction() = default;
virtual double activate(double x) const = 0;
virtual double derivative(double x) const = 0;
virtual std::string getName() const = 0;
};

class SigmoidActivation : public ActivationFunction {
public:
double activate(double x) const override {
    return 1.0 / (1.0 + std::exp(-x));
}

double derivative(double x) const override {
    double sigmoid = activate(x);
    return sigmoid * (1.0 - sigmoid);
}

std::string getName() const override {
    return "Sigmoid";
}
};

class ReLUActivation : public ActivationFunction {
public:
double activate(double x) const override {
    return std::max(0.0, x);
}

double derivative(double x) const override {
    return x > 0.0 ? 1.0 : 0.0;
}

std::string getName() const override {
    return "ReLU";
}
};

class TanhActivation : public ActivationFunction {
public:
double activate(double x) const override {
    return std::tanh(x);
}

double derivative(double x) const override {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

std::string getName() const override {
    return "Tanh";
}
};

/**
* @brief Neuron class representing a single unit in the neural network
*/
class Neuron {
private:
std::vector<double> weights;
double bias;
double output;
double delta;
std::shared_ptr<ActivationFunction> activationFunc;

public:
// Constructor with random initialization
Neuron(size_t numInputs, std::shared_ptr<ActivationFunction> activation)
    : bias(0.0), output(0.0), delta(0.0), activationFunc(std::move(activation)) {
    // Xavier/Glorot initialization for better convergence
    std::random_device rd;
    std::mt19937 gen(rd());
    double factor = 2.0 / std::sqrt(static_cast<double>(numInputs));
    std::uniform_real_distribution<double> distribution(-factor, factor);
    
    weights.resize(numInputs);
    for (auto& weight : weights) {
        weight = distribution(gen);
    }
    bias = distribution(gen);
}

// Forward pass: calculate the output of this neuron
double feedForward(const std::vector<double>& inputs) const {
    if (inputs.size() != weights.size()) {
        throw std::invalid_argument("Input size doesn't match weight size in neuron");
    }
    
    double sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }
    
    output = activationFunc->activate(sum);
    return output;
}

// Update weights and bias during backpropagation
void updateWeights(const std::vector<double>& inputs, double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += learningRate * delta * inputs[i];
    }
    bias += learningRate * delta;
}

// Getters and setters
double getOutput() const { return output; }
void setOutput(double val) { output = val; }
void setDelta(double d) { delta = d; }
double getDelta() const { return delta; }
const std::vector<double>& getWeights() const { return weights; }
void setWeights(const std::vector<double>& w) { weights = w; }
double getBias() const { return bias; }
void setBias(double b) { bias = b; }
std::shared_ptr<ActivationFunction> getActivationFunction() const { return activationFunc; }
};

/**
* @brief Layer class representing a collection of neurons
*/
class Layer {
private:
std::vector<Neuron> neurons;
mutable std::vector<double> outputs;
std::string layerName;

public:
// Constructor
Layer(size_t numNeurons, size_t numInputsPerNeuron, std::shared_ptr<ActivationFunction> activation, 
      const std::string& name = "Hidden Layer")
    : layerName(name) {
    neurons.reserve(numNeurons);
    outputs.resize(numNeurons);
    
    for (size_t i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron, activation);
    }
}

// Forward pass: process inputs through all neurons in this layer
std::vector<double> feedForward(const std::vector<double>& inputs) const {
    for (size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].feedForward(inputs);
    }
    return outputs;
}

// Backpropagation for output layer
void calculateOutputLayerDeltas(const std::vector<double>& targets) {
    if (targets.size() != neurons.size()) {
        throw std::invalid_argument("Target size doesn't match neuron count in output layer");
    }
    
    for (size_t i = 0; i < neurons.size(); ++i) {
        double output = neurons[i].getOutput();
        // Error derivative for MSE: (output - target)
        double error = output - targets[i];
        // Delta = error * derivative of activation function
        double delta = error * neurons[i].getActivationFunction()->derivative(output);
        neurons[i].setDelta(delta);
    }
}

// Backpropagation for hidden layers
void calculateHiddenLayerDeltas(const Layer& nextLayer) {
    for (size_t i = 0; i < neurons.size(); ++i) {
        double sum = 0.0;
        // Sum up all the deltas from the next layer weighted by connections
        for (size_t j = 0; j < nextLayer.getNeurons().size(); ++j) {
            sum += nextLayer.getNeurons()[j].getDelta() * 
                   nextLayer.getNeurons()[j].getWeights()[i];
        }
        // Delta = sum * derivative of activation function
        double output = neurons[i].getOutput();
        double delta = sum * neurons[i].getActivationFunction()->derivative(output);
        neurons[i].setDelta(delta);
    }
}

// Update all neuron weights in this layer
void updateWeights(const std::vector<double>& inputs, double learningRate) {
    for (auto& neuron : neurons) {
        neuron.updateWeights(inputs, learningRate);
    }
}

// Getters
const std::vector<Neuron>& getNeurons() const { return neurons; }
std::vector<Neuron>& getNeurons() { return neurons; }
const std::vector<double>& getOutputs() const { return outputs; }
const std::string& getName() const { return layerName; }
};

/**
* @brief Configuration for the neural network
*/
struct NeuralNetConfig {
std::vector<size_t> layerSizes;
std::vector<std::shared_ptr<ActivationFunction>> activations;
double learningRate;
size_t maxEpochs;
double errorThreshold;
bool verbose;

// Constructor with default values
NeuralNetConfig(
    std::vector<size_t> sizes = {8, 12, 8, 1},
    std::vector<std::shared_ptr<ActivationFunction>> acts = {
        std::make_shared<ReLUActivation>(),
        std::make_shared<ReLUActivation>(),
        std::make_shared<SigmoidActivation>()
    },
    double lr = 0.01,
    size_t epochs = 1000,
    double threshold = 0.001,
    bool verb = false
) : layerSizes(std::move(sizes)),
    activations(std::move(acts)),
    learningRate(lr),
    maxEpochs(epochs),
    errorThreshold(threshold),
    verbose(verb) {
    validate();
}

// Validate configuration
void validate() const {
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Neural network must have at least input and output layers");
    }
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Number of activation functions must match number of layers minus input layer");
    }
    if (learningRate <= 0.0 || learningRate > 1.0) {
        throw std::invalid_argument("Learning rate must be between 0 and 1");
    }
    if (maxEpochs == 0) {
        throw std::invalid_argument("Maximum epochs must be greater than 0");
    }
    if (errorThreshold <= 0.0) {
        throw std::invalid_argument("Error threshold must be greater than 0");
    }
}
};

// Forward declaration
class DataProcessor;

/**
* @brief Neural Network class for predicting employee contribution
*/
class NeuralNetwork {
private:
std::vector<Layer> layers;
double learningRate;
size_t maxEpochs;
double errorThreshold;
bool verbose;
mutable std::shared_mutex networkMutex; // For thread-safety during prediction and training

// Statistics
double totalError;
size_t trainingEpochs;
double validationAccuracy;

public:
// Constructor
NeuralNetwork(const NeuralNetConfig& config) 
    : learningRate(config.learningRate),
      maxEpochs(config.maxEpochs),
      errorThreshold(config.errorThreshold),
      verbose(config.verbose),
      totalError(0.0),
      trainingEpochs(0),
      validationAccuracy(0.0) {
    
    // Create layers based on configuration
    for (size_t i = 0; i < config.layerSizes.size() - 1; ++i) {
        size_t inputSize = config.layerSizes[i];
        size_t outputSize = config.layerSizes[i + 1];
        std::string layerName;
        
        if (i == 0) {
            layerName = "Input Layer";
        } else if (i == config.layerSizes.size() - 2) {
            layerName = "Output Layer";
        } else {
            layerName = "Hidden Layer " + std::to_string(i);
        }
        
        layers.emplace_back(outputSize, inputSize, config.activations[i], layerName);
    }
}

// Forward pass through the entire network
std::vector<double> feedForward(const std::vector<double>& inputs) const {
    // Use shared lock for read-only access to network
    std::shared_lock<std::shared_mutex> lock(networkMutex);
    
    std::vector<double> outputs = inputs;
    for (const auto& layer : layers) {
        outputs = layer.feedForward(outputs);
    }
    return outputs;
}

// Backpropagation to train the network
void backpropagate(const std::vector<double>& inputs, const std::vector<double>& targets) {
    // Use exclusive lock for modifying network weights
    std::unique_lock<std::shared_mutex> lock(networkMutex);
    
    // Forward pass to get outputs
    std::vector<double> currentInputs = inputs;
    std::vector<std::vector<double>> layerInputs;
    layerInputs.push_back(currentInputs);
    
    for (auto& layer : layers) {
        currentInputs = layer.feedForward(currentInputs);
        layerInputs.push_back(currentInputs);
    }
    
    // Calculate deltas for output layer
    layers.back().calculateOutputLayerDeltas(targets);
    
    // Calculate deltas for hidden layers
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
        layers[i].calculateHiddenLayerDeltas(layers[i + 1]);
    }
    
    // Update weights for all layers
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].updateWeights(layerInputs[i], learningRate);
    }
}

// Train the network with a dataset
double train(const std::vector<std::vector<double>>& trainingInputs, 
             const std::vector<std::vector<double>>& trainingTargets) {
    if (trainingInputs.size() != trainingTargets.size() || trainingInputs.empty()) {
        throw std::invalid_argument("Training inputs and targets must have the same non-zero size");
    }
    
    double prevError = std::numeric_limits<double>::max();
    
    // Train for specified number of epochs or until error threshold is reached
    for (size_t epoch = 0; epoch < maxEpochs; ++epoch) {
        double epochError = 0.0;
        
        // Create indices for random shuffling
        std::vector<size_t> indices(trainingInputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Process each training example
        for (size_t idx : indices) {
            // Forward pass
            std::vector<double> outputs = feedForward(trainingInputs[idx]);
            
            // Calculate error (MSE)
            double sampleError = 0.0;
            for (size_t i = 0; i < outputs.size(); ++i) {
                double error = outputs[i] - trainingTargets[idx][i];
                sampleError += error * error;
            }
            sampleError /= outputs.size();
            epochError += sampleError;
            
            // Backpropagation
            backpropagate(trainingInputs[idx], trainingTargets[idx]);
        }
        
        // Average error across all samples
        epochError /= trainingInputs.size();
        
        // Store statistics
        totalError = epochError;
        trainingEpochs = epoch + 1;
        
        // Print progress if verbose
        if (verbose && (epoch % 100 == 0 || epoch == maxEpochs - 1)) {
            std::cout << "Epoch " << epoch + 1 << "/" << maxEpochs
                      << ", Error: " << epochError << std::endl;
        }
        
        // Check for convergence
        if (std::abs(prevError - epochError) < errorThreshold) {
            if (verbose) {
                std::cout << "Converged after " << epoch + 1 << " epochs." << std::endl;
            }
            break;
        }
        
        prevError = epochError;
    }
    
    return totalError;
}

// Evaluate the network on validation data
double validate(const std::vector<std::vector<double>>& validationInputs, 
                const std::vector<std::vector<double>>& validationTargets) {
    if (validationInputs.size() != validationTargets.size() || validationInputs.empty()) {
        throw std::invalid_argument("Validation inputs and targets must have the same non-zero size");
    }
    
    double totalError = 0.0;
    size_t correctPredictions = 0;
    
    for (size_t i = 0; i < validationInputs.size(); ++i) {
        std::vector<double> outputs = feedForward(validationInputs[i]);
        
        // Calculate error (MSE)
        double sampleError = 0.0;
        for (size_t j = 0; j < outputs.size(); ++j) {
            double error = outputs[j] - validationTargets[i][j];
            sampleError += error * error;
        }
        sampleError /= outputs.size();
        totalError += sampleError;
        
        // For binary classification tasks with threshold of 0.5
        if (outputs.size() == 1 && validationTargets[i].size() == 1) {
            if ((outputs[0] > 0.5 && validationTargets[i][0] > 0.5) ||
                (outputs[0] <= 0.5 && validationTargets[i][0] <= 0.5)) {
                correctPredictions++;
            }
        }
    }
    
    // Average error across all samples
    totalError /= validationInputs.size();
    
    // Calculate accuracy for classification tasks
    if (validationTargets[0].size() == 1) {
        validationAccuracy = static_cast<double>(correctPredictions) / validationInputs.size();
        if (verbose) {
            std::cout << "Validation accuracy: " << (validationAccuracy * 100.0) << "%" << std::endl;
        }
    }
    
    return totalError;
}

// Predict employee contribution based on metrics
double predictContribution(const EmployeeMetrics& metrics) const {
    try {
        std::vector<double> inputs = metrics.toInputVector();
        std::vector<double> outputs = feedForward(inputs);
        
        if (outputs.empty()) {
            throw std::runtime_error("Neural network produced no output");
        }
        
        return outputs[0]; // Assuming single output neuron for contribution score
    } catch (const std::exception& e) {
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        return -1.0; // Indicate error
    }
}

// Save the model to a file
bool saveModel(const std::string& filename) const {
    try {
        std::ofstream file(filename, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            return false;
        }
        
        // Use shared lock for reading model parameters
        std::shared_lock<std::shared_mutex> lock(networkMutex);
        
        // Write number of layers
        size_t numLayers = layers.size();
        file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
        
        // Write each layer
        for (const auto& layer : layers) {
            // Write number of neurons
            size_t numNeurons = layer.getNeurons().size();
            file.write(reinterpret_cast<const char*>(&numNeurons), sizeof(numNeurons));
            
            // Write each neuron
            for (const auto& neuron : layer.getNeurons()) {
                // Write activation function name
                std::string activationName = neuron.getActivationFunction()->getName();
                size_t nameLength = activationName.size();
                file.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
                file.write(activationName.c_str(), nameLength);
                
                // Write bias
                double bias = neuron.getBias();
                file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
                
                // Write weights
                const auto& weights = neuron.getWeights();
                size_t numWeights = weights.size();
                file.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
                file.write(reinterpret_cast<const char*>(weights.data()), 
                           numWeights * sizeof(double));
            }
        }
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

// Load the model from a file
bool loadModel(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            return false;
        }
        
        // Use exclusive lock for writing model parameters
        std::unique_lock<std::shared_mutex> lock(networkMutex);
        
        // Read number of layers
        size_t numLayers;
        file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
        
        // Temporary storage for new layers
        std::vector<Layer> newLayers;
        
        // Read each layer
        for (size_t i = 0; i < numLayers; ++i) {
            // Read number of neurons
            size_t numNeurons;
            file.read(reinterpret_cast<char*>(&numNeurons), sizeof(numNeurons));
            
            // Create a temporary layer - we'll need to determine input size first
            std::vector<std::tuple<std::shared_ptr<ActivationFunction>, double, std::vector<double>>> neuronData;
            
            // Read each neuron
            for (size_t j = 0; j < numNeurons; ++j) {
                // Read activation function name
                size_t nameLength;
                file.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
                std::string activationName(nameLength, ' ');
                file.read(&activationName[0], nameLength);
                
                // Create activation function
                std::shared_ptr<ActivationFunction> activation;
                if (activationName == "Sigmoid") {
                    activation = std::make_shared<SigmoidActivation>();
                } else if (activationName == "ReLU") {
                    activation = std::make_shared<ReLUActivation>();
                } else if (activationName == "Tanh") {
                    activation = std::make_shared<TanhActivation>();
                } else {
                    throw std::runtime_error("Unknown activation function: " + activationName);
                }
                
                // Read bias
                double bias;
                file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
                
                // Read weights
                size_t numWeights;
                file.read(reinterpret_cast<char*>(&numWeights), sizeof(numWeights));
                std::vector<double> weights(numWeights);
                file.read(reinterpret_cast<char*>(weights.data()), 
                          numWeights * sizeof(double));
                
                // Store neuron data
                neuronData.emplace_back(activation, bias, std::move(weights));
            }
            
            // Create layer with proper dimensions
            size_t inputSize = std::get<2>(neuronData[0]).size();
            std::string layerName = (i == 0) ? "Input Layer" : 
                                   (i == numLayers - 1) ? "Output Layer" : 
                                   "Hidden Layer " + std::to_string(i);
            
            // Use first neuron's activation for the layer (all neurons in a layer typically use the same)
            Layer layer(numNeurons, inputSize, std::get<0>(neuronData[0]), layerName);
            
            // Set weights and biases for each neuron
            for (size_t j = 0; j < numNeurons; ++j) {
                layer.getNeurons()[j].setBias(std::get<1>(neuronData[j]));
                layer.getNeurons()[j].setWeights(std::get<2>(neuronData[j]));
            }
            
            newLayers.push_back(std::move(layer));
        }
        
        // Replace existing layers with loaded ones
        layers = std::move(newLayers);
        
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

// Get statistics
double getTotalError() const { return totalError; }
size_t getTrainingEpochs() const { return trainingEpochs; }
double getValidationAccuracy() const { return validationAccuracy; }
};

/**
* @brief Data processor for normalizing and preparing employee metrics
*/
class DataProcessor {
private:
// Statistics for each feature for normalization
std::unordered_map<std::string, double> minValues;
std::unordered_map<std::string, double> maxValues;
std::unordered_map<std::string, double> meanValues;
std::unordered_map<std::string, double> stdDevValues;

// Feature names
std::vector<std::string> featureNames = {
    "codeCommits", "linesOfCode", "codeReviews", "bugsFixed", 
    "documentationEdits", "meetingAttendance", "teamCollaboration", 
    "technicalDifficulty", "contributionScore"
};

// Normalization method
enum class NormMethod { MinMax, ZScore };
NormMethod normMethod;

public:
// Constructor
DataProcessor(NormMethod method = NormMethod::MinMax) : normMethod(method) {
    // Initialize statistics
    for (const auto& feature : featureNames) {
        minValues[feature] = std::numeric_limits<double>::max();
        maxValues[feature] = std::numeric_limits<double>::lowest();
        meanValues[feature] = 0.0;
        stdDevValues[feature] = 0.0;
    }
}

// Compute statistics from a dataset
void computeStatistics(const std::vector<EmployeeMetrics>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute statistics from empty dataset");
    }
    
    // Reset statistics
    for (const auto& feature : featureNames) {
        minValues[feature] = std::numeric_limits<double>::max();
        maxValues[feature] = std::numeric_limits<double>::lowest();
        meanValues[feature] = 0.0;
        stdDevValues[feature] = 0.0;
    }
    
    // Compute min, max, and mean
    for (const auto& employee : data) {
        updateStatisticsFromEmployee(employee, true);
    }
    
    // Compute standard deviation
    if (normMethod == NormMethod::Z