#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm> // std::max_element

// Matrix를 2차원 벡터로 표현합니다.
// 예를 들어, 3x4 행렬은 3개의 행(vector)이 있고 각 행은 4개의 double 값을 갖습니다.
using Matrix = std::vector<std::vector<double>>;

/**
 * @brief 두 행렬 A와 B의 곱셈을 수행하는 함수입니다.
 *
 * 수학적으로 행렬 곱셈은 아래와 같이 정의됩니다.
 * 만약 A가 (n x m) 크기이고 B가 (m x p) 크기라면,
 * 결과 행렬 C의 각 원소 C(i, j)는 다음과 같이 계산됩니다.
 *
 *     C(i, j) = A(i,1) * B(1,j) + A(i,2) * B(2,j) + ... + A(i,m) * B(m,j)
 *
 * @param A 행렬 A (예: Query 행렬 또는 Attention weight 계산을 위한 행렬)
 * @param B 행렬 B (예: Key의 전치행렬 또는 Value 행렬)
 * @return Matrix A와 B의 곱셈 결과인 행렬 C.
 * @throws std::invalid_argument 만약 행렬의 크기가 곱셈에 적합하지 않다면 예외를 발생시킵니다.
 *
 * 시간 복잡도: O(n * m * p)
 * 메모리 복잡도: O(n * p)
 */
Matrix matMul(const Matrix &A, const Matrix &B) {
    // A의 열의 수는 B의 행의 수와 같아야 곱셈이 가능함
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("행렬 곱셈에 적합하지 않은 차원입니다.");
    }
    
    size_t numRowsA = A.size();        // A의 행의 수 (n)
    size_t numColsA = A[0].size();       // A의 열의 수 (m) → B의 행의 수와 동일해야 함
    size_t numColsB = B[0].size();       // B의 열의 수 (p)
    
    // 결과 행렬 C의 크기는 (n x p) 입니다.
    Matrix result(numRowsA, std::vector<double>(numColsB, 0.0));
    
    // 행렬 곱셈 수행:
    // 외부 루프(i)는 결과 행렬의 각 행에 대해,
    // 내부 루프(k)는 A의 각 열(B의 행에 해당)마다 곱셈 후 합산,
    // 가장 안쪽 루프(j)는 결과 행렬의 각 열에 대해 계산합니다.
    for (size_t i = 0; i < numRowsA; i++) {
        for (size_t k = 0; k < numColsA; k++) {
            for (size_t j = 0; j < numColsB; j++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

/**
 * @brief 주어진 행렬의 전치(transpose)를 계산합니다.
 *
 * 행렬의 전치는 행과 열을 서로 바꾸는 것을 의미합니다.
 * 즉, 원래 행렬의 (i, j) 원소가 전치 행렬에서는 (j, i) 원소가 됩니다.
 *
 * @param matrix 입력 행렬.
 * @return Matrix 전치된 행렬.
 * @throws std::invalid_argument 만약 행렬의 각 행의 길이가 일관되지 않으면 예외 발생.
 *
 * 시간 복잡도: O(n * m)
 * 메모리 복잡도: O(m * n)
 */
Matrix transpose(const Matrix &matrix) {
    if (matrix.empty()) return Matrix{};
    
    size_t numRows = matrix.size();        // 행렬의 행의 수
    size_t numCols = matrix[0].size();       // 행렬의 열의 수
    Matrix result(numCols, std::vector<double>(numRows, 0.0));
    
    // 각 행과 열을 뒤바꾸어 저장합니다.
    for (size_t i = 0; i < numRows; i++) {
        if (matrix[i].size() != numCols) {
            throw std::invalid_argument("행마다 열의 개수가 일치하지 않습니다.");
        }
        for (size_t j = 0; j < numCols; j++) {
            // 원래 행렬의 (i, j) 원소를 전치 행렬의 (j, i) 위치에 저장
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

/**
 * @brief 입력 벡터에 대해 Softmax 함수를 계산합니다.
 *
 * Softmax 함수는 주어진 값들을 0과 1 사이의 확률로 변환하며,
 * 각 값이 전체 합에서 차지하는 비율을 나타냅니다.
 * 수학적으로 softmax는 아래와 같이 계산됩니다.
 *
 *     softmax(x_i) = exp(x_i) / (exp(x_1) + exp(x_2) + ... + exp(x_n))
 *
 * 수치적 안정성을 위해, 최대값을 빼주는 방식으로 계산합니다.
 *
 * @param inputVec 입력 벡터 (예: Attention score들의 한 행)
 * @return std::vector<double> Softmax가 적용된 확률 벡터.
 *
 * 시간 복잡도: O(n)
 * 메모리 복잡도: O(n)
 */
std::vector<double> softmax(const std::vector<double>& inputVec) {
    std::vector<double> outputVec(inputVec.size());
    
    // 입력 벡터에서 가장 큰 값을 찾습니다.
    // 이는 exp 계산 시 매우 큰 값으로 인한 오버플로우를 방지하기 위함입니다.
    double maxElem = *std::max_element(inputVec.begin(), inputVec.end());
    
    double sumExp = 0.0;
    // 모든 원소에 대해 exp(x - maxElem)을 계산하고 그 합을 구합니다.
    for (double x : inputVec) {
        sumExp += std::exp(x - maxElem);
    }
    
    // 각 원소에 대해 softmax 값을 계산합니다.
    // exp(x_i - maxElem) / (전체 exp의 합)
    for (size_t i = 0; i < inputVec.size(); i++) {
        outputVec[i] = std::exp(inputVec[i] - maxElem) / sumExp;
    }
    return outputVec;
}

/**
 * @brief 행렬의 각 행(row)에 대해 Softmax 함수를 적용합니다.
 *
 * @param matrix 입력 행렬 (각 행은 독립적으로 softmax 적용 대상)
 * @return Matrix 각 행에 softmax가 적용된 행렬.
 *
 * 시간 복잡도: O(n * m)  (n: 행의 수, m: 각 행의 열의 수)
 * 메모리 복잡도: O(n * m)
 */
Matrix softmaxRowWise(const Matrix& matrix) {
    Matrix result = matrix; // 원본을 복사합니다.
    // 각 행마다 softmax 함수를 적용합니다.
    for (auto &row : result) {
        row = softmax(row);
    }
    return result;
}

/**
 * @brief Scaled Dot-Product Attention을 계산하는 클래스입니다.
 *
 * Attention 메커니즘의 수학적 개념:
 * - **Query (Q)**: 주어진 입력에서 관심있는 정보를 뽑아내기 위한 '질문'
 * - **Key (K)**: 입력의 각 부분에 대한 '답' 역할을 하는 벡터
 * - **Value (V)**: 실제로 가져올 정보(답변)
 *
 * 계산 과정:
 * 1. **Dot-Product**: Query와 Key의 전치행렬을 곱하여 각 Query가 각 Key와 얼마나 연관되어 있는지(유사도)를 계산합니다.
 *    - 수학적으로: Attention Score(i, j) = Q(i) · K(j)
 * 2. **Scaling**: 유사도 값이 너무 커지는 것을 방지하기 위해 \( \frac{1}{\sqrt{d_k}} \)로 스케일링합니다.
 *    - \( d_k \)는 Key 벡터의 차원입니다.
 * 3. **Softmax**: 스케일된 점수에 Softmax 함수를 적용해 확률 값(가중치)로 변환합니다.
 * 4. **Weighted Sum**: 이 확률 값을 Value 행렬에 곱하여 최종 Attention 결과를 얻습니다.
 */
class ScaledDotProductAttention {
public:
    /**
     * @brief Query, Key, Value 행렬을 받아서 Attention 결과를 계산합니다.
     *
     * @param query Query 행렬 (각 행은 하나의 Query, 크기는 numQueries x d_k)
     * @param key Key 행렬 (각 행은 하나의 Key, 크기는 numKeys x d_k)
     * @param value Value 행렬 (각 행은 하나의 Value, 크기는 numKeys x d_v)
     * @return Matrix 최종 Attention 결과 행렬 (크기는 numQueries x d_v)
     * @throws std::invalid_argument 만약 행렬의 차원이 올바르지 않으면 예외 발생.
     */
    Matrix computeAttention(const Matrix &query, const Matrix &key, const Matrix &value) const {
        // 입력 행렬들이 비어있지 않은지 확인합니다.
        if (query.empty() || key.empty() || value.empty()) {
            throw std::invalid_argument("입력 행렬은 비어 있을 수 없습니다.");
        }
        
        size_t numQueries = query.size();    // Query 행렬의 행의 수
        size_t d_k = query[0].size();          // 각 Query의 차원 (벡터의 길이)
        size_t numKeys = key.size();           // Key 행렬의 행의 수
        
        // Query 행렬의 각 행이 모두 같은 크기인지 확인합니다.
        for (const auto &row : query) {
            if (row.size() != d_k) {
                throw std::invalid_argument("Query 행렬의 행마다 크기가 다릅니다.");
            }
        }
        // Key 행렬의 각 행의 크기가 Query와 동일한지 확인합니다.
        for (const auto &row : key) {
            if (row.size() != d_k) {
                throw std::invalid_argument("Key 행렬의 각 벡터의 차원이 Query와 다릅니다.");
            }
        }
        
        // Value 행렬은 각 행이 d_v 차원을 가지며, Key 행렬의 행 수와 동일해야 합니다.
        size_t d_v = value[0].size();          // Value 벡터의 차원
        for (const auto &row : value) {
            if (row.size() != d_v) {
                throw std::invalid_argument("Value 행렬의 행마다 크기가 다릅니다.");
            }
        }
        if (numKeys != value.size()) {
            throw std::invalid_argument("Key와 Value 행렬의 행의 수가 일치하지 않습니다.");
        }
        
        // ====================== 단계 1: Query와 Key의 Dot-Product ======================
        // Key 행렬의 전치(transpose)를 구합니다.
        // 전치 후의 행렬 크기는 (d_k x numKeys)가 됩니다.
        Matrix keyTransposed = transpose(key);
        
        // Query 행렬 (numQueries x d_k)와 keyTransposed (d_k x numKeys)를 곱합니다.
        // 결과 행렬 attentionScores의 크기는 (numQueries x numKeys)입니다.
        // attentionScores(i, j)는 i번째 Query와 j번째 Key의 유사도를 나타냅니다.
        Matrix attentionScores = matMul(query, keyTransposed);
        
        // ====================== 단계 2: Scaling ======================
        // d_k는 Key 벡터의 차원입니다.
        // 스케일링 팩터(scaleFactor)는 1 / sqrt(d_k)입니다.
        // 이는 내적 값이 너무 커지는 것을 방지하여 softmax 계산의 안정성을 높입니다.
        double scaleFactor = 1.0 / std::sqrt(static_cast<double>(d_k));
        for (auto &row : attentionScores) {
            for (auto &score : row) {
                // 각 유사도(score)를 scaleFactor로 곱합니다.
                score *= scaleFactor;
            }
        }
        
        // ====================== 단계 3: Softmax 적용 ======================
        // 각 Query에 대해, 즉 attentionScores의 각 행에 대해 softmax 함수를 적용합니다.
        // softmax를 통해 각 Key와의 연관성을 확률(0~1)로 변환합니다.
        Matrix attentionWeights = softmaxRowWise(attentionScores);
        
        // ====================== 단계 4: Weighted Sum (Attention 결과 계산) ======================
        // Attention 결과는 attentionWeights (numQueries x numKeys)와 Value 행렬 (numKeys x d_v)의 행렬 곱으로 계산됩니다.
        // 결과 행렬의 크기는 (numQueries x d_v)가 됩니다.
        Matrix output = matMul(attentionWeights, value);
        return attentionWeights;
    }
};

int main() {
    try {
        // ====================== 예제 입력 데이터 ======================
        // 예를 들어:
        // - d_k (Query와 Key의 벡터 차원)는 4,
        // - d_v (Value의 벡터 차원)는 3,
        // - numQueries (질문 개수)는 2,
        // - numKeys (답변 후보 개수)는 3입니다.
        
        // Query 행렬: 각 행은 하나의 Query 벡터
        Matrix query = {
            {1.0, 0.0, 1.0, 0.0},  // 첫 번째 Query: [1, 0, 1, 0]
            {0.0, 1.0, 0.0, 1.0}   // 두 번째 Query: [0, 1, 0, 1]
        };
        
        // Key 행렬: 각 행은 하나의 Key 벡터 (Query와 같은 차원)
        Matrix key = {
            {1.0, 0.0, 1.0, 0.0},  // 첫 번째 Key
            {0.0, 1.0, 0.0, 1.0},  // 두 번째 Key
            {1.0, 1.0, 0.0, 0.0}   // 세 번째 Key
        };
        
        // Value 행렬: 각 행은 하나의 Value 벡터 (각 Key에 해당하는 Value, 차원 d_v = 3)
        Matrix value = {
            {1.0, 2.0, 3.0},       // 첫 번째 Value
            {4.0, 5.0, 6.0},       // 두 번째 Value
            {7.0, 8.0, 9.0}        // 세 번째 Value
        };
        
        // ====================== Scaled Dot-Product Attention 계산 ======================
        // ScaledDotProductAttention 객체를 생성하고 computeAttention 함수를 호출합니다.
        ScaledDotProductAttention attention;
        Matrix attentionOutput = attention.computeAttention(query, key, value);
        
        // ====================== 결과 출력 ======================
        std::cout << "Attention Output (최종 결과 행렬):\n";
        // attentionOutput 행렬의 각 행을 출력합니다.
        for (const auto &row : attentionOutput) {
            for (double element : row) {
                std::cout << element << " ";
            }
            std::cout << "\n";
        }
    } catch (const std::exception &ex) {
        // 예외 발생 시 에러 메시지 출력
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
