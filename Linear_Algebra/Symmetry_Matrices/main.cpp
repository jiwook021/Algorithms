#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <cmath>
#include <numeric>

/**
 * @brief 대칭 행렬(Symmetry Matrix)의 특성을 보여주는 클래스
 * 
 * 대칭 행렬은 A[i][j] = A[j][i]를 만족하는 정방행렬입니다.
 * 이 클래스는 다음 성질들을 보여줍니다:
 * 1. 전치행렬과 동일함 (A = A^T)
 * 2. 고유값(eigenvalues)이 항상 실수
 * 3. 행렬식(determinant)이 고유값들의 곱과 같음
 * 4. 직교 대각화 가능
 * 5. 대칭 행렬의 곱은 일반적으로 대칭이 아님 (단, AB=BA일 경우 대칭)
 */
class SymmetryMatrix {
private:
    std::vector<std::vector<double>> matrix;
    size_t size;
    mutable std::mutex mtx; // mutable로 변경하여 const 메서드에서 사용 가능하게 함

public:
    /**
     * @brief 지정된 크기의 랜덤 대칭 행렬 생성
     * 
     * @param n 행렬의 크기 (n x n)
     * @return 생성된 대칭 행렬
     * 
     * 시간 복잡도: O(n²) - 모든 원소를 초기화해야 함
     * 공간 복잡도: O(n²) - n x n 행렬 저장
     */
    SymmetryMatrix(size_t n) : size(n) {
        if (n == 0) {
            throw std::invalid_argument("행렬 크기는 0보다 커야 합니다.");
        }
        
        // 행렬 크기 초기화
        matrix.resize(n, std::vector<double>(n, 0.0));
        
        // 랜덤 대칭 행렬 생성
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-10.0, 10.0);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                double value = dist(gen);
                matrix[i][j] = value;
                // 대칭성 보장: A[i][j] = A[j][i]
                if (i != j) {
                    matrix[j][i] = value;
                }
            }
        }
    }
    
    /**
     * @brief 행렬 복사 생성자 (스레드 안전)
     */
    SymmetryMatrix(const SymmetryMatrix& other) {
        std::lock_guard<std::mutex> lock(other.mtx);
        matrix = other.matrix;
        size = other.size;
    }
    
    /**
     * @brief 대입 연산자 (스레드 안전)
     */
    SymmetryMatrix& operator=(const SymmetryMatrix& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(mtx);
            std::lock_guard<std::mutex> lock2(other.mtx);
            matrix = other.matrix;
            size = other.size;
        }
        return *this;
    }

    /**
     * @brief 행렬 원소 설정 (단위행렬 생성 등을 위해)
     */
    void setValue(size_t i, size_t j, double value) {
        std::lock_guard<std::mutex> lock(mtx);
        if (i >= size || j >= size) {
            throw std::out_of_range("인덱스가 범위를 벗어났습니다.");
        }
        matrix[i][j] = value;
    }

    /**
     * @brief 행렬 출력
     */
    void print() const {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief 대칭 행렬인지 확인
     * 
     * @return 대칭이면 true, 아니면 false
     * 
     * 시간 복잡도: O(n²) - 모든 원소를 비교해야 함
     * 공간 복잡도: O(1) - 추가 공간 불필요
     */
    bool isSymmetric() const {
        std::lock_guard<std::mutex> lock(mtx);
        constexpr double EPSILON = 1e-10;
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = i + 1; j < size; ++j) {
                if (std::abs(matrix[i][j] - matrix[j][i]) > EPSILON) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief 전치행렬(Transpose) 계산
     * 
     * @return 전치행렬 (대칭행렬에서는 원행렬과 같음)
     * 
     * 시간 복잡도: O(n²) - 모든 원소를 복사해야 함
     * 공간 복잡도: O(n²) - n x n 행렬 저장
     */
    SymmetryMatrix transpose() const {
        std::lock_guard<std::mutex> lock(mtx);
        SymmetryMatrix result(size);
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                result.setValue(i, j, matrix[j][i]);
            }
        }
        
        return result;
    }

    /**
     * @brief 행렬 덧셈 (두 대칭행렬의 합은 항상 대칭)
     * 
     * @param other 더할 행렬
     * @return 두 행렬의 합
     * 
     * 시간 복잡도: O(n²) - 모든 원소를 더해야 함
     * 공간 복잡도: O(n²) - n x n 결과 행렬 저장
     */
    SymmetryMatrix add(const SymmetryMatrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("행렬 크기가 일치하지 않습니다.");
        }
        
        std::lock_guard<std::mutex> lock1(mtx);
        std::lock_guard<std::mutex> lock2(other.mtx);
        
        SymmetryMatrix result(size);
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                result.setValue(i, j, matrix[i][j] + other.matrix[i][j]);
            }
        }
        
        return result;
    }

    /**
     * @brief 행렬 곱셈 (두 대칭행렬의 곱은 일반적으로 대칭이 아님)
     * 
     * @param other 곱할 행렬
     * @return 두 행렬의 곱
     * 
     * 시간 복잡도: O(n³) - 행렬 곱셈의 표준 알고리즘
     * 공간 복잡도: O(n²) - n x n 결과 행렬 저장
     */
    std::vector<std::vector<double>> multiply(const SymmetryMatrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("행렬 크기가 일치하지 않습니다.");
        }
        
        std::lock_guard<std::mutex> lock1(mtx);
        std::lock_guard<std::mutex> lock2(other.mtx);
        
        std::vector<std::vector<double>> result(size, std::vector<double>(size, 0.0));
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                for (size_t k = 0; k < size; ++k) {
                    result[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        
        return result;
    }

    /**
     * @brief 대각합(Trace) 계산 - 대각선 원소들의 합
     * 
     * @return 대각합
     * 
     * 시간 복잡도: O(n) - 대각선 원소만 접근
     * 공간 복잡도: O(1) - 추가 공간 불필요
     */
    double trace() const {
        std::lock_guard<std::mutex> lock(mtx);
        double sum = 0.0;
        
        for (size_t i = 0; i < size; ++i) {
            sum += matrix[i][i];
        }
        
        return sum;
    }

    /**
     * @brief 2x2 또는 3x3 행렬의 행렬식(Determinant) 계산
     * 
     * @return 행렬식 값
     * 
     * 시간 복잡도: O(1) for 2x2, O(n) for 3x3
     * 공간 복잡도: O(1) - 추가 공간 불필요
     */
    double determinant() const {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (size == 1) {
            return matrix[0][0];
        } else if (size == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        } else if (size == 3) {
            double det = 0.0;
            det += matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]);
            det -= matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]);
            det += matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
            return det;
        } else {
            throw std::logic_error("4x4 이상 행렬의 행렬식은 구현되지 않았습니다.");
        }
    }

    /**
     * @brief 제곱합(Frobenius Norm) 계산 - 모든 원소의 제곱의 합의 제곱근
     * 
     * @return 제곱합
     * 
     * 시간 복잡도: O(n²) - 모든 원소를 제곱하고 더함
     * 공간 복잡도: O(1) - 추가 공간 불필요
     */
    double frobeniusNorm() const {
        std::lock_guard<std::mutex> lock(mtx);
        double sum = 0.0;
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                sum += matrix[i][j] * matrix[i][j];
            }
        }
        
        return std::sqrt(sum);
    }

    /**
     * @brief 멱급수 방법을 사용한 최대 고유값 근사 계산
     * (대칭행렬의 고유값은 항상 실수)
     * 
     * @param iterations 반복 횟수
     * @return 최대 고유값의 근사치
     * 
     * 시간 복잡도: O(n² * iterations)
     * 공간 복잡도: O(n) - 벡터 저장
     */
    double approximateLargestEigenvalue(size_t iterations = 100) const {
        std::lock_guard<std::mutex> lock(mtx);
        
        // 초기 단위 벡터 생성
        std::vector<double> v(size, 1.0);
        double norm = std::sqrt(std::accumulate(v.begin(), v.end(), 0.0, 
                                               [](double sum, double val) { return sum + val * val; }));
        
        for (auto& val : v) {
            val /= norm;
        }
        
        // 멱급수 방법 반복
        for (size_t iter = 0; iter < iterations; ++iter) {
            // v = A*v 계산
            std::vector<double> new_v(size, 0.0);
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    new_v[i] += matrix[i][j] * v[j];
                }
            }
            
            // 정규화
            norm = std::sqrt(std::accumulate(new_v.begin(), new_v.end(), 0.0, 
                                            [](double sum, double val) { return sum + val * val; }));
            
            for (auto& val : new_v) {
                val /= norm;
            }
            
            v = new_v;
        }
        
        // 레일리 몫 계산 (Rayleigh quotient)
        double numerator = 0.0;
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                numerator += v[i] * matrix[i][j] * v[j];
            }
        }
        
        double denominator = std::accumulate(v.begin(), v.end(), 0.0, 
                                           [](double sum, double val) { return sum + val * val; });
        
        return numerator / denominator;
    }

    /**
     * @brief 행렬이 양의 정부호(Positive definite) 인지 확인
     * (모든 고유값이 양수인 경우)
     * 
     * 단순 구현을 위해 2x2, 3x3 행렬에 대해서만 구현
     * 
     * @return 양의 정부호이면 true, 아니면 false
     * 
     * 시간 복잡도: O(1) for 2x2, O(n) for 3x3
     * 공간 복잡도: O(1) - 추가 공간 불필요
     */
    bool isPositiveDefinite() const {
        std::lock_guard<std::mutex> lock(mtx);
        
        // 모든 대각 원소가 양수여야 함 (필요 조건)
        for (size_t i = 0; i < size; ++i) {
            if (matrix[i][i] <= 0) {
                return false;
            }
        }
        
        // 실바의 기준(Sylvester's criterion) 사용
        if (size == 1) {
            return matrix[0][0] > 0;
        } else if (size == 2) {
            double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            return matrix[0][0] > 0 && det > 0;
        } else if (size == 3) {
            double det1 = matrix[0][0];
            double det2 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            double det3 = determinant();
            
            return det1 > 0 && det2 > 0 && det3 > 0;
        } else {
            throw std::logic_error("4x4 이상 행렬의 양의 정부호 판별은 구현되지 않았습니다.");
        }
    }

    /**
     * @brief 두 대칭 행렬이 교환 가능한지 확인 (AB = BA)
     * 
     * @param other 비교할 행렬
     * @return 교환 가능하면 true, 아니면 false
     * 
     * 시간 복잡도: O(n³) - 행렬 곱셈 2회
     * 공간 복잡도: O(n²) - 두 결과 행렬 저장
     */
    bool isCommutative(const SymmetryMatrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("행렬 크기가 일치하지 않습니다.");
        }
        
        auto AB = multiply(other);
        auto BA = other.multiply(*this);
        
        constexpr double EPSILON = 1e-10;
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (std::abs(AB[i][j] - BA[i][j]) > EPSILON) {
                    return false;
                }
            }
        }
        
        return true;
    }

    /**
     * @brief 곱의 대칭성 확인
     * (두 대칭 행렬의 곱은 일반적으로 대칭이 아님, 단 AB=BA일 경우는 대칭)
     * 
     * @param other 곱할 행렬
     * @return 곱이 대칭이면 true, 아니면 false
     * 
     * 시간 복잡도: O(n³) - 행렬 곱셈 + 대칭 확인
     * 공간 복잡도: O(n²) - 결과 행렬 저장
     */
    bool isProductSymmetric(const SymmetryMatrix& other) const {
        if (size != other.size) {
            throw std::invalid_argument("행렬 크기가 일치하지 않습니다.");
        }
        
        auto product = multiply(other);
        
        constexpr double EPSILON = 1e-10;
        
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = i + 1; j < size; ++j) {
                if (std::abs(product[i][j] - product[j][i]) > EPSILON) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /**
     * @brief 행렬 원소에 접근
     */
    double getValue(size_t i, size_t j) const {
        std::lock_guard<std::mutex> lock(mtx);
        if (i >= size || j >= size) {
            throw std::out_of_range("인덱스가 범위를 벗어났습니다.");
        }
        return matrix[i][j];
    }
    
    /**
     * @brief 행렬 크기 반환
     */
    size_t getSize() const {
        return size;
    }
};

/**
 * @brief 단위행렬 생성
 */
SymmetryMatrix createIdentityMatrix(size_t n) {
    SymmetryMatrix result(n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.setValue(i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    
    return result;
}

int main() {
    try {
        // 테스트할 행렬 크기
        constexpr size_t MATRIX_SIZE = 3;
        
        std::cout << "============ 대칭 행렬(Symmetry Matrix)의 성질 ============" << std::endl;
        
        // 1. 랜덤 대칭 행렬 생성
        std::cout << "\n1. 랜덤 대칭 행렬 A 생성:" << std::endl;
        SymmetryMatrix A(MATRIX_SIZE);
        A.print();
        
        // 2. 대칭 확인
        std::cout << "\n2. 행렬 A는 대칭인가? " << (A.isSymmetric() ? "예" : "아니오") << std::endl;
        
        // 3. 전치행렬 확인 (A^T = A)
        std::cout << "\n3. 전치행렬 A^T:" << std::endl;
        SymmetryMatrix AT = A.transpose();
        AT.print();
        
        // 4. 대각합(Trace) 계산
        std::cout << "\n4. 대각합(Trace): " << A.trace() << std::endl;
        
        // 5. 행렬식(Determinant) 계산
        std::cout << "\n5. 행렬식(Determinant): " << A.determinant() << std::endl;
        
        // 6. 프로베니우스 노름(Frobenius norm) 계산
        std::cout << "\n6. 프로베니우스 노름(Frobenius norm): " << A.frobeniusNorm() << std::endl;
        
        // 7. 최대 고유값 근사 계산
        std::cout << "\n7. 최대 고유값 근사값: " << A.approximateLargestEigenvalue() << std::endl;
        
        // 8. 양의 정부호(Positive definite) 확인
        std::cout << "\n8. 행렬 A는 양의 정부호인가? " << (A.isPositiveDefinite() ? "예" : "아니오") << std::endl;
        
        // 9. 두 번째 대칭 행렬 생성
        std::cout << "\n9. 두 번째 대칭 행렬 B 생성:" << std::endl;
        SymmetryMatrix B(MATRIX_SIZE);
        B.print();
        
        // 10. 합의 대칭성 확인 (A + B)
        std::cout << "\n10. 두 대칭 행렬의 합 (A + B):" << std::endl;
        SymmetryMatrix sum = A.add(B);
        sum.print();
        std::cout << "합이 대칭인가? " << (sum.isSymmetric() ? "예" : "아니오") << std::endl;
        
        // 11. 곱의 대칭성 확인 (AB)
        std::cout << "\n11. 두 대칭 행렬의 곱 (AB):" << std::endl;
        auto product = A.multiply(B);
        for (const auto& row : product) {
            for (const auto& val : row) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << val << " ";
            }
            std::cout << std::endl;
        }
        
        // 12. 곱의 대칭성 확인
        std::cout << "\n12. 곱 AB는 일반적으로 대칭인가? " << (A.isProductSymmetric(B) ? "예" : "아니오") << std::endl;
        
        // 13. 교환법칙 확인 (AB = BA?)
        std::cout << "\n13. AB = BA가 성립하는가? " << (A.isCommutative(B) ? "예" : "아니오") << std::endl;
        
        // 14. 두 행렬이 교환 가능할 때 곱의 대칭성 확인
        // 교환 가능한 행렬 생성(단위행렬과 A)
        SymmetryMatrix I = createIdentityMatrix(MATRIX_SIZE);
        
        std::cout << "\n14. 단위행렬 I:" << std::endl;
        I.print();
        
        std::cout << "AI = IA가 성립하는가? " << (A.isCommutative(I) ? "예" : "아니오") << std::endl;
        std::cout << "곱 AI는 대칭인가? " << (A.isProductSymmetric(I) ? "예" : "아니오") << std::endl;
        
        std::cout << "\n결론: 대칭 행렬 A, B에 대해" << std::endl;
        std::cout << "1. A + B는 항상 대칭" << std::endl;
        std::cout << "2. AB는 일반적으로 대칭이 아님" << std::endl;
        std::cout << "3. AB = BA일 경우에만 AB가 대칭" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "예외 발생: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}