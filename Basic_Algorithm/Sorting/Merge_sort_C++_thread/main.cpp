#include <iostream>
#include <vector>
#include <functional>
#include <future>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <type_traits>
#include <iterator>

/**
 * @brief 뮤텍스 없는 병렬 병합 정렬 알고리즘 구현
 * 
 * @tparam RandomIt 임의 접근 반복자 타입
 * @tparam Compare 비교 함수 타입
 * @param first 정렬할 범위의 시작 반복자
 * @param last 정렬할 범위의 끝 반복자 (이 위치는 포함되지 않음)
 * @param comp 두 요소를 비교하는 함수 객체
 * @param depth 현재 재귀 깊이 (병렬화 제어용)
 * 
 * @throws std::invalid_argument 유효하지 않은 반복자 범위가 제공될 경우
 * 
 * @note 시간 복잡도: O(n log n), 병렬 처리로 상수 계수 개선
 * @note 공간 복잡도: O(n)
 */
template<typename RandomIt, typename Compare>
void mysort(RandomIt first, RandomIt last, Compare comp, int depth = 0) {
    // C++17 컴파일러를 위한 임의 접근 반복자 검사
    static_assert(std::is_same_v
        typename std::iterator_traits<RandomIt>::iterator_category,
        std::random_access_iterator_tag
    >, "RandomIt must be a random access iterator");
    
    // 입력 검증
    if (first > last) {
        throw std::invalid_argument("유효하지 않은 반복자 범위: first는 last보다 작거나 같아야 합니다");
    }
    
    // 기본 케이스: 요소가 0개 또는 1개인 경우 이미 정렬됨
    auto distance = std::distance(first, last);
    if (distance <= 1) {
        return;
    }
    
    // 범위를 두 부분으로 나눔
    auto middle = first + distance / 2;
    
    // 병렬화 결정 조건:
    // 1. 배열이 충분히 큼 (PARALLEL_THRESHOLD 이상)
    // 2. 재귀 깊이가 MAX_DEPTH 미만 (스레드 과다 생성 방지)
    constexpr size_t PARALLEL_THRESHOLD = 10000;
    constexpr int MAX_DEPTH = 3; // 최대 2^3=8개 스레드로 제한
    
    // 하드웨어 동시성 수준 확인 (가용 코어 수)
    static const int hardware_threads = std::thread::hardware_concurrency();
    
    // 병렬 처리 결정
    if (distance > PARALLEL_THRESHOLD && depth < MAX_DEPTH && 
        hardware_threads > 1) {
        
        // 왼쪽 부분을 비동기로 처리 (값 캡처로 안전하게 전달)
        std::future<void> future = std::async(
            std::launch::async,
            [=, first_copy = first, middle_copy = middle]() {
                mysort(first_copy, middle_copy, comp, depth + 1);
            }
        );
        
        // 현재 스레드에서 오른쪽 부분 처리
        mysort(middle, last, comp, depth + 1);
        
        // 왼쪽 부분 완료 대기
        future.wait();
    } else {
        // 순차 처리
        mysort(first, middle, comp, MAX_DEPTH);
        mysort(middle, last, comp, MAX_DEPTH);
    }
    
    // 정렬된 두 부분 배열 병합
    std::inplace_merge(first, middle, last, comp);
}

/**
 * @brief 기본 비교 연산자를 사용하는 간편 버전
 */
template<typename RandomIt>
void mysort(RandomIt first, RandomIt last) {
    mysort(first, last, std::less<typename std::iterator_traits<RandomIt>::value_type>(), 0);
}

/**
 * @brief 벡터 내용을 출력하는 유틸리티 함수
 */
template<typename T>
void printVector(const std::vector<T>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

// Person 클래스 정의를 전역 범위로 이동
struct Person {
    std::string name;
    int age;
};

// Person 클래스를 위한 전역 연산자 오버로딩
std::ostream& operator<<(std::ostream& os, const Person& p) {
    os << "(" << p.name << ", " << p.age << ")";
    return os;
}

// 테스트 케이스
int main() {
    std::cout << "사용 가능한 하드웨어 스레드: " 
              << std::thread::hardware_concurrency() << std::endl;
    
    // 테스트 케이스 1: 정수 벡터 오름차순 정렬
    {
        std::vector<int> numbers = {9, 3, 7, 1, 5, 8, 2, 4, 6};
        printVector(numbers, "정렬 전");
        
        mysort(numbers.begin(), numbers.end());
        printVector(numbers, "오름차순 정렬 후");
    }
    
    // 테스트 케이스 2: 정수 벡터 내림차순 정렬
    {
        std::vector<int> numbers = {9, 3, 7, 1, 5, 8, 2, 4, 6};
        printVector(numbers, "정렬 전");
        
        mysort(numbers.begin(), numbers.end(), [](int a, int b) { return a > b; });
        printVector(numbers, "내림차순 정렬 후");
    }
    
    // 테스트 케이스 3: 사용자 정의 구조체 정렬
    {
        std::vector<Person> people = {
            {"Alice", 30}, {"Bob", 25}, {"Charlie", 35}, {"David", 20}
        };
        
        printVector(people, "정렬 전");
        
        // 나이 기준 정렬
        mysort(people.begin(), people.end(), 
               [](const Person& a, const Person& b) { return a.age < b.age; });
        
        printVector(people, "나이순 정렬 후");
    }
    
    // 테스트 케이스 4-7: 엣지 케이스
    {
        // 빈 벡터
        std::vector<int> empty;
        mysort(empty.begin(), empty.end());
        printVector(empty, "빈 벡터 정렬");
        
        // 단일 요소 벡터
        std::vector<int> single = {42};
        mysort(single.begin(), single.end());
        printVector(single, "단일 요소 벡터 정렬");
        
        // 이미 정렬된 벡터
        std::vector<int> sorted = {1, 2, 3, 4, 5};
        mysort(sorted.begin(), sorted.end());
        printVector(sorted, "이미 정렬된 벡터");
        
        // 역순 정렬된 벡터
        std::vector<int> reversed = {5, 4, 3, 2, 1};
        mysort(reversed.begin(), reversed.end());
        printVector(reversed, "역순 벡터 정렬 후");
    }
    
    // 테스트 케이스 8: 대용량 데이터 병렬 정렬 성능 테스트
    {
        constexpr size_t LARGE_SIZE = 1000000;
        std::vector<int> large_vector(LARGE_SIZE);
        
        // 랜덤 데이터 생성
        for (size_t i = 0; i < LARGE_SIZE; ++i) {
            large_vector[i] = rand() % LARGE_SIZE;
        }
        
        std::cout << "대용량 데이터(" << LARGE_SIZE << "개) 정렬 시작..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        mysort(large_vector.begin(), large_vector.end());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "정렬 완료: " << elapsed.count() << "초 소요" << std::endl;
        
        // 정렬 검증
        bool is_sorted = std::is_sorted(large_vector.begin(), large_vector.end());
        std::cout << "정렬 확인: " << (is_sorted ? "성공" : "실패") << std::endl;
        
        // STL sort와 성능 비교
        std::vector<int> stl_vector = large_vector; // 복사
        // C++17에서는 random_shuffle이 사용 중단되었으므로 shuffle 사용
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(stl_vector.begin(), stl_vector.end(), g);
        
        start = std::chrono::high_resolution_clock::now();
        std::sort(stl_vector.begin(), stl_vector.end());
        end = std::chrono::high_resolution_clock::now();
        
        elapsed = end - start;
        std::cout << "STL sort 완료: " << elapsed.count() << "초 소요" << std::endl;
    }
    
    return 0;
}