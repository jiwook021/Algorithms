#include <iostream>
#include <vector>
#include <algorithm>

int maxSumSubarray(const std::vector<int>& arr, int k) {
    // 배열의 크기가 k보다 작은 경우 예외 처리
    if (arr.size() < k) {
        return -1; // 또는 다른 적절한 에러 값
    }
    
    // 첫 번째 윈도우의 합 계산
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    int maxSum = windowSum; // 최대 합 초기화
    
    // 슬라이딩 윈도우 이동
    for (int i = k; i < arr.size(); i++) {
        // 새 요소 추가하고 첫 번째 요소 제거
        windowSum = windowSum + arr[i] - arr[i - k];
        maxSum = std::max(maxSum, windowSum);
    }
    
    return maxSum;
}

int main() {
    std::vector<int> arr = {1, 4, 2, 10, 2, 3, 1, 0, 20};
    int k = 4;
    std::cout << "크기가 " << k << "인 하위 배열의 최대 합: " 
              << maxSumSubarray(arr, k) << std::endl;
    return 0;
}