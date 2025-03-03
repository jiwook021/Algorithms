#include <opencv2/opencv.hpp>  // OpenCV 라이브러리 포함 (이미지 처리 및 컴퓨터 비전 기능 제공)
#include <vector>             // std::vector 사용을 위해 포함 (동적 배열)
#include <iostream>           // std::cout, std::cerr 등의 입출력 기능 사용을 위해 포함

// Harris 코너 검출 함수: 이미지에서 코너(특징점)를 검출
// - image: 입력 이미지 (그레이스케일)
// - corners: 검출된 특징점의 좌표를 저장할 벡터
// - blockSize: 코너 검출에 사용될 이웃 픽셀의 크기 (기본값: 2)
// - ksize: Sobel 연산자의 크기 (기본값: 3)
// - k: Harris 코너 검출 공식에서 사용하는 자유 파라미터 (기본값: 0.04)
void detectHarrisCorners(const cv::Mat& image, std::vector<cv::Point>& corners, int blockSize = 2, int ksize = 3, double k = 0.04) {
    // Harris 코너 응답을 저장할 행렬 생성 (이미지와 동일한 크기, 32비트 실수형)
    cv::Mat dst = cv::Mat::zeros(image.size(), CV_32FC1);
    
    // Harris 코너 검출 수행
    // - image: 입력 이미지 (그레이스케일)
    // - dst: 코너 응답이 저장될 행렬
    // - blockSize: 코너 검출에 사용될 이웃 픽셀의 크기
    // - ksize: Sobel 연산자의 크기
    // - k: Harris 코너 검출 공식에서 사용하는 자유 파라미터
    cv::cornerHarris(image, dst, blockSize, ksize, k);

    // dst 행렬의 최소값과 최대값을 구함
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    // minMaxLoc 함수를 사용해 dst의 최소값, 최대값, 그리고 그 위치를 찾음
    // - dst: 입력 행렬
    // - &minVal: 최소값을 저장할 변수의 포인터
    // - &maxVal: 최대값을 저장할 변수의 포인터
    // - &minLoc: 최소값의 위치를 저장할 cv::Point 객체의 포인터
    // - &maxLoc: 최대값의 위치를 저장할 cv::Point 객체의 포인터
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    // 특징점으로 선택할 임계값 계산 (최대 응답값의 1%)
    double threshold = 0.01 * maxVal;

    // dst 행렬을 순회하며 응답값이 임계값보다 큰 지점을 특징점으로 선택
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            if (dst.at<float>(y, x) > threshold) {
                // 응답값이 임계값보다 크면 해당 좌표(x, y)를 특징점으로 저장
                corners.push_back(cv::Point(x, y));
            }
        }
    }
}

// 간단한 설명자 추출 함수: 특징점 주변의 픽셀 패치를 설명자로 사용
// - image: 입력 이미지 (그레이스케일)
// - corners: 검출된 특징점의 좌표를 저장한 벡터
// - descriptors: 각 특징점에 대한 설명자를 저장할 행렬
// - patchSize: 설명자로 사용할 픽셀 패치의 크기 (기본값: 5, 즉 5x5 패치)
void computeSimpleDescriptors(const cv::Mat& image, const std::vector<cv::Point>& corners, cv::Mat& descriptors, int patchSize = 5) {
    int halfPatch = patchSize / 2;  // 패치의 절반 크기 (예: 5x5 패치의 경우 2)
    
    // 설명자를 저장할 행렬 초기화
    // - 행 수: 특징점의 개수 (corners.size())
    // - 열 수: 패치 크기 (patchSize * patchSize, 예: 25)
    // - 데이터 타입: 32비트 실수형 (CV_32FC1)
    descriptors = cv::Mat::zeros(static_cast<int>(corners.size()), patchSize * patchSize, CV_32FC1);

    // 각 특징점에 대해 설명자 추출
    for (size_t i = 0; i < corners.size(); i++) {
        int x = corners[i].x;  // 특징점의 x 좌표
        int y = corners[i].y;  // 특징점의 y 좌표
        int idx = 0;  // 설명자 벡터의 인덱스
        
        // 특징점 주변의 패치 영역을 순회 (예: -2 ~ 2)
        for (int dy = -halfPatch; dy <= halfPatch; dy++) {
            for (int dx = -halfPatch; dx <= halfPatch; dx++) {
                int px = x + dx;  // 패치 내 픽셀의 x 좌표
                int py = y + dy;  // 패치 내 픽셀의 y 좌표
                
                // 픽셀이 이미지 경계 내에 있는지 확인
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
                    // 이미지의 픽셀 값을 실수형으로 변환하여 설명자에 저장
                    descriptors.at<float>(i, idx) = static_cast<float>(image.at<uchar>(py, px));
                } else {
                    // 경계 밖의 픽셀은 0으로 설정
                    descriptors.at<float>(i, idx) = 0.0f;
                }
                idx++;  // 다음 설명자 요소로 이동
            }
        }
    }
}

int main() {
    // 1. 이미지 로드 (그레이스케일)
    // - "1.png" 파일을 그레이스케일로 읽어옴
    cv::Mat image = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        // 이미지를 로드하지 못한 경우 에러 메시지 출력 후 종료
        std::cerr << "Error: Could not load image." << std::endl;
        return 1;
    }

    // 2. 특징점 검출 (Harris 코너 검출)
    std::vector<cv::Point> corners;  // 특징점 좌표를 저장할 벡터
    detectHarrisCorners(image, corners);  // Harris 코너 검출 함수 호출

    // 3. 설명자 추출 (5x5 픽셀 패치)
    cv::Mat descriptors;  // 설명자를 저장할 행렬
    computeSimpleDescriptors(image, corners, descriptors);  // 설명자 추출 함수 호출

    // 4. 결과 출력 (특징점 수와 설명자 크기)
    // - 검출된 특징점의 개수 출력
    std::cout << "Number of corners detected: " << corners.size() << std::endl;
    // - 각 특징점의 설명자 크기 출력 (5x5 패치가 1차원으로 펼쳐져 25개의 실수값)
    std::cout << "Descriptor size per corner: " << descriptors.cols << " (5x5 patch flattened)" << std::endl;

    // 5. 특징점 시각화
    cv::Mat output = image.clone();  // 원본 이미지를 복사하여 시각화용 이미지 생성
    for (const auto& corner : corners) {
        // 각 특징점에 반지름 5, 흰색(255), 두께 2의 원을 그림
        cv::circle(output, corner, 5, cv::Scalar(255), 2);
    }
    // 시각화된 이미지를 "Detected Corners"라는 창에 표시
    cv::imshow("Detected Corners", output);
    // 사용자가 키를 누를 때까지 대기
    cv::waitKey(0);

    return 0;  // 프로그램 정상 종료
}