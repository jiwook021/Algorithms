# 사용할 컴파일러
CXX = g++

# 생성할 실행 파일 이름
TARGET = feature_detector

# 소스 파일
SRC = main.cpp

# pkg-config를 사용한 OpenCV 설정
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# 컴파일러 플래그
CXXFLAGS = -Wall -O2

# 링커 플래그 (필요한 경우, 일반적으로 pkg-config로 처리됨)
# LDFLAGS =

# 링크할 라이브러리
LDLIBS = $(OPENCV_FLAGS)

# 기본 타겟
all: $(TARGET)

# 타겟을 빌드하는 규칙
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDLIBS)

# 빌드 결과물을 제거하는 clean 타겟
clean:
	rm -f $(TARGET)