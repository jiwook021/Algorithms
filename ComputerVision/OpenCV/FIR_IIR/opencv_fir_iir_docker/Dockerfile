FROM ubuntu:22.04

# 필수 패키지 설치 및 설정
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    g++ \
    make \
    libopencv-dev \
    python3-opencv \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 소스 파일 복사
COPY main.cpp .
COPY Makefile .

# X11 GUI 환경 설정을 위한 라이브러리
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 소스 코드 컴파일
RUN make

# 컨테이너 실행 시 실행할 명령
CMD ["./main"]
