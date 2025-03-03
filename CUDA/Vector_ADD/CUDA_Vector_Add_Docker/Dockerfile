FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 소스 파일 복사
COPY main.cu .
COPY Makefile .

# 컴파일
RUN make

# 실행 명령
CMD ["./main"]
