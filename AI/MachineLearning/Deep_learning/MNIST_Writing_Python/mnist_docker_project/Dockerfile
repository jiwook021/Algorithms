FROM python:3.10-slim

WORKDIR /app

# 필수 패키지 설치 (빌드 의존성)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 환경변수 설정 - CUDA 비활성화 (CPU 모드로 실행)
ENV CUDA_VISIBLE_DEVICES=""

# 이미지 처리 모드 (train 또는 inference)
ENV MODE="inference"
ENV IMAGE_PATH="5.png"

# 진입점 스크립트 설정
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
