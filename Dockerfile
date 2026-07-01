# Google Cloud Run 배포용 컨테이너 (강의 슬라이드 12)
FROM python:3.10-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Cloud Run은 $PORT(기본 8080)로 트래픽을 보냅니다.
ENV PORT=8080
EXPOSE 8080

# 컨테이너 시작 시 Streamlit 서버 실행
CMD python -m streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
