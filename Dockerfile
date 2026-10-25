FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY amharic_tts ./amharic_tts
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[omnivoice,mms,server]"
ENV AMHARIC_TTS_HOME=/data HF_HOME=/data/hf OMP_NUM_THREADS=4
VOLUME ["/data"]
EXPOSE 8080
CMD ["amharic-tts", "serve", "--host", "0.0.0.0", "--port", "8080"]
