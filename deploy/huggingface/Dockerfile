# Hugging Face Space (Docker SDK) running the amharic-tts server + studio on CPU.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
ARG TTS_REF=claude/awesome-pascal-dmnvu2
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "amharic-tts[omnivoice,server] @ git+https://github.com/henokmengistu/tts@${TTS_REF}" gtts google-genai
USER user
ENV HOME=/home/user AMHARIC_TTS_HOME=/home/user/data HF_HOME=/home/user/hf OMP_NUM_THREADS=2 AMHARIC_TTS_PRELOAD=1
EXPOSE 7860
CMD ["amharic-tts", "serve", "--host", "0.0.0.0", "--port", "7860", "--steps", "8", "--preload"]
