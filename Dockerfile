FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get install -y ffmpeg zip \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 7860

CMD ["python3", "webui.py"]