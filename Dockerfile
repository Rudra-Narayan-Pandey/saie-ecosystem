FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Ensure /app is on the Python path so all modules resolve correctly
ENV PYTHONPATH="/app"

# Default environment variables
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV TASK="easy"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
