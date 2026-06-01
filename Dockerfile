# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY src/ ./src/
COPY data/ ./data/

# Запускаем сервис
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]