FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app

# Expose port (informational)
ENV PORT=8001
# (optional) allow overriding CORS from env later
ENV FRONTEND_ORIGINS="*"

# Use uvicorn to serve FastAPI (respect PORT env var Cloud Run provides)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
