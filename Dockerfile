FROM python:3.11-slim

WORKDIR /app

# Install production Python dependencies
COPY server/requirements.txt server/requirements.txt
COPY requirements-prod.txt requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application source
COPY server/ server/
COPY static/ static/
COPY shared/ shared/
COPY common/ common/
COPY nodes.json .

# Ensure output directories exist
RUN mkdir -p /app/static/outputs /app/db

ENV PORT=8080
EXPOSE 8080

# Use gunicorn in production; falls back to flask dev server if gunicorn unavailable
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "server.app:app"]
