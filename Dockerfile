FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Linux libraries required by MediaPipe / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project
COPY . .

# Collect Django static files
RUN python Web/manage.py collectstatic --noinput
# data base ke leye
RUN python Web/manage.py migrate --noinput --skip-checks

# Render uses PORT environment variable
EXPOSE 10000

# Run migrations and start Django
CMD ["sh", "-c", "python -m gunicorn --chdir Web config.asgi:application -k uvicorn.workers.UvicornWorker --timeout 180 --bind 0.0.0.0:${PORT:-10000}"]