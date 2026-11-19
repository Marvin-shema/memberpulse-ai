FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memberpulse ./memberpulse
# Build demo data and train the churn model at image build time.
RUN python -m memberpulse seed && python -m memberpulse train

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').getenv('PORT', '8000') + '/health')" || exit 1
CMD ["sh", "-c", "uvicorn memberpulse.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
