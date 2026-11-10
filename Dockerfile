FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
RUN pip install --no-cache-dir '.[vision,server]'
EXPOSE 8000
CMD ["uvicorn", "visioninspect.api:app", "--host", "0.0.0.0", "--port", "8000"]
