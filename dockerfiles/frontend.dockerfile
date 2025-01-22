# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements_frontend.txt requirements_frontend.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements_frontend.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

WORKDIR /src/oracle_mnist

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
