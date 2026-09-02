FROM python:3.13-slim-bookworm

ARG MLOPS_UID=50000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_REPOSITORY_ROOT=/opt/mlops \
    HOME=/opt/mlops

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        curl \
        libgomp1 \
        openjdk-17-jre-headless \
    && java_home="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" \
    && ln -s "$java_home" /opt/java \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java \
    PATH="/opt/java/bin:${PATH}"

WORKDIR /opt/mlops

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[orchestration]"

COPY dbt_project ./dbt_project
COPY monitoring ./monitoring
COPY orchestration ./orchestration

RUN mkdir -p \
        /opt/mlops/data/airflow \
        /opt/mlops/data/features \
        /opt/mlops/data/mlflow/artifacts \
        /opt/mlops/data/processed \
        /opt/mlops/data/raw \
        /opt/mlops/.ivy2.5.2 \
    && useradd --uid "$MLOPS_UID" --gid 0 --home-dir /opt/mlops --no-create-home mlops \
    && chown -R "$MLOPS_UID":0 /opt/mlops \
    && chmod -R g=u /opt/mlops

USER mlops

CMD ["python", "-m", "churn_platform.health"]
