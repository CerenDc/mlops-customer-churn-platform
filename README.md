# MLOps Customer Churn Platform

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-V3-informational.svg)](#project-status)

A portfolio project that incrementally builds a production-style, end-to-end
MLOps platform for predicting customer churn.

## Business problem

Customer churn reduces recurring revenue and increases the cost of replacing
departed customers. A reliable churn signal can help retention teams identify
at-risk customers early, prioritize outreach, and measure the impact of
interventions.

## Project objective

Build a maintainable platform that ingests customer data, creates reproducible
features, trains and evaluates churn models, deploys predictions, and monitors
data and model quality. Each capability will be introduced in a small,
testable version so the operational decisions remain understandable.

## Planned architecture

```text
Data sources
    |
    v
Ingestion -> Raw data -> Processing / feature engineering
                              |
                              v
                    Model training and evaluation
                              |
                              v
                    Model registry and serving
                              |
                              v
                    Monitoring and retraining
```

V3 implements the ingestion, raw, Spark processing, and Delta storage portion
of this architecture. Modeling, serving, and orchestration remain future work.

## Technology roadmap

| Phase | Planned focus | Candidate technologies |
| --- | --- | --- |
| V1 | Python project foundation, linting, and tests | Python 3.13, pytest, Ruff |
| V1.1 | Developer automation and continuous integration | GitHub Actions, VS Code tasks |
| V2 | Reproducible data ingestion and validation | pandas, HTTPX |
| V3 (current) | Typed local processing and transactional storage | Apache Spark, Delta Lake |
| V4 | Model training, evaluation, and experiment tracking | scikit-learn, MLflow |
| V5 | Workflow orchestration and transformation management | Airflow, dbt |
| V6 | Packaging, serving, CI/CD, and observability | Docker, API framework, monitoring stack |

The roadmap is directional. Technology choices will be evaluated when their
phase begins rather than installed in advance.

## Local setup

Prerequisites: Python 3.13, Java 17, and Git.

On Apple Silicon macOS with Homebrew, install and expose Java 17 with:

```bash
brew install openjdk@17
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

Linux and CI environments may provide Java 17 through their normal JDK
distribution. Confirm the active runtime with `java -version`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Run the quality checks:

```bash
pytest -q
ruff check .
ruff format --check .
```

The `.env` file is ignored by Git. Do not store secrets in `.env.example`.

The same checks can be launched in VS Code from **Tasks: Run Task** by choosing
**Quality Check**.

## Continuous integration

GitHub Actions automatically validates Ruff linting, Ruff formatting, and the
pytest suite for every push to `main` and every pull request targeting `main`.
The workflow uses Python 3.13, Temurin Java 17, and cached Python dependencies.

## V2 data ingestion

V2 downloads the public IBM Telco Customer Churn CSV, validates its schema and
key business constraints, and atomically saves the accepted file. Run it from
the repository root after completing the local setup:

```bash
python -m churn_platform.ingestion.telco
```

The downloaded file is written to `data/raw/telco_customer_churn.csv`. Raw data
is reproducible from the public source and can be large, so it is deliberately
ignored by Git; only the `.gitkeep` placeholder is versioned.

No cleaning, feature engineering, or model work occurs during ingestion.

## V3 — Spark + Delta Lake

Spark provides a distributed-style DataFrame processing model locally and
prepares the project for processing patterns used on larger data platforms. V3
uses an explicit schema, normalizes field types, safely parses `TotalCharges`,
validates business constraints, and adds minimal lineage metadata.

Delta Lake stores the processed data in data lake files with an ACID
transaction log. This improves schema reliability and establishes a foundation
for table versioning while remaining compatible with Spark. The same format is
useful for a later Databricks phase, but Databricks is not implemented here.

Run both independent pipeline stages from the repository root:

```bash
python -m churn_platform.ingestion.telco
python -m churn_platform.spark.transform_telco
```

The processing command reads the immutable raw CSV and writes a genuine Delta
table to `data/processed/telco_customer_churn_delta/`. Both generated datasets
are ignored by Git.

### Data layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| RAW | `data/raw/` | Original source data; treated as immutable |
| PROCESSED | `data/processed/` | Typed, minimally cleaned data stored as Delta Lake |
| FEATURES | `data/features/` | Reserved for later ML feature engineering |

## Repository layout

```text
src/churn_platform/   Installable ingestion and Spark processing package
tests/                Automated tests
data/                 Raw, processed, and feature data zones
ml/                   Future modeling implementation
notebooks/            Exploratory notebooks
docs/                 Project documentation
.github/workflows/    GitHub Actions continuous integration
.vscode/              Shared editor configuration
```

## Project status

**V3 — Spark + Delta Lake.** The platform processes the immutable IBM Telco raw
CSV through a typed, validated Spark transformation and publishes a real Delta
Lake dataset with technical lineage metadata. dbt, Airflow, MLflow, Docker,
Databricks, feature engineering, model training, and serving are intentionally
not implemented in this version.
