# MLOps Customer Churn Platform

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-V1-informational.svg)](#project-status)

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

The `ingestion/`, `spark/`, and `ml/` directories are boundaries for future
phases. V1 deliberately contains no pipeline, distributed processing, model
training, serving, or infrastructure implementation.

## Technology roadmap

| Phase | Planned focus | Candidate technologies |
| --- | --- | --- |
| V1 (current) | Python project foundation, linting, and tests | Python 3.13, pytest, Ruff |
| V2 | Data ingestion and validation | Python, Pandera or Great Expectations |
| V3 | Scalable transformation and feature engineering | Apache Spark |
| V4 | Model training, evaluation, and experiment tracking | scikit-learn, MLflow |
| V5 | Workflow orchestration and transformation management | Airflow, dbt |
| V6 | Packaging, serving, CI/CD, and observability | Docker, API framework, monitoring stack |

The roadmap is directional. Technology choices will be evaluated when their
phase begins rather than installed in advance.

## Local setup

Prerequisites: Python 3.13 and Git.

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
```

The `.env` file is ignored by Git. Do not store secrets in `.env.example`.

## Repository layout

```text
src/churn_platform/   Minimal installable Python package
tests/                Automated tests
data/                 Raw, processed, and feature data zones
ingestion/            Future ingestion implementation
spark/                Future distributed processing implementation
ml/                   Future modeling implementation
notebooks/            Exploratory notebooks
docs/                 Project documentation
.github/workflows/    Future CI workflows
.vscode/              Shared editor configuration
```

## Project status

**V1 — project foundation.** The Python 3.13 package structure, pytest suite,
Ruff configuration, environment template, and editor recommendations are in
place. Airflow, Spark, dbt, MLflow, and Docker are intentionally not installed
or configured in this version.
