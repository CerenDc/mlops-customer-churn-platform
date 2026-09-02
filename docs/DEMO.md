# Reproducible portfolio demo

This guide is designed for a local interview demonstration. Start with
synthetic data; it exercises the real pipeline without downloading a dataset.

## Prerequisites

- Python 3.13, Java 17, and Git
- Docker Engine/Desktop with Compose v2
- For Kubernetes: `kubectl` and `kind`
- For IaC validation: Terraform 1.13+

## Installation and tests

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,orchestration]"
pytest -q
ruff check .
ruff format --check .
```

## Start the local platform

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose ps
```

Local demo URLs:

- Airflow: <http://localhost:8080>
- MLflow: <http://localhost:5000>
- Prediction API/OpenAPI: <http://localhost:8001/docs>
- Serving metrics: <http://localhost:8001/metrics>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (`admin` / `grafana-dev-only` in the example)

Airflow generates its simple-auth development password. Retrieve it with:

```bash
docker compose exec airflow-api-server \
  cat /opt/mlops/data/airflow/simple_auth_manager_passwords.json.generated
```

## Airflow demo

Trigger `mlops_customer_churn_pipeline` in the UI, or run:

```bash
docker compose exec airflow-scheduler \
  airflow dags trigger mlops_customer_churn_pipeline
```

Show this task chain:

1. `prepare_raw_data`: deterministic fixture or real IBM Telco ingestion.
2. `spark_processing`: typed validation and Delta write.
3. `dbt_build`: staging, derived metrics, feature mart, and tests.
4. `train_models`: logistic/XGBoost evaluation and MLflow tracking.
5. `registry_lifecycle`: registration, gates, challenger/champion aliases.
6. `verify_champion`: Registry reload and five real predictions.

Wait for the DAG to succeed before demonstrating `/predict`; a fresh MLflow
registry intentionally has no champion.

## MLflow demo

At <http://localhost:5000>, show:

- candidate parameters and validation metrics;
- final test metrics only on the selected candidate;
- classification reports, confusion matrices, signature, and feature schema;
- Registered Model versions and the `champion` alias;
- the source run and logged model lineage.

## Prediction API demo

```bash
curl --fail http://localhost:8001/health
curl --fail http://localhost:8001/model/info
```

Use the exact feature contract below:

```bash
curl --fail --request POST http://localhost:8001/predict \
  --header 'Content-Type: application/json' \
  --data '{
    "tenure": 5,
    "monthly_charges": 75.5,
    "total_charges": 377.5,
    "senior_citizen": 0,
    "gender": "Female",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "has_internet": 1,
    "has_phone": 1,
    "service_count": 5,
    "is_month_to_month": 1,
    "has_long_term_contract": 0,
    "monthly_to_total_charge_ratio": 0.2,
    "tenure_group": "0-12 months"
  }'
```

Explain that Pydantic validates the request, then the API invokes the complete
registered preprocessing pipeline rather than recreating encoders or scalers.

## Monitoring demo

Run both deterministic scenarios:

```bash
docker compose exec airflow-scheduler \
  python -m churn_platform.monitoring.run --scenario normal
docker compose exec airflow-scheduler \
  python -m churn_platform.monitoring.run --scenario drifted
```

Evidently compares reference/current distributions. The metrics exporter
publishes drift, data quality, model and Airflow state. The API publishes
inference count, failures, class distribution, and latency. Prometheus scrapes
both services; Grafana provisions the dashboard from Git.

## Kubernetes demo

```bash
kind create cluster --name churn-mlops --config k8s/kind-config.yaml
docker build -t churn-mlops:local .
kind load docker-image churn-mlops:local --name churn-mlops
kubectl apply -k k8s/overlays/local
kubectl get pods,svc -n churn-mlops
kubectl port-forward -n churn-mlops svc/serving-api 8001:8001
```

The local overlay contains explicitly marked disposable development
credentials; the reusable base and AWS overlay contain none. Render-only
validation needs no cluster:

```bash
kubectl kustomize k8s/overlays/local >/tmp/churn-local.yaml
kubectl kustomize k8s/overlays/aws >/tmp/churn-aws.yaml
```

## Terraform demo

These commands create no AWS resources:

```bash
cd infra/terraform/environments/dev
terraform fmt -check -recursive ../..
terraform init -backend=false
terraform validate
```

Do not include `terraform plan` in the standard demo: this configuration uses
AWS account and availability-zone data sources, so a meaningful plan requires
AWS authentication. Never run `apply` for a portfolio walkthrough.

## Stop local services

```bash
docker compose down
kind delete cluster --name churn-mlops
```

Add `--volumes` to `docker compose down` only when intentionally deleting all
local demo databases and artifacts.
