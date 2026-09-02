# Architecture decisions

These concise decisions describe the repository as implemented, including its
trade-offs. They are not claims of enterprise production operation.

## Data and feature engineering

### Spark and Delta Lake

Spark makes the processing contract explicit, typed, and scalable beyond a
single pandas script. Delta adds transactional writes, schema enforcement, and
an auditable table log. For the IBM Telco dataset this is deliberately local
Spark; a distributed cluster would add cost without improving the demo.

### dbt and DuckDB

dbt owns SQL transformations, tests, lineage, and the customer-grain feature
mart. DuckDB is a fast embedded analytical engine that keeps local and CI runs
self-contained. It is not presented as a concurrent production warehouse.

## Model lifecycle

### MLflow Tracking and Model Registry

MLflow records parameters, validation/test metrics, data lineage, model
signatures, artifacts, and the complete sklearn preprocessing pipeline. The
Registry gives deployed identity to a selected run instead of copying an
anonymous model file.

### Champion/challenger aliases

Registration assigns a candidate to `challenger`. Deterministic absolute and
incumbent-relative gates decide promotion. `champion` is the stable consumer
contract used by verification and serving; rerunning the lifecycle is
idempotent.

### FastAPI serving

FastAPI provides typed validation, OpenAPI documentation, health/readiness,
and low-overhead synchronous inference. The API loads the complete registered
sklearn pipeline through `models:/...@champion`, preventing a separately fitted
serving preprocessor. It fails with 503 when no champion exists; it never emits
fake predictions.

## Operations

### Airflow

Airflow expresses ordering, retries, logs, and operational state for ingestion,
Spark, dbt, training, registry promotion, and champion verification. The
components remain independently executable and testable.

### Docker, Kubernetes, and Kustomize

One Python/Java image packages all project code. Compose is the easiest local
multi-service demo. Kubernetes adds declarative scheduling, probes, resources,
and service discovery. Kustomize reuses one base for local kind and AWS targets
without duplicating manifests.

### Evidently, Prometheus, and Grafana

Evidently compares reference and current feature distributions. A lightweight
exporter exposes bounded pipeline/model/drift state, while the serving API
exports request, failure, prediction, and latency metrics. Prometheus stores
time series and Grafana provisions dashboards as code. Data drift is a signal
for investigation, not an automatic promotion trigger.

## Infrastructure and CI

### Terraform and AWS EKS target

Terraform defines a reproducible dev VPC, ECR, EKS, RDS PostgreSQL, S3, and
IRSA roles. EKS extends the Kubernetes model already demonstrated by the
project. This improves portability and demonstrates IaC, but costs more and is
operationally heavier than Compose or a managed inference service.

Airflow stays on EKS instead of MWAA, MLflow stays in place instead of
SageMaker, and Spark stays local-mode. RDS stores service metadata; it does not
replace DuckDB. See `aws-architecture.md` for cloud storage boundaries.

### No automatic cloud provisioning

Pull-request CI runs formatting, initialization without a backend, and
Terraform validation only. An apply requires credentials, cost review, change
approval, state management, and a controlled deployment identity. A future CD
workflow should use GitHub OIDC, immutable ECR tags, protected environments,
reviewed plans, and explicit approval.

### Synthetic data in CI

The deterministic 60-row fixture removes network and licensing dependencies
while exercising the real pipeline graph. A real 7,043-row IBM Telco Airflow
run has been validated separately; the dataset itself is not committed.

## Enterprise evolution

At larger scale, use object-storage-native Delta with tested Hadoop connectors,
a distributed Spark runtime only when measurement justifies it, managed
PostgreSQL HA, private endpoints, managed secret synchronization, TLS/ingress,
autoscaled serving replicas, an online contract or feature service, immutable
release promotion, alerting/SLOs, backup restore drills, and load/performance
tests. Those capabilities are not claimed by this portfolio implementation.
