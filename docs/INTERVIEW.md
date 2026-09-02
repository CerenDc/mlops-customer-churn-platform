# Technical interview cheat sheet

## Explain the project in 30 seconds

It is a production-style customer-churn MLOps platform, not just a notebook.
It ingests IBM Telco data, validates and transforms it with Spark/Delta and dbt,
trains reproducible sklearn/XGBoost pipelines, tracks and promotes them through
MLflow, orchestrates the lifecycle with Airflow, serves the champion through a
typed FastAPI, and monitors drift and inference with Evidently, Prometheus, and
Grafana. Docker, Kubernetes/Kustomize, GitHub Actions, and Terraform provide
reproducibility, deployment definitions, CI, and an undeployed AWS target.

## Explain the architecture

The data plane is CSV → RAW → Spark → Delta → dbt/DuckDB feature mart. The ML
plane trains candidates, records them in MLflow, applies deterministic promotion
gates, and assigns the `champion` alias. FastAPI resolves that alias and invokes
the complete preprocessing pipeline. Airflow orchestrates, Prometheus/Grafana
observe, Docker packages, Kubernetes deploys, and Terraform describes AWS.

## Why Spark and Delta Lake?

Spark demonstrates typed, scalable transformations and data-quality failures.
Delta provides transactional writes and schema/history metadata. Local mode is
appropriate for 7,043 rows; a distributed cluster would be unjustified today.

## Why dbt and DuckDB?

dbt separates SQL modeling and tests from Python processing and makes feature
lineage reviewable. DuckDB is fast, embedded, CI-friendly, and sufficient for
the local feature mart. It is not a networked production warehouse.

## Why MLflow?

It connects parameters, metrics, dataset lineage, artifacts, signatures, runs,
and registered versions. This makes the selected model reproducible and
traceable rather than an unnamed pickle.

## How does the Registry work and how is champion selected?

The lifecycle finds the latest run tagged `selected`, verifies its final test
metrics and READY logged pipeline, then registers or reuses its version. It
assigns `challenger`, evaluates absolute ROC-AUC/F1/recall floors and incumbent
guardrails, and moves `champion` only when every gate passes. It reloads the
alias and makes five validation predictions. Repeated runs are idempotent.

## Why Airflow?

The six stages have dependencies, retries, logs, and operational state. Airflow
models those concerns while keeping each component independently executable.

## How do Docker and Kubernetes fit together?

Docker creates one repeatable Python 3.13/Java 17 runtime. Compose connects
services for local development. Kubernetes adds scheduling, probes, resources,
service discovery, and persistent volumes. Kustomize keeps a common base and
small local/AWS differences.

## Why Kubernetes rather than Compose in production? Why Kustomize?

Compose is excellent for a single host but lacks a cluster scheduler and
declarative rollout primitives. Kubernetes is useful when availability,
scaling, policy, and multi-node operations justify its complexity. Kustomize
avoids copied YAML while preserving native manifests.

## How is drift detected?

Evidently compares a reference feature sample to the current sample and emits
dataset and per-feature drift results. A deterministic drifted scenario makes
the behavior demoable. Drift recommends investigation/retraining; it does not
silently promote a new model.

## Data drift versus model-performance drift

Data drift means input distributions changed; it can be measured without new
labels. Performance drift means predictive metrics degraded and requires
ground-truth outcomes. The project reports model metrics when labels exist and
does not infer performance degradation from feature drift alone.

## Why Prometheus and Grafana?

Prometheus collects bounded numerical time series from pipeline monitoring and
the serving API. Grafana renders provisioned dashboards. This separates metric
collection from presentation and keeps dashboards version-controlled.

## Why FastAPI, and how is training-serving skew avoided?

FastAPI provides Pydantic validation and OpenAPI with little framework code.
The model artifact is the full fitted sklearn pipeline, including imputers,
scalers, and encoders. Serving loads `models:/...@champion`; it never refits or
recreates preprocessing. The request fields match the dbt feature contract.

## Why Terraform and AWS EKS? Why not SageMaker?

Terraform makes network, ECR, EKS, RDS, S3, and IAM reviewable and reproducible.
EKS extends the Kubernetes implementation already in the project and highlights
portable operations, at the cost of complexity and charges. SageMaker could be
sensible for a managed-first organization, but replacing a working MLflow and
Kubernetes lifecycle would obscure the learning objective.

## Why is AWS not deployed automatically?

PR validation should not incur cost or mutate an account. Deployment requires
credential design, remote state, cost review, protected environments, approval,
and rollback ownership. The repository validates IaC only and makes no AWS
deployment claim.

## What happens in CI?

CI installs Python 3.13 dependencies, checks Ruff, splits unit/component and
Spark integration tests, validates Airflow discovery, executes the full
synthetic DAG, runs monitoring scenarios, validates Compose and both Kustomize
overlays, builds the image, checks its runtime, then separately runs Terraform
fmt, backend-free init, and validate. It has read-only repository permissions
and no AWS credentials.

## How would this scale to millions of customers?

Measure first. Move Delta to tested object-storage-backed paths, execute Spark
on appropriately sized distributed infrastructure, replace local DuckDB with a
concurrent analytical platform while retaining dbt contracts, partition data,
separate batch and online inference, autoscale stateless API replicas, load test,
and introduce SLOs. Preserve signatures, lineage, promotion, and observability.

## How would you secure it?

Use private subnets/endpoints, TLS, authenticated ingress, network policies,
non-root/read-only containers, vulnerability scanning, least-privilege IRSA,
encrypted storage, audit logs, dependency updates, and isolated environments.
Never expose MLflow, Airflow, Prometheus, Grafana, or PostgreSQL publicly by
default.

## How would you handle secrets?

Store them in Secrets Manager, synchronize or inject them at runtime, rotate
them, restrict access by workload identity, and keep them out of Terraform
outputs, Git, images, ConfigMaps, and logs. The AWS overlay deliberately expects
a runtime Kubernetes Secret rather than committing values.

## How would you implement CD?

Use GitHub OIDC, build once, scan and push an immutable ECR digest, review a
Terraform plan, require protected-environment approval, apply from controlled
state, deploy a digest-pinned Kustomize overlay, wait for readiness, run smoke
tests, and roll back to the prior image/model alias when needed.

## Current limitations and next improvements

The system is a portfolio implementation, not serving real users. AWS is not
deployed; local Spark and DuckDB are not distributed; Kubernetes PVC sharing is
dev-oriented; secrets are not synchronized automatically; there is no public
TLS ingress, load test, SLO alerting, HA proof, restore drill, online feature
store, or automated CD. The next work should be evidence-driven hardening,
performance testing, champion hot-refresh after alias promotion, and deployment
controls—not additional random tools.

## How would you deploy it for a real client?

First clarify volume, latency, label availability, compliance, recovery, and
budget requirements. Then choose managed versus self-hosted components from
those constraints, establish separate accounts/environments and remote state,
secure networking and secrets, define data contracts and SLOs, validate cost
and restore procedures, implement controlled CI/CD, load test, stage a canary,
and document operational ownership.
