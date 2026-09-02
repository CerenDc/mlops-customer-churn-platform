# Portfolio copy

## GitHub short description

End-to-end churn MLOps platform: Spark, Delta, dbt, MLflow, Airflow, FastAPI,
Kubernetes, monitoring, CI and Terraform/AWS IaC.

## CV version

- Built a production-style churn platform from raw IBM Telco data to a typed
  FastAPI serving the MLflow Registry champion model.
- Industrialized Spark/Delta processing, dbt/DuckDB feature engineering, and
  reproducible sklearn/XGBoost training with deterministic promotion gates.
- Orchestrated and observed the lifecycle with Airflow, Evidently, Prometheus,
  Grafana, Docker Compose, Kubernetes/Kustomize, and automated GitHub Actions.
- Designed and validated modular Terraform for an AWS target using VPC, EKS,
  ECR, RDS, S3, Secrets Manager, and least-privilege IRSA; no cloud resources
  are automatically provisioned.

## Malt title

Ingénieure MLOps / Data — Industrialisation de pipelines ML de bout en bout

## Malt description

Je conçois des plateformes ML reproductibles qui relient préparation des
données, entraînement, traçabilité, déploiement et monitoring. Ce projet
portfolio industrialise un cas de prédiction du churn avec Spark, Delta Lake,
dbt, MLflow, Airflow et FastAPI, puis le package avec Docker et Kubernetes. La
qualité est automatisée par GitHub Actions et l’architecture cible AWS est
décrite en Terraform, avec une attention particulière portée aux contrats de
données, au cycle champion/challenger, à l’observabilité et à la sécurité.

## LinkedIn project description

Production-style customer-churn MLOps platform covering Spark/Delta processing,
dbt feature engineering, sklearn/XGBoost training, MLflow tracking and model
promotion, Airflow orchestration, FastAPI inference, drift monitoring, Docker,
Kubernetes, CI, and validated Terraform for an AWS target architecture.

## LinkedIn post draft

J’arrive au terme d’un projet personnel qui m’a permis de travailler le cycle
MLOps au-delà de l’entraînement d’un modèle : une plateforme de prédiction du
churn client, construite progressivement de la donnée brute jusqu’au serving.

Le pipeline transforme le dataset IBM Telco avec Spark et Delta Lake, construit
un feature mart testé avec dbt/DuckDB, compare des pipelines scikit-learn et
XGBoost, puis trace et versionne les résultats avec MLflow. Airflow orchestre le
workflow, un cycle champion/challenger contrôle la promotion, et FastAPI charge
le vrai modèle `champion` avec son preprocessing pour produire les prédictions.

J’ai également travaillé la partie opérationnelle : Docker Compose, manifests
Kubernetes réutilisables avec Kustomize, drift Evidently, métriques Prometheus,
dashboard Grafana, CI GitHub Actions et architecture AWS décrite en Terraform.

Point important : il s’agit d’un projet d’ingénierie portfolio, pas d’une
plateforme exploitée pour un client. L’infrastructure AWS est validée comme
Infrastructure as Code mais n’est pas déployée automatiquement. L’objectif
était justement d’apprendre à distinguer une démonstration reproductible d’une
affirmation de production.

Ce parcours m’a surtout appris à relier contrats de données, reproductibilité,
cycle de vie des modèles, observabilité, sécurité et coûts — et à choisir les
outils pour une responsabilité précise plutôt que pour allonger une stack.
