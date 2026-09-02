# Visual asset checklist

Capture these screenshots manually from real running services. Do not stage
credentials, personal paths, account IDs, or cloud secrets in any image.

| Suggested filename | Open | What should be visible |
| --- | --- | --- |
| `architecture.png` | README Mermaid diagram | Complete data, ML, serving, monitoring, deployment flow |
| `airflow-dag.png` | `http://localhost:8080` | Successful `mlops_customer_churn_pipeline` graph and its six tasks |
| `mlflow-experiments.png` | `http://localhost:5000` | Candidate runs, parameters, validation/test metrics, and artifacts |
| `mlflow-registry.png` | MLflow Registered Models | Model versions and resolved `champion` alias |
| `serving-swagger.png` | `http://localhost:8001/docs` | `/health`, `/model/info`, and `/predict` contracts plus a real response |
| `grafana-dashboard.png` | `http://localhost:3000` | Drift, pipeline, champion, inference volume/failure/latency panels |
| `kubernetes-workloads.png` | Terminal: `kubectl get pods,svc -n churn-mlops` | Ready local workloads and ClusterIP services |
| `github-actions.png` | Repository Actions tab | One real green `MLOps Continuous Integration` run and both jobs |
| `terraform-validation.png` | Terminal in `infra/terraform/environments/dev` | Successful fmt, backend-free init, and validate; no apply output |

Crop only for readability. Keep timestamps and service names when they help
prove the run, and never fabricate a successful state.
