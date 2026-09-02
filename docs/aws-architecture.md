# AWS architecture (V12)

V12 adds a production-oriented, portfolio-grade AWS deployment target without
replacing host, Docker Compose, or local Kubernetes execution. Terraform is the
infrastructure source of truth; Kustomize continues to describe the workloads.
No AWS resources have been deployed as part of V12.

## Service choices

- **EKS** extends the Kubernetes deployment model introduced in V10. It offers
  portability and managed control-plane operations at the cost of meaningful
  operational complexity and a paid control plane.
- **RDS PostgreSQL** provides durable Airflow and MLflow service metadata. One
  cost-conscious instance is intended to host separate logical databases. It
  does not replace DuckDB, which remains the embedded analytical feature store.
- **S3** is the private, encrypted artifact store for MLflow. Its `mlflow/`,
  `monitoring/`, and `data/` prefixes also provide an intended path for portable
  artifacts when application support is added; V12 does not pretend local
  Delta or DuckDB files are automatically S3-compatible.
- **ECR** holds the single reusable platform image with immutable tags, push
  scanning, and bounded retention.
- **IAM/IRSA** gives the `mlflow` and `pipeline` service accounts access only to
  the project artifact bucket. No long-lived AWS keys are placed in pods.
- **VPC** places EKS nodes and RDS in private subnets across two availability
zones. Public subnets and one optional NAT gateway provide deliberate dev
egress. RDS accepts PostgreSQL only from the EKS node security group.
- **Terraform** makes the network, compute, database, object storage, registry,
  and workload identities reproducible. Airflow migrations and the creation of
  the second logical database remain application bootstrap concerns, not
  Terraform `local-exec` provisioners.

Airflow remains self-hosted rather than moving to MWAA, MLflow remains the
registry rather than moving to SageMaker, and Prometheus/Grafana remain the
observability stack. Those choices preserve working project behavior and avoid
unnecessary managed services. Spark continues in local mode inside the pipeline
runtime; the current dataset does not justify EMR or a distributed Spark
cluster.

## Runtime and storage boundaries

The AWS overlay removes the in-cluster PostgreSQL StatefulSet. MLflow uses the
runtime-provided `MLFLOW_BACKEND_STORE_URI` for RDS and writes artifacts to the
configured S3 URI. Airflow reads its RDS SQLAlchemy connection from the
runtime-provided `churn-secrets` Kubernetes Secret. Terraform stores the RDS
administrator credential in AWS Secrets Manager, but V12 intentionally does
not install a secret synchronization controller. An operator must securely
create the Kubernetes Secret at deployment time and run an application
bootstrap that creates the `mlflow` logical database before MLflow starts.

Delta Lake, DuckDB, generated monitoring reports, Prometheus, and Grafana retain
their existing PVC-backed filesystem behavior. On EKS these claims are expected
to use the installed EBS CSI add-on and the cluster's EBS-backed default storage
class. Node root volumes are encrypted gp3 disks. `ReadWriteOnce` is not a
multi-node shared filesystem: pods sharing `pipeline-data` must be scheduled
compatibly, and this is a dev limitation rather than a high-availability claim.
S3 migration for those file-oriented workloads requires explicit application
and Hadoop connector work and is outside V12.

## Security and access

RDS is encrypted, private, backed up for seven days, and never accepts
`0.0.0.0/0` on port 5432. S3 blocks all public access, uses server-side
encryption, and is versioned. ECR tags are immutable. AWS-managed encryption is
used to keep this dev architecture understandable; customer-managed KMS keys
are a future hardening option.

The EKS API is IAM-authenticated and currently configured with a public
endpoint so a future operator can administer the dev cluster. Workloads are not
given public Kubernetes Services or an ALB. Airflow, MLflow, Prometheus, and
Grafana should initially be accessed through `kubectl port-forward`. Production
network restrictions, private API access, DNS, TLS, ingress, and external secret
synchronization are deliberately not claimed.

## Cost and resilience tradeoffs

Defaults use two `m7i.large` on-demand nodes (scaling from one to three), a
`db.t4g.micro` single-AZ RDS instance, one optional shared NAT gateway, 20 GiB
of RDS storage, no GPU, no ALB, and retention of only the newest 20 ECR images.
These are cost-conscious functional defaults, not free-tier or highly available
guarantees. EKS, EC2, NAT, RDS, EBS, S3, ECR, Secrets Manager, and data transfer
can all incur charges.

RDS and S3 provide durable managed persistence, ECR preserves immutable images,
Terraform reproduces infrastructure, and Git/Kustomize reproduce workloads.
This is not a complete disaster-recovery design: no restore drill, cross-region
replication, high-availability test, or cloud load test has been performed.

## State and future deployment

Local validation uses `terraform init -backend=false`. The supplied
`backend.hcl.example` documents an optional S3 backend using native S3 lockfiles;
the backend bucket must be bootstrapped separately and is never created by this
configuration. Active state, plans, local variables, and AWS credential files
are ignored by Git.

For a future deployment, use standard AWS authentication. Human workstations
may use AWS profiles or temporary environment credentials; GitHub Actions
should use OIDC rather than stored access keys. Review a Terraform plan and AWS
costs before any apply. Destruction is also a manual, destructive operation:
deletion protection, non-empty S3 buckets, snapshots, and retained backups can
prevent or outlive it, while application data can be permanently removed.
