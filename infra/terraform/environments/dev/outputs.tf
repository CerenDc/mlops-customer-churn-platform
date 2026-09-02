output "aws_region" { value = var.aws_region }
output "eks_cluster_name" { value = module.eks.cluster_name }
output "ecr_repository_url" { value = module.ecr.repository_url }
output "artifact_bucket_name" { value = module.s3.bucket_name }
output "rds_endpoint" { value = module.rds.endpoint }
output "database_secret_arn" { value = module.rds.secret_arn }
output "mlflow_role_arn" { value = module.iam.mlflow_role_arn }
output "pipeline_role_arn" { value = module.iam.pipeline_role_arn }
