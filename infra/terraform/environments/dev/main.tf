data "aws_availability_zones" "available" { state = "available" }
data "aws_caller_identity" "current" {}
locals {
  name = "${var.project_name}-${var.environment}"
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Repository  = "mlops-customer-churn-platform"
  }
  availability_zones   = slice(data.aws_availability_zones.available.names, 0, 2)
  artifact_bucket_name = coalesce(var.artifact_bucket_name, "${local.name}-artifacts-${data.aws_caller_identity.current.account_id}")
}
module "networking" {
  source             = "../../modules/networking"
  name               = local.name
  vpc_cidr           = var.vpc_cidr
  availability_zones = local.availability_zones
  enable_nat_gateway = var.enable_nat_gateway
  tags               = local.tags
}
module "ecr" {
  source = "../../modules/ecr"
  name   = var.project_name
  tags   = local.tags
}
module "eks" {
  source              = "../../modules/eks"
  name                = local.name
  kubernetes_version  = var.kubernetes_version
  vpc_id              = module.networking.vpc_id
  private_subnet_ids  = module.networking.private_subnet_ids
  node_instance_types = var.node_instance_types
  node_min_size       = var.node_min_size
  node_desired_size   = var.node_desired_size
  node_max_size       = var.node_max_size
  tags                = local.tags
}
module "s3" {
  source      = "../../modules/s3"
  bucket_name = local.artifact_bucket_name
  tags        = local.tags
}
module "rds" {
  source                     = "../../modules/rds"
  name                       = local.name
  vpc_id                     = module.networking.vpc_id
  subnet_ids                 = module.networking.private_subnet_ids
  eks_node_security_group_id = module.eks.node_security_group_id
  instance_class             = var.rds_instance_class
  allocated_storage          = var.rds_allocated_storage
  multi_az                   = var.rds_multi_az
  deletion_protection        = var.rds_deletion_protection
  skip_final_snapshot        = var.rds_skip_final_snapshot
  tags                       = local.tags
}
module "iam" {
  source            = "../../modules/iam"
  name              = local.name
  bucket_arn        = module.s3.bucket_arn
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider     = module.eks.oidc_provider
  tags              = local.tags
}
