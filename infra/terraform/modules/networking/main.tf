module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.7.0"

  name = var.name
  cidr = var.vpc_cidr
  azs  = var.availability_zones

  private_subnets = [for index, _ in var.availability_zones : cidrsubnet(var.vpc_cidr, 4, index)]
  public_subnets  = [for index, _ in var.availability_zones : cidrsubnet(var.vpc_cidr, 4, index + 8)]

  enable_nat_gateway     = var.enable_nat_gateway
  single_nat_gateway     = true
  one_nat_gateway_per_az = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  private_subnet_tags = { "kubernetes.io/role/internal-elb" = "1" }
  public_subnet_tags  = { "kubernetes.io/role/elb" = "1" }
  tags                = var.tags
}
