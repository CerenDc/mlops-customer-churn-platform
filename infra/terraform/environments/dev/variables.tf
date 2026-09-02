variable "project_name" {
  type    = string
  default = "mlops-customer-churn-platform"
}
variable "environment" {
  type    = string
  default = "dev"
}
variable "aws_region" {
  type    = string
  default = "eu-west-3"
}
variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}
variable "enable_nat_gateway" {
  type    = bool
  default = true
}
variable "kubernetes_version" {
  type    = string
  default = "1.34"
}
variable "node_instance_types" {
  type    = list(string)
  default = ["m7i.large"]
}
variable "node_min_size" {
  type    = number
  default = 1
}
variable "node_desired_size" {
  type    = number
  default = 2
}
variable "node_max_size" {
  type    = number
  default = 3
}
variable "rds_instance_class" {
  type    = string
  default = "db.t4g.micro"
}
variable "rds_allocated_storage" {
  type    = number
  default = 20
}
variable "rds_multi_az" {
  type    = bool
  default = false
}
variable "rds_deletion_protection" {
  type    = bool
  default = false
}
variable "rds_skip_final_snapshot" {
  type    = bool
  default = true
}
variable "artifact_bucket_name" {
  type     = string
  default  = null
  nullable = true
}
