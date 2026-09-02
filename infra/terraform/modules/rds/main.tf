resource "random_password" "master" {
  length  = 32
  special = false
}
resource "aws_db_subnet_group" "this" {
  name       = var.name
  subnet_ids = var.subnet_ids
  tags       = var.tags
}
resource "aws_security_group" "this" {
  name_prefix = "${var.name}-rds-"
  description = "PostgreSQL access from EKS nodes only"
  vpc_id      = var.vpc_id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.eks_node_security_group_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = var.tags
}
resource "aws_db_instance" "this" {
  identifier                   = var.name
  engine                       = "postgres"
  engine_version               = "17.6"
  instance_class               = var.instance_class
  allocated_storage            = var.allocated_storage
  max_allocated_storage        = var.allocated_storage * 2
  storage_type                 = "gp3"
  storage_encrypted            = true
  db_name                      = "airflow"
  username                     = "platform_admin"
  password                     = random_password.master.result
  db_subnet_group_name         = aws_db_subnet_group.this.name
  vpc_security_group_ids       = [aws_security_group.this.id]
  publicly_accessible          = false
  multi_az                     = var.multi_az
  backup_retention_period      = 7
  deletion_protection          = var.deletion_protection
  skip_final_snapshot          = var.skip_final_snapshot
  final_snapshot_identifier    = var.skip_final_snapshot ? null : "${var.name}-final"
  auto_minor_version_upgrade   = true
  apply_immediately            = false
  performance_insights_enabled = false
  tags                         = var.tags
}
resource "aws_secretsmanager_secret" "database" {
  name                    = "${var.name}/database"
  recovery_window_in_days = 7
  tags                    = var.tags
}
resource "aws_secretsmanager_secret_version" "database" {
  secret_id = aws_secretsmanager_secret.database.id
  secret_string = jsonencode({
    host             = aws_db_instance.this.address
    port             = 5432
    username         = aws_db_instance.this.username
    password         = random_password.master.result
    airflow_database = aws_db_instance.this.db_name
    mlflow_database  = "mlflow"
  })
}
