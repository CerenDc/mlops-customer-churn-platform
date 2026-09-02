output "endpoint" { value = aws_db_instance.this.endpoint }
output "secret_arn" { value = aws_secretsmanager_secret.database.arn }
