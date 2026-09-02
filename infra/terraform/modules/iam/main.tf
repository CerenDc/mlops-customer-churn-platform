data "aws_iam_policy_document" "s3" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [var.bucket_arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${var.bucket_arn}/*"]
  }
}
resource "aws_iam_policy" "s3" {
  name   = "${var.name}-artifacts"
  policy = data.aws_iam_policy_document.s3.json
  tags   = var.tags
}
data "aws_iam_policy_document" "assume" {
  for_each = toset(["mlflow", "pipeline"])
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.oidc_provider}:sub"
      values   = ["system:serviceaccount:churn-mlops:${each.key}"]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "workload" {
  for_each           = data.aws_iam_policy_document.assume
  name               = "${var.name}-${each.key}"
  assume_role_policy = each.value.json
  tags               = var.tags
}
resource "aws_iam_role_policy_attachment" "workload" {
  for_each   = aws_iam_role.workload
  role       = each.value.name
  policy_arn = aws_iam_policy.s3.arn
}
