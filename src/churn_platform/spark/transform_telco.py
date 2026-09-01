"""Transform raw Telco churn data into a typed Delta Lake dataset."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from churn_platform.spark.session import create_spark_session

DEFAULT_INPUT_PATH = Path("data/raw/telco_customer_churn.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/telco_customer_churn_delta")

BUSINESS_COLUMNS = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)
TECHNICAL_COLUMNS = ("_processed_at", "_source_file")

TELCO_RAW_SCHEMA = StructType(
    [
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", IntegerType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("tenure", IntegerType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("MonthlyCharges", DoubleType(), True),
        StructField("TotalCharges", StringType(), True),
        StructField("Churn", StringType(), True),
    ]
)

TELCO_PROCESSED_SCHEMA = StructType(
    [
        *TELCO_RAW_SCHEMA.fields[:19],
        StructField("TotalCharges", DoubleType(), True),
        TELCO_RAW_SCHEMA.fields[20],
    ]
)


class SparkDataValidationError(ValueError):
    """Raised when transformed Spark data violates the processing contract."""


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Metadata describing a successful Spark and Delta processing run."""

    input_path: Path
    output_path: Path
    input_rows: int
    output_rows: int
    business_columns: int
    total_columns: int
    spark_version: str
    delta_version: str


def read_raw_telco(spark: SparkSession, input_path: Path) -> DataFrame:
    """Read the raw Telco CSV with an explicit production schema."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Raw Telco dataset not found at {input_path}. Run "
            "`python -m churn_platform.ingestion.telco` first."
        )

    return (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("enforceSchema", True)
        .schema(TELCO_RAW_SCHEMA)
        .csv(str(input_path))
    )


def transform_telco(raw_dataset: DataFrame) -> DataFrame:
    """Normalize raw field types without performing feature engineering."""
    normalized_columns = []
    for field in TELCO_PROCESSED_SCHEMA.fields:
        if field.name == "TotalCharges":
            normalized_columns.append(
                F.expr("try_cast(nullif(trim(TotalCharges), '') as double)").alias(
                    field.name
                )
            )
        else:
            normalized_columns.append(
                F.col(field.name).cast(field.dataType).alias(field.name)
            )
    return raw_dataset.select(*normalized_columns)


def validate_transformed_telco(dataset: DataFrame, *, input_row_count: int) -> int:
    """Validate processed business data and return its row count."""
    if tuple(dataset.columns) != BUSINESS_COLUMNS:
        raise SparkDataValidationError(
            "Processed dataset must contain exactly the 21 ordered business columns"
        )

    metrics = dataset.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(F.col("customerID").isNull(), 1).otherwise(0)).alias(
            "null_customer_ids"
        ),
        F.sum(
            F.when(
                F.col("Churn").isNull() | ~F.col("Churn").isin("Yes", "No"),
                1,
            ).otherwise(0)
        ).alias("invalid_churn"),
        F.sum(
            F.when(
                F.col("SeniorCitizen").isNull() | ~F.col("SeniorCitizen").isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_senior_citizen"),
        F.sum(
            F.when(F.col("tenure").isNull() | (F.col("tenure") < 0), 1).otherwise(0)
        ).alias("invalid_tenure"),
        F.sum(
            F.when(
                F.col("MonthlyCharges").isNull() | (F.col("MonthlyCharges") < 0),
                1,
            ).otherwise(0)
        ).alias("invalid_monthly_charges"),
        F.sum(F.when(F.col("TotalCharges") < 0, 1).otherwise(0)).alias(
            "invalid_total_charges"
        ),
    ).first()
    if metrics is None:
        raise SparkDataValidationError("Unable to calculate dataset quality metrics")

    row_count = metrics["row_count"]
    if row_count == 0:
        raise SparkDataValidationError("Processed dataset contains no rows")
    if row_count != input_row_count:
        raise SparkDataValidationError(
            f"Row count changed during transformation: {input_row_count} -> {row_count}"
        )

    checks = (
        ("null_customer_ids", "customerID contains null values"),
        ("invalid_churn", "Churn must contain only Yes or No"),
        ("invalid_senior_citizen", "SeniorCitizen must contain only 0 or 1"),
        ("invalid_tenure", "tenure must be non-null and non-negative"),
        (
            "invalid_monthly_charges",
            "MonthlyCharges must be non-null and non-negative",
        ),
        (
            "invalid_total_charges",
            "non-null TotalCharges values must be non-negative",
        ),
    )
    for metric_name, message in checks:
        if metrics[metric_name]:
            raise SparkDataValidationError(message)

    duplicate_exists = (
        dataset.groupBy("customerID").count().where(F.col("count") > 1).limit(1).count()
    )
    if duplicate_exists:
        raise SparkDataValidationError("customerID contains duplicate values")
    return row_count


def add_technical_metadata(dataset: DataFrame, source_file: Path) -> DataFrame:
    """Add minimal processing-time and source lineage metadata."""
    return dataset.withColumns(
        {
            "_processed_at": F.current_timestamp(),
            "_source_file": F.lit(str(source_file)),
        }
    )


def _verify_delta_dataset(
    spark: SparkSession, output_path: Path, expected_rows: int
) -> int:
    """Read and verify a genuine Delta dataset from disk."""
    if not (output_path / "_delta_log").is_dir():
        raise SparkDataValidationError(
            f"Delta transaction log was not created at {output_path}"
        )

    reloaded = spark.read.format("delta").load(str(output_path))
    expected_columns = (*BUSINESS_COLUMNS, *TECHNICAL_COLUMNS)
    if tuple(reloaded.columns) != expected_columns:
        raise SparkDataValidationError("Delta read-back returned an unexpected schema")
    for actual_field, expected_field in zip(
        reloaded.schema.fields[: len(BUSINESS_COLUMNS)],
        TELCO_PROCESSED_SCHEMA.fields,
        strict=True,
    ):
        if actual_field.dataType != expected_field.dataType:
            raise SparkDataValidationError(
                f"Delta field {actual_field.name} has type "
                f"{actual_field.dataType.simpleString()}, expected "
                f"{expected_field.dataType.simpleString()}"
            )
    if not isinstance(reloaded.schema["_processed_at"].dataType, TimestampType):
        raise SparkDataValidationError("_processed_at must be a timestamp")
    if not isinstance(reloaded.schema["_source_file"].dataType, StringType):
        raise SparkDataValidationError("_source_file must be a string")

    output_rows = reloaded.count()
    if output_rows != expected_rows:
        raise SparkDataValidationError(
            f"Delta read-back row count mismatch: {expected_rows} -> {output_rows}"
        )
    if not reloaded.limit(1).collect():
        raise SparkDataValidationError("Delta read-back query returned no data")
    return output_rows


def write_delta(
    spark: SparkSession,
    dataset: DataFrame,
    output_path: Path,
    *,
    expected_rows: int,
) -> int:
    """Write, verify, and publish a Delta dataset using local atomic renames."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
    )
    temporary_path.rmdir()
    backup_path: Path | None = None

    try:
        dataset.write.format("delta").mode("overwrite").save(str(temporary_path))
        _verify_delta_dataset(spark, temporary_path, expected_rows)

        if output_path.exists():
            backup_path = output_path.with_name(
                f".{output_path.name}.{uuid4().hex}.backup"
            )
            os.replace(output_path, backup_path)
        try:
            os.replace(temporary_path, output_path)
        except OSError:
            if backup_path is not None:
                os.replace(backup_path, output_path)
                backup_path = None
            raise

        output_rows = _verify_delta_dataset(spark, output_path, expected_rows)
        if backup_path is not None:
            shutil.rmtree(backup_path)
            backup_path = None
        return output_rows
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        if backup_path is not None and backup_path.exists():
            if not output_path.exists():
                os.replace(backup_path, output_path)
            else:
                shutil.rmtree(backup_path)


def run_processing_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    spark: SparkSession | None = None,
) -> ProcessingResult:
    """Run raw-to-Delta Telco processing and return structured metadata."""
    owns_session = spark is None
    active_spark = spark or create_spark_session()
    try:
        raw_dataset = read_raw_telco(active_spark, input_path)
        input_rows = raw_dataset.count()
        processed_dataset = transform_telco(raw_dataset)
        validated_rows = validate_transformed_telco(
            processed_dataset, input_row_count=input_rows
        )
        enriched_dataset = add_technical_metadata(processed_dataset, input_path)
        output_rows = write_delta(
            active_spark,
            enriched_dataset,
            output_path,
            expected_rows=validated_rows,
        )
        return ProcessingResult(
            input_path=Path(input_path),
            output_path=Path(output_path),
            input_rows=input_rows,
            output_rows=output_rows,
            business_columns=len(BUSINESS_COLUMNS),
            total_columns=len(enriched_dataset.columns),
            spark_version=active_spark.version,
            delta_version=version("delta-spark"),
        )
    finally:
        if owns_session:
            active_spark.stop()


def main() -> None:
    """Run the Spark processing pipeline from the command line."""
    try:
        result = run_processing_pipeline()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    print("Spark processing completed")
    print(f"Input rows: {result.input_rows}")
    print(f"Output rows: {result.output_rows}")
    print(f"Business columns: {result.business_columns}")
    print(f"Total columns: {result.total_columns}")
    print("Output format: Delta Lake")
    print(f"Spark: {result.spark_version}")
    print(f"Delta: {result.delta_version}")
    print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()
