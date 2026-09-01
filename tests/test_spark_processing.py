"""Spark and Delta Lake tests for the Telco processing layer."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType

from churn_platform.spark.session import create_spark_session
from churn_platform.spark.transform_telco import (
    BUSINESS_COLUMNS,
    TECHNICAL_COLUMNS,
    TELCO_RAW_SCHEMA,
    SparkDataValidationError,
    add_technical_metadata,
    read_raw_telco,
    transform_telco,
    validate_transformed_telco,
    write_delta,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def spark(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SparkSession]:
    """Start one Delta-enabled SparkSession for the complete test run."""
    warehouse = tmp_path_factory.mktemp("spark-warehouse")
    session = create_spark_session(
        app_name="churn-platform-tests", warehouse_path=warehouse
    )
    yield session
    session.stop()


def sample_rows() -> list[dict[str, object]]:
    """Return two raw rows representing valid Telco source data."""
    return [
        {
            "customerID": "0001-AAAAA",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85",
            "Churn": "No",
        },
        {
            "customerID": "0002-BBBBB",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 89.1,
            "TotalCharges": "   ",
            "Churn": "Yes",
        },
    ]


def raw_dataframe(
    spark: SparkSession, *, overrides: dict[str, object] | None = None
) -> DataFrame:
    """Create a small raw DataFrame with optional first-row overrides."""
    rows = sample_rows()
    if overrides:
        rows[0].update(overrides)
    values = [tuple(row[column] for column in BUSINESS_COLUMNS) for row in rows]
    return spark.createDataFrame(values, schema=TELCO_RAW_SCHEMA)


def write_raw_csv(path: Path) -> Path:
    """Write the synthetic rows as a raw CSV input."""
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=BUSINESS_COLUMNS)
        writer.writeheader()
        writer.writerows(sample_rows())
    return path


@pytest.fixture(scope="session")
def delta_output(spark: SparkSession, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write one real Delta dataset reused by read-back assertions."""
    output_path = tmp_path_factory.mktemp("delta-output") / "telco"
    transformed = transform_telco(raw_dataframe(spark))
    row_count = validate_transformed_telco(transformed, input_row_count=2)
    enriched = add_technical_metadata(transformed, Path("synthetic.csv"))
    write_delta(spark, enriched, output_path, expected_rows=row_count)
    return output_path


def test_spark_session_starts_successfully(spark: SparkSession) -> None:
    assert spark.sparkContext.master == "local[2]"
    assert spark.range(1).count() == 1


def test_spark_version_is_available(spark: SparkSession) -> None:
    assert spark.version
    assert spark.version.startswith("4.2.")


def test_raw_sample_loads_with_explicit_schema(
    spark: SparkSession, tmp_path: Path
) -> None:
    raw_path = write_raw_csv(tmp_path / "telco.csv")

    loaded = read_raw_telco(spark, raw_path)

    assert loaded.schema == TELCO_RAW_SCHEMA
    assert loaded.count() == 2


def test_output_row_count_equals_input_count(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark))

    output_count = validate_transformed_telco(transformed, input_row_count=2)

    assert output_count == 2


def test_valid_total_charges_becomes_numeric(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark))

    first_row = transformed.where("customerID = '0001-AAAAA'").first()

    assert first_row is not None
    assert first_row["TotalCharges"] == pytest.approx(29.85)
    assert isinstance(transformed.schema["TotalCharges"].dataType, DoubleType)


def test_blank_total_charges_becomes_null(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark))

    second_row = transformed.where("customerID = '0002-BBBBB'").first()

    assert second_row is not None
    assert second_row["TotalCharges"] is None


def test_monthly_charges_has_numeric_type(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark))

    assert isinstance(transformed.schema["MonthlyCharges"].dataType, DoubleType)


def test_tenure_has_integer_type(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark))

    assert isinstance(transformed.schema["tenure"].dataType, IntegerType)


def test_invalid_churn_fails_validation(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark, overrides={"Churn": "Maybe"}))

    with pytest.raises(SparkDataValidationError, match="Churn"):
        validate_transformed_telco(transformed, input_row_count=2)


def test_duplicate_customer_id_fails_validation(spark: SparkSession) -> None:
    transformed = transform_telco(
        raw_dataframe(spark, overrides={"customerID": "0002-BBBBB"})
    )

    with pytest.raises(SparkDataValidationError, match="duplicate"):
        validate_transformed_telco(transformed, input_row_count=2)


def test_negative_tenure_fails_validation(spark: SparkSession) -> None:
    transformed = transform_telco(raw_dataframe(spark, overrides={"tenure": -1}))

    with pytest.raises(SparkDataValidationError, match="tenure"):
        validate_transformed_telco(transformed, input_row_count=2)


def test_delta_write_succeeds(delta_output: Path) -> None:
    assert delta_output.is_dir()
    assert any(delta_output.glob("*.parquet"))


def test_delta_dataset_can_be_read_back(
    spark: SparkSession, delta_output: Path
) -> None:
    reloaded = spark.read.format("delta").load(str(delta_output))

    assert reloaded.count() == 2


def test_delta_log_is_created(delta_output: Path) -> None:
    assert (delta_output / "_delta_log").is_dir()
    assert any((delta_output / "_delta_log").iterdir())


def test_processed_data_has_technical_metadata(
    spark: SparkSession, delta_output: Path
) -> None:
    reloaded = spark.read.format("delta").load(str(delta_output))

    assert tuple(reloaded.columns[-2:]) == TECHNICAL_COLUMNS
    assert reloaded.where("_processed_at IS NULL OR _source_file IS NULL").count() == 0


def test_missing_raw_input_has_useful_error(
    spark: SparkSession, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="churn_platform.ingestion.telco"):
        read_raw_telco(spark, missing_path)
