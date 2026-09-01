"""Create a small genuine Delta dataset for deterministic CI validation."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession

from churn_platform.spark.session import create_spark_session
from churn_platform.spark.transform_telco import (
    TELCO_RAW_SCHEMA,
    add_technical_metadata,
    transform_telco,
    validate_transformed_telco,
    write_delta,
)
from churn_platform.testing.synthetic_telco import (
    SYNTHETIC_ROW_COUNT,
    SYNTHETIC_TELCO_ROWS,
)


def create_synthetic_delta_fixture(
    output_path: Path, *, spark: SparkSession | None = None
) -> int:
    """Create and validate a two-row Delta source suitable for dbt CI builds."""
    owns_session = spark is None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if spark is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="churn-spark-ci-")
        spark = create_spark_session(
            app_name="churn-platform-ci-fixture",
            warehouse_path=Path(temporary_directory.name) / "warehouse",
        )

    try:
        raw_dataset = spark.createDataFrame(
            SYNTHETIC_TELCO_ROWS, schema=TELCO_RAW_SCHEMA
        )
        processed_dataset = transform_telco(raw_dataset)
        row_count = validate_transformed_telco(
            processed_dataset, input_row_count=SYNTHETIC_ROW_COUNT
        )
        enriched_dataset = add_technical_metadata(
            processed_dataset, Path("synthetic-ci-fixture")
        )
        return write_delta(
            spark,
            enriched_dataset,
            output_path,
            expected_rows=row_count,
        )
    finally:
        if owns_session:
            spark.stop()
        if temporary_directory is not None:
            temporary_directory.cleanup()


def main() -> None:
    """Create a synthetic Delta fixture from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    row_count = create_synthetic_delta_fixture(arguments.output)
    print(f"Synthetic Delta fixture created: {arguments.output} ({row_count} rows)")


if __name__ == "__main__":
    main()
