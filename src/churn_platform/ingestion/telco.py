"""Download and validate the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

TELCO_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_OUTPUT_PATH = Path("data/raw/telco_customer_churn.csv")
HTTP_TIMEOUT_SECONDS = 30.0

EXPECTED_COLUMNS = (
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


class DatasetValidationError(ValueError):
    """Raised when the downloaded Telco dataset does not meet its contract."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Metadata describing a successfully ingested dataset."""

    output_path: Path
    rows: int
    columns: int
    file_size_bytes: int
    sha256: str


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_telco_dataset(path: Path) -> pd.DataFrame:
    """Validate the Telco CSV and return its unmodified tabular contents."""
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {path}")
    if path.stat().st_size == 0:
        raise DatasetValidationError(f"Dataset file is empty: {path}")

    try:
        dataset = pd.read_csv(path)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as error:
        raise DatasetValidationError(
            f"Dataset CSV could not be parsed: {error}"
        ) from error

    actual_columns = set(dataset.columns)
    expected_columns = set(EXPECTED_COLUMNS)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if len(dataset.columns) != len(EXPECTED_COLUMNS) or missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected columns: {', '.join(unexpected)}")
        if not details:
            details.append("column names are duplicated")
        raise DatasetValidationError("Invalid dataset schema; " + "; ".join(details))

    if dataset.empty:
        raise DatasetValidationError("Dataset contains no data rows")
    if dataset["customerID"].isna().any():
        raise DatasetValidationError("customerID contains null values")
    if not dataset["customerID"].is_unique:
        raise DatasetValidationError("customerID contains duplicate values")
    if dataset["Churn"].isna().any():
        raise DatasetValidationError("Churn contains null values")

    invalid_churn = sorted(
        str(value) for value in set(dataset["Churn"].dropna()) - {"Yes", "No"}
    )
    if invalid_churn:
        raise DatasetValidationError(
            f"Churn contains invalid values: {', '.join(invalid_churn)}"
        )

    invalid_senior = set(dataset["SeniorCitizen"].dropna()) - {0, 1}
    if dataset["SeniorCitizen"].isna().any() or invalid_senior:
        values = ", ".join(sorted(str(value) for value in invalid_senior))
        raise DatasetValidationError(
            "SeniorCitizen must contain only 0 or 1; "
            f"invalid values: {values or 'null'}"
        )

    _validate_non_negative_numeric(dataset, "tenure")
    _validate_non_negative_numeric(dataset, "MonthlyCharges")
    return dataset


def _validate_non_negative_numeric(dataset: pd.DataFrame, column: str) -> None:
    """Validate that a dataset column contains non-negative numeric values."""
    numeric_values = pd.to_numeric(dataset[column], errors="coerce")
    if numeric_values.isna().any():
        raise DatasetValidationError(f"{column} contains null or non-numeric values")
    if (numeric_values < 0).any():
        raise DatasetValidationError(f"{column} contains negative values")


def download_telco_dataset(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    source_url: str = TELCO_DATASET_URL,
    *,
    client: httpx.Client | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> IngestionResult:
    """Download, validate, and atomically publish the Telco dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if client is None:
        with httpx.Client(follow_redirects=True) as owned_client:
            response = owned_client.get(source_url, timeout=timeout)
    else:
        response = client.get(source_url, timeout=timeout)
    response.raise_for_status()

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(response.content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        dataset = validate_telco_dataset(temporary_path)
        file_size = temporary_path.stat().st_size
        checksum = calculate_sha256(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return IngestionResult(
        output_path=output_path,
        rows=len(dataset),
        columns=len(dataset.columns),
        file_size_bytes=file_size,
        sha256=checksum,
    )


def main() -> None:
    """Run the Telco ingestion pipeline from the command line."""
    output_path = Path(os.getenv("TELCO_RAW_PATH", str(DEFAULT_OUTPUT_PATH)))
    result = download_telco_dataset(output_path=output_path)
    print("Telco ingestion completed")
    print(f"Rows: {result.rows}")
    print(f"Columns: {result.columns}")
    print(f"Output: {result.output_path}")
    print(f"SHA256: {result.sha256}")


if __name__ == "__main__":
    main()
