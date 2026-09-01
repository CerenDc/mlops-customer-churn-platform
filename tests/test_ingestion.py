"""Unit tests for the IBM Telco dataset ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pandas as pd
import pytest

from churn_platform.ingestion.telco import (
    EXPECTED_COLUMNS,
    DatasetValidationError,
    calculate_sha256,
    download_telco_dataset,
    validate_telco_dataset,
)


@pytest.fixture
def valid_dataset() -> pd.DataFrame:
    """Return a minimal dataset matching the raw Telco schema."""
    rows = [
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
            "tenure": 12,
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
            "TotalCharges": "1069.20",
            "Churn": "Yes",
        },
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def write_dataset(dataset: pd.DataFrame, path: Path) -> Path:
    """Write a synthetic CSV fixture and return its path."""
    dataset.to_csv(path, index=False)
    return path


def csv_bytes(dataset: pd.DataFrame) -> bytes:
    """Serialize a synthetic dataset for mocked HTTP responses."""
    return dataset.to_csv(index=False).encode()


def mock_client(content: bytes, status_code: int = 200) -> httpx.Client:
    """Create an HTTP client backed by a deterministic in-memory transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=content, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_valid_dataset_passes_validation(
    tmp_path: Path, valid_dataset: pd.DataFrame
) -> None:
    dataset_path = write_dataset(valid_dataset, tmp_path / "valid.csv")

    validated = validate_telco_dataset(dataset_path)

    assert len(validated) == 2
    assert tuple(validated.columns) == EXPECTED_COLUMNS


def test_missing_required_column_fails(
    tmp_path: Path, valid_dataset: pd.DataFrame
) -> None:
    dataset_path = write_dataset(
        valid_dataset.drop(columns="Churn"), tmp_path / "missing.csv"
    )

    with pytest.raises(DatasetValidationError, match="missing columns: Churn"):
        validate_telco_dataset(dataset_path)


def test_unexpected_column_fails(tmp_path: Path, valid_dataset: pd.DataFrame) -> None:
    valid_dataset["Unexpected"] = "value"
    dataset_path = write_dataset(valid_dataset, tmp_path / "unexpected.csv")

    with pytest.raises(DatasetValidationError, match="unexpected columns: Unexpected"):
        validate_telco_dataset(dataset_path)


def test_invalid_churn_value_fails(tmp_path: Path, valid_dataset: pd.DataFrame) -> None:
    valid_dataset.loc[0, "Churn"] = "Maybe"
    dataset_path = write_dataset(valid_dataset, tmp_path / "invalid-churn.csv")

    with pytest.raises(DatasetValidationError, match="Churn contains invalid values"):
        validate_telco_dataset(dataset_path)


def test_duplicated_customer_id_fails(
    tmp_path: Path, valid_dataset: pd.DataFrame
) -> None:
    valid_dataset.loc[1, "customerID"] = valid_dataset.loc[0, "customerID"]
    dataset_path = write_dataset(valid_dataset, tmp_path / "duplicate.csv")

    with pytest.raises(DatasetValidationError, match="duplicate values"):
        validate_telco_dataset(dataset_path)


def test_null_customer_id_fails(tmp_path: Path, valid_dataset: pd.DataFrame) -> None:
    valid_dataset.loc[0, "customerID"] = None
    dataset_path = write_dataset(valid_dataset, tmp_path / "null-id.csv")

    with pytest.raises(DatasetValidationError, match="customerID contains null values"):
        validate_telco_dataset(dataset_path)


@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges"])
def test_negative_numeric_value_fails(
    tmp_path: Path, valid_dataset: pd.DataFrame, column: str
) -> None:
    valid_dataset.loc[0, column] = -1
    dataset_path = write_dataset(valid_dataset, tmp_path / f"negative-{column}.csv")

    with pytest.raises(
        DatasetValidationError, match=f"{column} contains negative values"
    ):
        validate_telco_dataset(dataset_path)


def test_http_failure_does_not_create_final_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "raw" / "telco.csv"
    with (
        mock_client(b"service unavailable", status_code=503) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        download_telco_dataset(output_path, client=client)

    assert not output_path.exists()
    assert not any(path.suffix == ".tmp" for path in output_path.parent.iterdir())


def test_successful_download_creates_expected_file(
    tmp_path: Path, valid_dataset: pd.DataFrame
) -> None:
    content = csv_bytes(valid_dataset)
    output_path = tmp_path / "nested" / "raw" / "telco.csv"
    with mock_client(content) as client:
        download_telco_dataset(output_path, client=client)

    assert output_path.read_bytes() == content


def test_ingestion_returns_correct_metadata(
    tmp_path: Path, valid_dataset: pd.DataFrame
) -> None:
    content = csv_bytes(valid_dataset)
    output_path = tmp_path / "telco.csv"
    with mock_client(content) as client:
        result = download_telco_dataset(output_path, client=client)

    assert result.output_path == output_path
    assert result.rows == 2
    assert result.columns == 21
    assert result.file_size_bytes == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()


def test_checksum_is_generated_consistently(tmp_path: Path) -> None:
    dataset_path = tmp_path / "content.csv"
    dataset_path.write_bytes(b"repeatable content\n")

    first_checksum = calculate_sha256(dataset_path)
    second_checksum = calculate_sha256(dataset_path)

    assert first_checksum == second_checksum
    assert first_checksum == hashlib.sha256(b"repeatable content\n").hexdigest()
