"""Generate deterministic raw Telco data for offline pipeline execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from churn_platform.ingestion.telco import EXPECTED_COLUMNS, validate_telco_dataset

SYNTHETIC_ROW_COUNT = 60


def synthetic_telco_rows() -> tuple[tuple[object, ...], ...]:
    """Return valid customer rows containing both churn classes."""
    rows = []
    contracts = ("Month-to-month", "One year", "Two year")
    for index in range(SYNTHETIC_ROW_COUNT):
        tenure = index % 60
        monthly_charges = 35.0 + (index % 20) * 3.0
        internet = "No" if index % 5 == 0 else ("DSL" if index % 2 else "Fiber optic")
        rows.append(
            (
                f"SYNTH-{index:04d}",
                "Female" if index % 2 == 0 else "Male",
                index % 2,
                "Yes" if index % 3 else "No",
                "Yes" if index % 4 else "No",
                tenure,
                "Yes",
                "Yes" if index % 3 == 0 else "No",
                internet,
                "Yes" if index % 2 else "No",
                "Yes" if index % 3 else "No",
                "Yes" if index % 4 else "No",
                "Yes" if index % 5 else "No",
                "Yes" if index % 2 else "No",
                "Yes" if index % 3 else "No",
                contracts[index % len(contracts)],
                "Yes" if index % 2 else "No",
                "Electronic check" if index % 2 else "Credit card (automatic)",
                monthly_charges,
                " " if tenure == 0 else f"{monthly_charges * tenure:.2f}",
                "Yes" if index % 4 == 0 else "No",
            )
        )
    return tuple(rows)


SYNTHETIC_TELCO_ROWS = synthetic_telco_rows()


def write_synthetic_raw_csv(output_path: Path) -> int:
    """Write and validate the 21-column RAW contract."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SYNTHETIC_TELCO_ROWS, columns=EXPECTED_COLUMNS).to_csv(
        path, index=False
    )
    return len(validate_telco_dataset(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    count = write_synthetic_raw_csv(arguments.output)
    print(f"Synthetic RAW Telco fixture created: {arguments.output} ({count} rows)")


if __name__ == "__main__":
    main()
