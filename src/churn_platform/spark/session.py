"""Reusable local SparkSession configuration with Delta Lake support."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

DEFAULT_APP_NAME = "mlops-customer-churn-platform"
DEFAULT_MASTER = "local[2]"
DEFAULT_WAREHOUSE_PATH = Path("spark-warehouse")


def _configure_homebrew_java() -> None:
    """Make a keg-only Homebrew Java 17 visible when JAVA_HOME is unset."""
    if os.environ.get("JAVA_HOME") or sys.platform != "darwin":
        return

    candidates = (
        Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
        Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    )
    for java_home in candidates:
        if (java_home / "bin" / "java").is_file():
            os.environ["JAVA_HOME"] = str(java_home)
            os.environ["PATH"] = f"{java_home / 'bin'}{os.pathsep}{os.environ['PATH']}"
            return


def create_spark_session(
    app_name: str = DEFAULT_APP_NAME,
    *,
    master: str = DEFAULT_MASTER,
    warehouse_path: Path = DEFAULT_WAREHOUSE_PATH,
) -> SparkSession:
    """Create a laptop-friendly local SparkSession configured for Delta Lake."""
    _configure_homebrew_java()
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    warehouse_uri = Path(warehouse_path).resolve().as_uri()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.warehouse.dir", warehouse_uri)
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
