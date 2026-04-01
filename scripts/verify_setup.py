"""Verify that the ML Council environment is correctly set up."""
import sys


def check_imports():
    packages = [
        ("polars", "polars"),
        ("duckdb", "duckdb"),
        ("lightgbm", "lightgbm"),
        ("hmmlearn", "hmmlearn"),
        ("mapie", "mapie"),
        ("mlflow", "mlflow"),
        ("dagster", "dagster"),
        ("sklearn", "scikit-learn"),
        ("cvxpy", "cvxpy"),
        ("yfinance", "yfinance"),
    ]
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)
    return failed


def check_postgres():
    import psycopg2
    conn = psycopg2.connect(
        dbname="mlcouncil",
        user="mlcouncil",
        password="mlcouncil_dev",
        host="localhost",
        port=5432,
        connect_timeout=5,
    )
    conn.close()
    print("  [OK] PostgreSQL")


def check_minio():
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="mlcouncil",
        aws_secret_access_key="mlcouncil_secret",
    )
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    print(f"  [OK] MinIO — buckets: {buckets}")


def main():
    errors = []

    print("\n=== Package imports ===")
    failed = check_imports()
    if failed:
        errors.append(f"Missing packages: {failed}")

    print("\n=== PostgreSQL ===")
    try:
        check_postgres()
    except Exception as e:
        print(f"  [FAIL] PostgreSQL: {e}")
        errors.append(str(e))

    print("\n=== MinIO ===")
    try:
        check_minio()
    except Exception as e:
        print(f"  [FAIL] MinIO: {e}")
        errors.append(str(e))

    print()
    if errors:
        print(f"Setup has {len(errors)} issue(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Setup OK")


if __name__ == "__main__":
    main()
