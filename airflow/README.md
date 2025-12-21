# Airflow stack

Docker Compose stack for running Apache Airflow to schedule the MPDB crawler, write text data into PostgreSQL, and persist downloaded photos on disk.

## Layout
- DAGS: `airflow/dags` mounted to `/opt/airflow/dags`
- Logs/plugins: `airflow/logs`, `airflow/plugins`
- Shared files (photo output, temp data): `airflow/files` mounted to `/opt/airflow/files` with photos under `airflow/files/photos`
- Crawler source mounted to `/opt/crawler`
- Airflow metadata DB: `airflow-db` (PostgreSQL)
- Target data DB: `plant-db` (PostgreSQL) exposed in Airflow as connection `plant_db`

## Quick start
```bash
# Optional: customize creds/UID/Fernet key before running (see Environment below)
docker compose -f docker-compose.airflow.yaml up airflow-init
docker compose -f docker-compose.airflow.yaml up -d
```

- Airflow UI: http://localhost:8080 (`AIRFLOW_USERNAME`/`AIRFLOW_PASSWORD`, defaults `airflow`/`airflow`)
- Photos land on the host at `airflow/files/photos` (visible to Airflow at `/opt/airflow/files/photos`)
- Text/structured data can be stored in `plant-db` via the Airflow connection ID `plant_db`

## Environment knobs
- `AIRFLOW_UID` (default `50000`): align container file ownership with your host user (`AIRFLOW_UID=$(id -u)`)
- `AIRFLOW_FERNET_KEY`: set a new 32-byte Fernet key for encrypting Airflow secrets
- `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD`: admin user created during `airflow-init`
- `AIRFLOW_CONN_PLANT_DB`: override the Postgres URI for the target data database if you need to point elsewhere

To regenerate a Fernet key:
```bash
python3 - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

## Day-to-day
- Stop: `docker compose -f docker-compose.airflow.yaml down`
- Recreate DBs: `docker compose -f docker-compose.airflow.yaml down -v` (removes Airflow + plant databases)
- Tail logs: `docker compose -f docker-compose.airflow.yaml logs -f airflow-scheduler`
