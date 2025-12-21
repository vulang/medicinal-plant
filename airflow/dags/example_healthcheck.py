"""
Simple DAG to validate the Airflow stack boots correctly.
Runs a short Bash command once a day.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="example_healthcheck",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["healthcheck"],
) as dag:
    BashOperator(
        task_id="print_date",
        bash_command="date",
    )

