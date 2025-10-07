from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
with DAG(dag_id="demo", start_date=datetime(2025, 6, 10), schedule="0 0 * * *") as dag:
    # Tasks are represented as operators
    fortune = BashOperator(task_id="fortune", bash_command="/usr/games/fortune")

    @task()
    def airflow():
        print("Airflow run of fortune cron job daily at midnight")

    # Set dependencies between tasks
    fortune >> airflow()
