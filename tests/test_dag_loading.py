from unittest import mock
from airflow.models import DagBag

def test_dag_loading():
    dag_bag = DagBag(dag_folder='../dags/DAG.py', include_examples=False)
    assert len(dag_bag.import_errors) == 0  # No import errors

