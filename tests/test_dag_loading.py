from unittest import mock
from airflow.models import DagBag

def test_dag_loading():
    dag_bag = DagBag(dag_folder='../dags/DAG.py',)
    assert len(dag_bag.import_errors) == 0  # No import errors
    dag = dag_bag.get_dag('your_dag_id')
    assert dag is not None  # Ensure the DAG is loaded correctly
