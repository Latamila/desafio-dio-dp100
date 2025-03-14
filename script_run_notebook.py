import azure

# Import the required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


# The workspace information from the previous experiment has been pre-filled for you.
subscription_id = "f2eaaace-175c-42f9-9bf9-fce1981a6713"
resource_group = "rg-projetoum-dpcem"
workspace_name = "worksp-projetoum-dpcem"

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
print(ml_client.workspace_name, workspace.resource_group, workspace.location, ml_client.connections._subscription_id, sep = '\n')

import os
import shutil

project_folder = os.path.join(".", 'code_folder')
os.makedirs(project_folder, exist_ok=True)
shutil.copy('script.py', project_folder)

from azure.ai.ml.entities import AmlCompute

# Choose a name for your CPU cluster
cluster_name = "cluster-projetoum-dpcem"

# Verify that cluster does not exist already
try:
    cluster = ml_client.compute.get(cluster_name)
    print('Found existing cluster, use it.')
except Exception:
    compute = AmlCompute(name=cluster_name, size='Standard_DS11_v2',
                         max_instances=4)
    cluster = ml_client.compute.begin_create_or_update(compute)



# To test the script with an environment referenced by a custom yaml file, uncomment the following lines and replace the `conda_file` value with the path to the yaml file.
# Set the value of `environment` in the `command` job below to `env`.

#env = Environment(
#        name="automl-tabular-env",
#        description="environment for automl inference",
#        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20210727.v1",
#        conda_file="conda.yaml",
#)


from azure.ai.ml import command, Input

# To test with new training / validation datasets, replace the default dataset id(s)/uri(s) taken from parent run below
command_str = 'python script.py --training_dataset_uri azureml://locations/eastus/workspaces/7bde530e-5562-40ec-a3b4-557721304d8e/data/baldeAcaiVendas/versions/1'
command_job = command(
    code=project_folder,
    command=command_str,
    tags=dict(automl_child_run_id='sleepy_mango_zrchp2f32n_8'),
    environment='AzureML-ai-ml-automl:15',
    compute='cluster-projetoum-dpcem',
    experiment_name='acai-designer')
 
returned_job = ml_client.create_or_update(command_job)
returned_job.studio_url



%pip install azureml-mlflow
%pip install mlflow



import mlflow

# Obtain the tracking URL from MLClient
MLFLOW_TRACKING_URI = ml_client.workspaces.get(
    name=ml_client.workspace_name
).mlflow_tracking_uri

# Set the MLFLOW TRACKING URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Retrieve the metrics logged to the run.
from mlflow.tracking.client import MlflowClient

# Initialize MLFlow client
mlflow_client = MlflowClient()
mlflow_run = mlflow_client.get_run(returned_job.name)
mlflow_run.data.metrics


import os

# Create local folder
#local_dir = "./artifact_downloads"
#if not os.path.exists(local_dir):
#    os.mkdir(local_dir)
#    local_path = mlflow_client.download_artifacts(
#        mlflow_run.info.run_id, "outputs", local_dir)
#print("Artifacts downloaded in: {}".format(local_path))
#print("Artifacts: {}".format(os.listdir(local_path)))
