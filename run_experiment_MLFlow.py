#va executando aos poucos no ambiente jupyter lab ou similar. 


data.sample(50).to_csv('sample.csv', index=False, header=True)
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')

run.complete()

%%writefile $folder_name/acai_experiment.py

from azureml.core import Run
import pandas as pd
import os 

run = Run.get_context()

data = pd.read_csv('dadosExperimentos.csv')

row_count = (len(data))
run.log('observations', row_count)
print('Analyzing {} rows de data'.format(row_count))

vendas_counts = data['baldesAcaiVendas'].value_counts()
print(vendas_counts)
for k, v in vendas_counts.items():
  run.log('Label:' + str(k), v)

os.makedirs('outputs', exist_ok=True)
data.sample(50).to_csv('outputs/testeAcai.csv', index=False, header=True)

run.complete()


import os 
import sys
from azureml.core import Experiment, ScriptRunConfig
from azureml.widgets import RunDetails

script_config = ScripRunConfig(source_directory=experiment_folder,
                               script='acai_experiment.py')

experiment = Experiment(workspace=ws, name='projetoum-dpcem')
run = experiment.submit(config=script_config)
run.wait_for_completion()

metrics = run.get_metrics()
for key in metrics.key():
  print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
  print(file)

from azureml.core import Experiment, Run

acai_experiment = ws.experiments['projetoum-dpcem']
for logged_run in acai_experiment.get_runs():
  print('RUN ID: ', logged_run.id)
  metrics = logged_run.get_metrics()
  for key in metrics.keys():
    print('-', key, metrics.get(key))

!pip install --upgrade mlflow azureml-mlflow
import mlflow
import pandas as pd
from azureml.core import Experiment


mlflow.set_tracking)uri(ws.get_mlflow_tracking_uri())

experiment = Experiment(workspace=ws, name='projetoum-dpcem-mlflow')
mlflow.set_experiment(experiment.name)

#começa o experimento mlflow
with mlflow.start_run(run_name="hello-world-example") as run:
    # Your code
    print("Loading Data...")
    data = pd.read_csv('dadosVendasAcai.csv')
    
    X,y = data[['temperaturaEmGraus']].values, data[['baldesAcaiVendas']].values
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)
    
    reg = 0.01

    print('Training a logistic regression model with regularization rate of', reg)
    model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

    #calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    #calculate Auc

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print('AUC: ' + str(auc))
    
    mlflow.end_run()


run = mlflow.get_run("<RUN_ID>")

metrics = run.data.metrics
params = run.data.params
tags = run.data.tags
print('Run complete')

print(metrics, params, tags) 

##############################################################

run = list(experiment.get_runs())[0]

print('\nMetrics: ')
metrics = run.get_metrics()
for key in metrics.keys():
  print(key, metrics.get(key))

experiment_url = experiment.get_portal_url()
print('See details at', experiment_url)
