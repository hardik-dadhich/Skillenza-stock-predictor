#!/usr/bin/env python
# coding: utf-8

# In[11]:



import azureml.core
print(azureml.core.VERSION)
get_ipython().system('pip3 install azureml-train-automl')


# In[ ]:


from azureml.core import Workspace

subscription_id = '<subscription-id>'
resource_group  = '<resource-group>'
workspace_name  = '<workspace-name>'

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')


# In[3]:


import os
import urllib
import shutil
import azureml

from azureml.core import Experiment
from azureml.core import Workspace, Run

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


# In[ ]:


import os


# In[ ]:


import urllib


# In[ ]:


import shutil


# In[6]:


import azureml


# In[7]:


from azureml.core import Experiment


# In[8]:


from azureml.core import Workspace, Run


# In[9]:


from azureml.core.compute import ComputeTarget, AmlCompute


# In[10]:



from azureml.core.compute_target import ComputeTargetException


# In[12]:


ws = Workspace.from_config()


# In[17]:



from azureml.core import Workspace

# Create the workspace using the specified parameters
ws = Workspace.create(name = "testing",
                      subscription_id = "c3ef02ec-8e19-415f-a0d7-b562b6b78b11",
                      resource_group = "default", 
                      location = "westus",
                      create_resource_group = True,
                      exist_ok = True)
ws.get_details()

# write the details of the workspace to a configuration file to the notebook library
ws.write_config()


# In[ ]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
from azureml.core import Workspace

# Create the workspace using the specified parameters
ws = Workspace.create(name = "testing",
                      subscription_id = "c3ef02ec-8e19-415f-a0d7-b562b6b78b11",
                      resource_group = "default", 
                      location = "westus",
                      create_resource_group = True,
                      exist_ok = True)
ws.get_details()

# write the details of the workspace to a configuration file to the notebook library
ws.write_config()
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpu-cluster")
except ComputeTargetException:
    print("Creating new cpu-cluster")
    
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)

    # Create the cluster with the specified name and configuration
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    
    # Wait for the cluster to complete, show the output log
    cpu_cluster.wait_for_completion(show_output=True)


# In[12]:


pip install matplotlib


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[27]:


dataset = pd.read_csv('Downloads/005930.KS.csv')


# In[77]:


dataset


# In[17]:


pip install sklearn


# In[ ]:





# In[78]:


# Taking care of missing data
dataset = dataset.drop(columns="AdjClose")
vol=dataset.iloc[:,-1].values
close=dataset.iloc[:,-1].values
dataset = dataset.drop(columns="Volume")
dataset = dataset.drop(columns="Close")
dataset['volume']=vol
dataset['Close']=close


# In[79]:


dataset


# In[80]:


dataset = dataset[dataset.Date != 'null']


# In[81]:


type(dataset)


# In[82]:


dataset


# In[ ]:





# In[76]:


dataset.isnull()


# In[95]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values


# In[96]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values =np.nan, strategy = 'mean',verbose=0)

imputer = imputer.fit(X[:, 1:5])

X[:, 1:5] = imputer.transform(X[:, 1:5])


# In[97]:


X=pd.DataFrame(X)


# In[98]:


X.isnull().sum()


# In[99]:


X


# In[100]:


from azureml.core.workspace import Workspace


# In[101]:


from azureml.core.experiment import Experiment


# In[107]:



import azureml.core
from azureml.core import Experiment, Workspace

# Check core SDK version number
print("This notebook was created using version 1.0.2 of the Azure ML SDK")
print("You are currently using version", azureml.core.VERSION, "of the Azure ML SDK")
print("")


ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')


# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
data = {
    "train":{"X": X_train, "y": y_train},        
    "test":{"X": X_test, "y": y_test}
}


# In[2]:


# Get an experiment object from Azure Machine Learning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

experiment = Experiment(workspace=ws, name="train-within-notebook")
run = experiment.start_logging(snapshot_directory=None)
# Create a run object in the experiment

# Log the algorithm parameter alpha to the run
run.log('alpha', 0.03)

# Create, fit, and test the scikit-learn Ridge regression model
regression_model = Ridge(alpha=0.03)
regression_model.fit(data['train']['X'], data['train']['y'])
preds = regression_model.predict(data['test']['X'])

# Output the Mean Squared Error to the notebook and to the run
print('Mean Squared Error is', mean_squared_error(data['test']['y'], preds))
run.log('mse', mean_squared_error(data['test']['y'], preds))

# Save the model to the outputs directory for capture
model_file_name = 'outputs/model.pkl'

joblib.dump(value = regression_model, filename = model_file_name)

# upload the model file explicitly into artifacts 
run.upload_file(name = model_file_name, path_or_stream = model_file_name)

# Complete the run
run.complete()


# In[121]:


from azureml.core import Workspace, Datastore


# In[122]:


# Default datastore 
def_data_store = ws.get_default_datastore()

# Get the blob storage associated with the workspace
def_blob_store = Datastore(ws, "workspaceblobstore")

# Get file storage associated with the workspace
def_file_store = Datastore(ws, "workspacefilestore")


# In[123]:


def_blob_store.upload_files(
    ["Downloads/005930.KS.csv"],
    target_path="xyz",
    overwrite=True)


# In[ ]:


def_blob_store.upload_files(
    ["./data/20news.pkl"],
    target_path="20newsgroups",
    overwrite=True)


# In[129]:


datastore = Datastore.register_azure_blob_container(workspace=ws, 
                                                    datastore_name='your', 
                                                    account_name='lokeshdata',
                                                    container_name='yourc',
                                                    account_key='L6ot0h04xROx/83/W6AymEAR7f66KuhVLKxOCm1SvcMAg70yrJv32mcY389mOoSPVyfRxuTYr3eSZpGF0WHPUg==',
                                                    subscription_id = "c3ef02ec-8e19-415f-a0d7-b562b6b78b11",
                      
                                                    create_if_not_exists=True)


# In[133]:


import azureml.data
from azureml.data.azure_storage_datastore import AzureFileDatastore, AzureBlobDatastore


# In[ ]:


from azureml.train.estimator import Estimator

script_params = {
    '--data_dir': datastore.path('Downloads').as_upload()
}

est = Estimator(source_directory='your code directory',
                entry_script='train.py',
                script_params=script_params,
                compute_target=compute_target
                )


# In[ ]:





# In[141]:


pip install azureml-sdk[notebooks,automl]


# In[3]:


get_ipython().system('pip3 install --upgrade azureml-sdk azureml-contrib-run')


# In[1]:


from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import logging

automl_config = AutoMLConfig(task='forecasting',
                             primary_metric='normalized_root_mean_squared_error',
                             iterations=10,
                             X=X_train,
                             y=y_train,
                             n_cross_validations=5,
                             enable_ensembling=False,
                             verbosity=logging.INFO,
                             **time_series_settings)

ws = Workspace.from_config()
experiment = Experiment(ws, "forecasting_example")
local_run = experiment.submit(automl_config, show_output=True)
best_run, fitted_model = local_run.get_output()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




