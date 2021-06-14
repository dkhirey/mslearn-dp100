ws = Workspace.from_config()
experiment = Experiment(ws,'mslearn-diabetes')
run = experiment.start_logging()

run.log('No of rows',row_count)
run.log_image(name='label distribution', plot=fig)
run.log_list('pregnancy categories', pregnancies)
run.log_row(col, stat=keys[index], value = values[index])
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')
run.get_metrics()
run.get_file_names()
run.download_files(prefix='outputs', output_directory=download_folder)
run.get_details_with_logs()

#%%writefile $folder_name/diabetes_experiment.py
run = Run.get_context()
script_config = ScriptRunConfig(source_directory=experiment_folder, 
                      script='diabetes_experiment.py')
run = experiment.submit(config=script_config)
run.wait_for_completion()
run.get_all_logs(destination=log_folder)
diabetes_experiment = ws.experiments['mslearn-diabetes']
diabetes_experiment.get_runs()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment.name)
mlflow.start_run()

experiment.get_portal_url()
mlflow_env = Environment("mlflow-env")
packages = CondaDependencies.create(conda_packages=['pandas','pip'],
                                    pip_packages=['mlflow','azureml-mlflow'])
mlflow_env.python.conda_dependencies = packages
script_mlflow = ScriptRunConfig(source_directory=experiment_folder,
                                script='mlflow_diabetes.py',
                                environment=mlflow_env) 

run.register_model(model_path='outputs/diabetes_model.pkl', 		   model_name='diabetes_model',
                   tags={'Training context':'Script'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': 		   run.get_metrics()['Accuracy']})

parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

script_config = ScriptRunConfig(source_directory=training_folder,
                                script='diabetes_training.py',
                                arguments = ['--reg_rate', 0.1],
                                environment=sklearn_env) 

default_ds = ws.get_default_datastore()
azureml_globaldatasets - Default = False
workspacefilestore - Default = False
workspaceblobstore - Default = True

blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789…')

default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'],
                       target_path='diabetes-data/', 
                       overwrite=True,
                       show_progress=True)

tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))
tab_data_set.take(20).to_pandas_dataframe()
file_data_set = Dataset.File.from_files(path=(default_ds, 'diabetes-data/*.csv'))

tab_data_set = tab_data_set.register(workspace=ws, 
                                        name='diabetes dataset',
                                        description='diabetes data',
                                        tags = {'format':'CSV'},
                                        create_new_version=True)

dataset = Dataset.get_by_name(ws, dataset_name)
dataset_v1 = Dataset.get_by_name(ws, 'diabetes dataset', version = 1)

diabetes = run.input_datasets['training_data'].to_pandas_dataframe()
dataset = Dataset.get_by_id(ws, id=args.training_dataset_id)

packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults','azureml-				    dataprep[pandas]'])
diabetes_ds = ws.datasets.get("diabetes dataset")

script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='diabetes_training.py',
                              arguments = ['--regularization', 0.1,
                                           '--input-data',diabetes_ds.as_named_inpu('training_data')],
                              environment=sklearn_env)


diabetes_env = Environment("diabetes-experiment-env")
diabetes_env.python.user_managed_dependencies = False
diabetes_env.docker.enabled = True
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib','pandas','pip'],
                                             pip_packages=['azureml-sdk','pyarrow'])
diabetes_env.python.conda_dependencies = diabetes_packages


diabetes_env.register(workspace=ws)
registered_env = Environment.get(ws, 'diabetes-experiment-env')
envs = Environment.list(workspace=ws)
envs[env].python.conda_dependencies.serialize_to_string()

training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
training_cluster.wait_for_completion(show_output=True)

script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='diabetes_training.py',
                                arguments = ['--input-data', diabetes_ds.as_named_input('training_data')],
                                environment=registered_env,
                                compute_target=cluster_name) 

cluster_state = training_cluster.get_status()


pipeline_run_config = RunConfiguration()
pipeline_run_config.target = pipeline_cluster
pipeline_run_config.environment = registered_env

prepped_data_folder = PipelineData("prepped_data_folder", datastore=ws.get_default_datastore())

prep_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_diabetes.py",
                                arguments = ['--input-data', diabetes_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data_folder],
                                outputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

train_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_diabetes.py",
                                arguments = ['--training-folder', prepped_data_folder],
                                inputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)

published_pipeline = pipeline_run.publish_pipeline(
    name="diabetes-training-pipeline", description="Trains diabetes model", version="1.0")
rest_endpoint = published_pipeline.endpoint


interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)

recurrence = ScheduleRecurrence(frequency="Week", interval=1, week_days=["Monday"], time_of_day="00:00")
weekly_schedule = Schedule.create(ws, name="weekly-diabetes-training", 
                                  description="Based on time",
                                  pipeline_id=published_pipeline.id, 
                                  experiment_name='mslearn-diabetes-pipeline', 
                                  recurrence=recurrence)
schedules = Schedule.list(ws)

model_path = Model.get_model_path('diabetes_model')
model = joblib.load(model_path)

myenv = CondaDependencies()
myenv.add_conda_package('scikit-learn')

inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service_name = "diabetes-service"
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)
service.wait_for_deployment(True)
service.get_logs()
webservice_name in ws.webservices
predictions = service.run(input_data = input_json)

endpoint = service.scoring_uri
headers = { 'Content-Type':'application/json' }
predictions = requests.post(endpoint, input_json, headers = headers)


parallel_run_config = ParallelRunConfig(
    source_directory=experiment_folder,
    entry_script="batch_diabetes.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=inference_cluster,
    node_count=2)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_batch')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)


params = GridParameterSampling(
    {
        # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)
    }
)

hyperdrive = HyperDriveConfig(run_config=script_config, 
                          hyperparameter_sampling=params, 
                          policy=None, # No early stopping policy
                          primary_metric_name='AUC', # Find the highest AUC metric
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=6, # Restict the experiment to 6 iterations
                          max_concurrent_runs=2) # Run up to 2 iterations in parallel
						  
						  
automl_run_config = RunConfiguration(framework='python')
automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             primary_metric = 'AUC_weighted',
                             compute_target=aml_compute,
                             training_data = train_dataset,
                             validation_data = test_dataset,
                             label_column_name='Label',
                             featurization='auto',
                             iterations=12,
                             max_concurrent_iterations=4)

featurization_config = FeaturizationConfig()
featurization_config.blocked_transformers = ['LabelEncoder']
featurization_config.drop_columns = ['aspiration', 'stroke']
featurization_config.add_column_purpose('engine-size', 'Numeric')
featurization_config.add_column_purpose('body-style', 'CategoricalHash')
#default strategy mean, add transformer param for for 3 columns
featurization_config.add_transformer_params('Imputer', ['engine-size'], {"strategy": "median"})
featurization_config.add_transformer_params('Imputer', ['city-mpg'], {"strategy": "median"})
featurization_config.add_transformer_params('Imputer', ['bore'], {"strategy": "most_frequent"})
featurization_config.add_transformer_params('HashOneHotEncoder', [], {"number_of_bits": 3})

mim_explainer = MimicExplainer(model=loan_model,
                             initialization_examples=X_test,
                             explainable_model = DecisionTreeExplainableModel,
                             features=['loan_amount','income','age','marital_status'], 
                             classes=['reject', 'approve'])

tab_explainer = TabularExplainer(model=loan_model,
                             initialization_examples=X_test,
                             features=['loan_amount','income','age','marital_status'],
                             classes=['reject', 'approve'])

pfi_explainer = PFIExplainer(model = loan_model,
                             features=['loan_amount','income','age','marital_status'],
                             classes=['reject', 'approve'])

global_mim_explanation = mim_explainer.explain_global(X_train)
global_mim_feature_importance = global_mim_explanation.get_feature_importance_dict()

global_pfi_explanation = pfi_explainer.explain_global(X_train, y_train)
global_pfi_feature_importance = global_pfi_explanation.get_feature_importance_dict()

local_mim_explanation = mim_explainer.explain_local(X_test[0:5])
local_mim_features = local_mim_explanation.get_ranked_local_names()
local_mim_importance = local_mim_explanation.get_ranked_local_values()

# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

client = ExplanationClient.from_run_id(workspace=ws,
                                       experiment_name=experiment.experiment_name, 
                                       run_id=run.id)
explanation = client.download_model_explanation()
feature_importances = explanation.get_feature_importance_dict()

overall_selection_rate = selection_rate(y_test, y_hat) # Get selection rate from fairlearn

metrics = {'selection_rate': selection_rate,
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}
group_metrics = MetricFrame(metrics,
                             y_test, y_hat,
                             sensitive_features=S_test['Age'])

FairlearnDashboard(sensitive_features=S_test, 
                   sensitive_feature_names=['Age'],
                   y_true=y_test,
                   y_pred={"diabetes_model": diabetes_model.predict(X_test)})

sweep = GridSearch(DecisionTreeClassifier(),
                   constraints=EqualizedOdds(),
                   grid_size=20)
sweep.fit(X_train, y_train, sensitive_features=S_train.Age)
models = sweep.predictors_

dep_config = AciWebservice.deploy_configuration(cpu_cores = 1,
                                                memory_gb = 1,
                                                enable_app_insights=True)
												
traces
|where message == "STDOUT"
  and customDimensions.["Service Name"] = "my-svc"
| project  timestamp, customDimensions.Content

alert_email = AlertConfiguration('data_scientists@contoso.com')
monitor = DataDriftDetector.create_from_datasets(workspace=ws,
                                                 name='dataset-drift-detector',
                                                 baseline_data_set=train_ds,
                                                 target_data_set=new_data_ds,
                                                 compute_target='aml-cluster',
                                                 frequency='Week',
                                                 feature_list=['age','height', 'bmi'],
                                                 latency=24,
                                                 drift_threshold=.3,
                                                 alert_configuration=alert_email)
backfill = monitor.backfill( dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())



________________________________________________________________________________________________


from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

bostonDF = (spark.read
  .option("HEADER", True)
  .option("inferSchema", True)
  .csv("/mnt/training/bostonhousing/bostonhousing/bostonhousing.csv")
  .drop("_c0")
)

trainDF, testDF = bostonDF.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(inputCols=bostonDF.columns[:-1], outputCol="features")

lr = (LinearRegression()
  .setLabelCol("medv")
  .setFeaturesCol("features")
)

pipeline = Pipeline(stages = [assembler, lr])

print(lr.explainParams())

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
  .addGrid(lr.maxIter, [1, 10, 100])
  .addGrid(lr.fitIntercept, [True, False])
  .addGrid(lr.standardization, [True, False])
  .build()
)


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(
  labelCol = "medv", 
  predictionCol = "prediction"
)

cv = CrossValidator(
  estimator = pipeline,             # Estimator (individual model or pipeline)
  estimatorParamMaps = paramGrid,   # Grid of parameters to try (grid search)
  evaluator=evaluator,              # Evaluator
  numFolds = 3,                     # Set k to 3
  seed = 42                         # Seed to sure our results are the same if ran again
)

cvModel = cv.fit(trainDF)

bestModel = cvModel.bestModel


import mlflow.azureml

model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri, 
                                                      workspace=workspace,
                                                      model_name="model",
                                                      image_name="model",
                                                      description="Sklearn ElasticNet image for predicting diabetes progression",
                                                      synchronous=False)
model_image.wait_for_creation(show_output=True)

from azureml.core.webservice import AciWebservice, Webservice

dev_webservice_name = "diabetes-model"
dev_webservice_deployment_config = AciWebservice.deploy_configuration()
dev_webservice = Webservice.deploy_from_image(name=dev_webservice_name, image=model_image, deployment_config=dev_webservice_deployment_config, workspace=workspace)
dev_webservice.wait_for_deployment()

