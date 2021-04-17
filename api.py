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