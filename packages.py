import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Run
from azureml.widgets import RunDetails
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Model
import argparse
from azureml.core import Dataset
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core.run import PipelineRun
from azureml.pipeline.core import ScheduleRecurrence, Schedule
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
from interpret.ext.blackbox import TabularExplainer
from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import DecisionTreeExplainableModel
from interpret.ext.blackbox import PFIExplainer

from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from fairlearn.widget import FairlearnDashboard
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id
from fairlearn.reductions import GridSearch, EqualizedOdds

from azureml.datadrift import DataDriftDetector


azureml-sdk,
azureml-widgets,
azureml-interpret,
azureml-contrib-interpret,
azureml-explain-model,
azureml-contrib-fairness
fairlearn
azureml-datadrift

