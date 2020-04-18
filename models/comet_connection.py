from utils import *
from comet_ml import Experiment, Optimizer, ExistingExperiment
import numpy as np
from models.networks.abstract_models.base_model import BaseModel


class CometConnection:

    def __init__(self, model):
        self.experiment = None
        self.model = model

    def serialize(self):
        params = dict()
        params["comet_exp_key"] = self.experiment.get_key()

        return params

    def load_comet_new(self, comet_name, dataset_config):
        self.experiment = Experiment(api_key=COMET_KEY, project_name=PROJECT_NAME)
        self.experiment.set_name(comet_name)
        self.log_data_attributes(dataset_config)

    def load_comet_continue(self, exp_key):
        self.experiment = ExistingExperiment(api_key=COMET_KEY, previous_experiment=exp_key)

    def log_data_attributes(self, dataset_config):
        if self.experiment is None:
            return

        for key, value in dataset_config.items():
            self.experiment.log_parameter("SPECTRUM_" + key, value)

    def log_imgs(self, dataset_name):
        if self.experiment is None:
            return

        try:
            imgs_dir = os.path.join(DATA_DIR, dataset_name, 'imgs')
            self.experiment.log_asset_folder(imgs_dir)
        except:
            print(f"No images found for dataset: {dataset_name}")

    def log_script(self, dataset_config):
        if self.experiment is None:
            return

        script_name = dataset_config['matlab_script']
        try:
            matlab_dir = os.path.join(GEN_DIR, script_name)
            self.experiment.log_asset(matlab_dir)
        except:
            print(f"Could not find {script_name} under {GEN_DIR}.")

    def format_classification_report(self, classification_report):
        return {f'{k}_test_{metric}': metric_val for k, v in classification_report.items() for
                metric, metric_val in v.items()}

    def get_classification_report(self, y_test, preds):
        preds_formatted = np.argmax(preds, axis=1)
        test_formatted = np.argmax(y_test, axis=1)
        peak_labels = [f"n_peaks_{1 + num_peak}" for num_peak in range(y_test.shape[1])]
        classif_report = classification_report(test_formatted, preds_formatted, target_names=peak_labels,
                                               output_dict=True)
        classif_report_str = classification_report(test_formatted, preds_formatted, target_names=peak_labels)

        if self.experiment is not None:
            formatted = self.format_classification_report(classif_report)
            self.experiment.log_metrics(formatted)
            self.experiment.log_text(classif_report_str)

        return classif_report