import math
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
from utils import *


def format_classification_report(classification_report, peak_labels):
    return {f'{p}_test_{metric}': metric_val for p in peak_labels for metric, metric_val in
            classification_report[p].items()}


def get_classification_report(test_formatted, preds_formatted, peak_labels):
    classif_report = classification_report(test_formatted, preds_formatted, target_names=peak_labels, output_dict=True)
    formatted = format_classification_report(classif_report, peak_labels)
    return formatted


class EvaluationReport:
    def __init__(self, model, spectra_preprocessor, labels=None):
        self.model = model
        self.X_test, self.y_test = spectra_preprocessor.transform_test()
        self.test_spectra_loader = spectra_preprocessor.test_spectra_loader
        self.peak_locs = self.test_spectra_loader.get_peak_locations()
        self.labels = labels
        if self.labels is None:
            self.labels = [i + 1 for i in range(self.y_test.shape[1])]
        self.numeric_labels = [i + 1 for i in range(self.y_test.shape[1])]
        self.probs = self.model.keras_model.predict_proba(self.X_test)
        self.preds = self.probs.argmax(axis=1) + 1
        self.y_true_num = self.y_test.argmax(axis=1) + 1

    def get_eval_classification_report(self):
        return get_classification_report(self.y_true_num, self.preds, self.labels)

    def plot_roc_curves(self, figsize=(9, 7)):
        plt.figure(figsize=figsize)
        lw = 2.5
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Number of Peaks')
        for i in range(len(self.labels)):
            probs_i = self.probs[:, i]
            y_true_i = self.y_test[:, i]
            fpr, tpr, _ = roc_curve(y_true_i, probs_i)
            roc_auc = roc_auc_score(y_true_i, probs_i)
            plt.plot(fpr, tpr,
                     lw=lw, label=f'{self.numeric_labels[i]} peaks ROC curve (area = %0.2f)' % roc_auc)
            plt.legend(loc='lower right')
        return plt

    def plot_mean_pred_prob(self, num_peaks, ax=None):
        subset_probs = self.probs[self.y_test.argmax(axis=1) + 1 == num_peaks].mean(axis=0)
        return self.plot_pred_prob(subset_probs, num_peaks=num_peaks, ax=ax,
                                   title_extension=f'Mean predicted probabilities when number peaks = {num_peaks}',
                                   y_extension='Mean ')

        # return sns.barplot(x = self.labels, y = subset_probs, palette=color_palette, ax=ax).set_title(f'Mean predicted probabilities when number peaks = {num_peaks}')

    def plot_mean_pred_probs(self):
        fig, axes = plt.subplots(len(self.numeric_labels), 1, figsize=(7, len(self.numeric_labels) * 5))
        for i, num_peak in enumerate(self.numeric_labels):
            self.plot_mean_pred_prob(num_peak, axes[i])
        plt.subplots_adjust(hspace=0.3)
        return plt

    def plot_pred_prob(self, peak_probs, num_peaks=None, ax=None, title_extension=None, y_extension=''):
        title = title_extension
        if title_extension == None:
            title = f'Predicted Probability for Num Peaks'

        color_palette = ['grey' if label != num_peaks else 'red' for label in self.numeric_labels]
        bar_plot = sns.barplot(x=self.labels, y=peak_probs, palette=color_palette, ax=ax)
        bar_plot.set_title(title)
        bar_plot.set_xlabel('Num Peaks')
        bar_plot.set_ylabel(f'{y_extension} Probability')
        return bar_plot

    def plot_predicted_probs(self, indices, num_channels, num_peaks, title_extension):
        num_plots = len(indices)
        if num_plots == 0:
            num_plots = self.probs.shape[0]

        fig, axes = plt.subplots(math.ceil(num_plots), num_channels + 1, figsize=(num_channels * 7, num_plots * 5))
        if len(indices) == 1: axes = [axes]

        for i in range(len(indices)):
            sample_idx = indices[i]
            sample_probs = self.probs[sample_idx]
            self.plot_pred_prob(sample_probs, num_peaks, ax=axes[i][0], title_extension=title_extension)

            for l in range(num_channels):
                self.test_spectra_loader.spectra[sample_idx].plot_channel(l, ax=axes[i][l + 1])

        plt.subplots_adjust(hspace=0.4)
        return plt

    def plot_predicted_probs_misclassified(self, num_peaks, num_channels, num_examples):
        subset_num_peaks_idx = np.where((self.y_true_num == num_peaks) & (self.preds != num_peaks))[0]
        sample_idx = np.random.choice(subset_num_peaks_idx, num_examples)
        return self.plot_predicted_probs(sample_idx, num_channels, num_peaks,
                                  f'Misclassified Predicted Probabilities, True Num Peaks: {num_peaks}')

    def plot_predicted_probs_misclassified_per_peak(self, num_channels, num_examples, directory, file_extension=None):
        for num_peaks in self.numeric_labels:
            try:
                misclass = self.plot_predicted_probs_misclassified(num_peaks, num_channels, num_examples)
                misclass.savefig(os.path.join(directory, f'misclassified_{num_peaks}-{file_extension}.png'))
            except:
                print(f'No misclassified {num_peaks} peaks.')


def complete_evaluation(evaluation_report, num_channels_to_show, num_examples_per_peak, directory, file_extension=None):
    roc_curve_plot = evaluation_report.plot_roc_curves()
    roc_curve_plot.savefig(os.path.join(directory, f'roc_curve-{file_extension}.png'))
    mean_preds = evaluation_report.plot_mean_pred_probs()
    mean_preds.savefig(os.path.join(directory, f'mean_preds-{file_extension}.png'))
    evaluation_report.plot_predicted_probs_misclassified_per_peak(num_channels_to_show, num_examples_per_peak,
                                                                  directory, file_extension)
