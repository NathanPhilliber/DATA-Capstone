import math
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationReport():
    def __init__(self, model, spectra_preprocessor, labels=None):
        self.model = model
        _, _, self.X_test, self.y_test = spectra_preprocessor.transform(encoded=True)
        self.test_spectra_loader = spectra_preprocessor.test_spectra_loader
        self.peak_locs = self.test_spectra_loader.get_peak_locations()
        self.labels = labels
        if self.labels is None:
            self.labels = [i + 1 for i in range(self.y_test.shape[1])]
        self.probs = self.model.keras_model.predict_proba(self.X_test)
        self.preds = self.probs.argmax(axis=1) + 1
        self.y_true_num = self.y_test.argmax(axis=1) + 1

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
                     lw=lw, label='%d peaks ROC curve (area = %0.2f)' % (self.labels[i], roc_auc))
            plt.legend(loc='lower right')
        return plt

    def plot_mean_pred_prob(self, num_peaks, ax=None):
        subset_probs = self.probs[self.y_test.argmax(axis=1) + 1 == num_peaks].mean(axis=0)
        return self.plot_pred_prob(subset_probs, num_peaks=num_peaks, ax=ax,
                                   title_extension=f'Mean predicted probabilities when number peaks = {num_peaks}',
                                   y_extension='Mean ')

        # return sns.barplot(x = self.labels, y = subset_probs, palette=color_palette, ax=ax).set_title(f'Mean predicted probabilities when number peaks = {num_peaks}')

    def plot_mean_pred_probs(self):
        fig, axes = plt.subplots(len(self.labels), 1, figsize=(7, len(self.labels) * 5))
        for i, np in enumerate(self.labels):
            self.plot_mean_pred_prob(np, axes[i])
        plt.subplots_adjust(hspace=0.3)

    def plot_pred_prob(self, peak_probs, num_peaks=None, ax=None, title_extension=None, y_extension=''):
        title = title_extension
        if title_extension == None:
            title = f'Predicted Probability for Num Peaks'

        color_palette = ['grey' if label != num_peaks else 'red' for label in self.labels]
        bar_plot = sns.barplot(x=[str(i) for i in self.labels], y=peak_probs, palette=color_palette, ax=ax)
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
            sample_peak_locs = self.peak_locs[sample_idx]
            self.plot_pred_prob(sample_probs, num_peaks, ax=axes[i][0], title_extension=title_extension)

            for l in range(num_channels):
                self.test_spectra_loader.spectra[sample_idx].plot_channel(l, ax=axes[i][l + 1])

        plt.subplots_adjust(hspace=0.4)

    def plot_predicted_probs_misclassified(self, num_peaks, num_channels, num_examples):
        subset_num_peaks_idx = np.where((self.y_true_num == num_peaks) & (self.preds != num_peaks))[0]
        sample_idx = np.random.choice(subset_num_peaks_idx, num_examples)
        self.plot_predicted_probs(sample_idx, num_channels, num_peaks,
                                  f'Misclassified Predicted Probabilities, True Num Peaks: {num_peaks}')
        plt.show()

    def plot_predicted_probs_misclassified_per_peak(self, num_channels, num_examples):
        for np in self.labels:
            try:
                self.plot_predicted_probs_misclassified(np, num_channels, num_examples)
            except:
                print(f'No misclassified {np} peaks.')


def complete_evaluation(evaluation_report, num_channels_to_show, num_examples_per_peak):
    return evaluation_report.plot_roc_curves()
    #evaluation_report.plot_mean_pred_probs()
    #evaluation_report.plot_predicted_probs_misclassified_per_peak(num_channels_to_show, num_examples_per_peak)
