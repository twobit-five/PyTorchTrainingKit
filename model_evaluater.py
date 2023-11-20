import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torch import nn, optim

class ModelEvaluator:
    """
    Class for evaluating a machine learning model.

    Args:
        model (torch.nn.Module): The machine learning model to evaluate.
        device (str, optional): The device to use for evaluation (default: 'cpu').
        class_names (list, optional): The names of the classes (default: None).

    Attributes:
        model (torch.nn.Module): The machine learning model being evaluated.
        device (str): The device used for evaluation.
        class_names (list): The names of the classes.
        true_labels (numpy.ndarray): The true labels of the data.
        predictions (numpy.ndarray): The predicted labels of the data.
        probabilities (numpy.ndarray): The predicted probabilities of the data.
        num_classes (int): The number of classes.

    Methods:
        evaluate(data_loader): Evaluates the model on the given data loader.
        plot_confusion_matrix(): Plots the confusion matrix.
        plot_roc_curve(): Plots the ROC curve.
    """

    def __init__(self, model: nn.Module, device: str='cpu', class_names=None):
        self.model = model.to(device)
        self.device = device
        self.true_labels = None
        self.predictions = None
        self.probabilities = None
        self.class_names = class_names


    def evaluate(self, data_loader: DataLoader):
        """
        Evaluates the model on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader to evaluate on.
        """
        self.model.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)

                _, predictions = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        self.true_labels = np.array(all_labels)
        self.predictions = np.array(all_predictions)
        self.probabilities = np.array(all_probabilities)


    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix.
        Raises:
            ValueError: If the model has not been evaluated yet.
        """
        if self.true_labels is None or self.predictions is None:
            raise ValueError("Model not evaluated yet.")
        
        # Ensure all classes are represented in the confusion matrix
        unique_labels = np.unique(np.concatenate((self.true_labels, self.predictions)))
        cm = confusion_matrix(self.true_labels, self.predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self):
        """
        Plots the ROC curve.
        Raises:
            ValueError: If the model has not been evaluated yet.
        """

        print("True labels shape:", self.true_labels.shape)
        print("Probabilities shape:", self.probabilities.shape)

        num_classes = len(np.unique(self.true_labels))
        all_targets_binarized = label_binarize(self.true_labels, classes=range(num_classes))

        print("Binarized labels shape:", all_targets_binarized.shape)
        print("Flattened binarized labels shape:", all_targets_binarized.ravel().shape)
        print("Flattened probabilities shape:", self.probabilities.ravel().shape)

        if self.true_labels is None or self.probabilities is None:
            raise ValueError("Model not evaluated yet.")
        
        # Dynamically determine the number of classes
        num_classes = len(np.unique(self.true_labels))

        # Binarize the labels for all classes
        all_targets_binarized = label_binarize(self.true_labels, classes=range(num_classes))

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(all_targets_binarized.ravel(), self.probabilities.ravel())
        roc_auc = auc(fpr, tpr)

        # Plot micro-average ROC curve
        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr, tpr, color='deeppink', linestyle=':', linewidth=4,
                 label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc))

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-Average ROC for Multi-Class')
        plt.legend(loc="lower right")
        plt.show()