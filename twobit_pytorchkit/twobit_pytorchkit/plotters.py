import matplotlib.pyplot as plt
import numpy as np
from training.records import TrainingRecord

class TrainingPlotter:
    def __init__(self, training_record: TrainingRecord):
        self.training_record = training_record

    def plot_training_history(self):
        if not self.training_record.history:
            print("No data to plot.")
            return

        epochs = range(1, len(self.training_record.history) + 1)

        # Extracting training and validation loss and accuracy using new methods
        train_losses = self.training_record.get_train_loss_history()
        val_losses = self.training_record.get_val_loss_history()
        train_accuracies = self.training_record.get_train_accuracy_history()
        val_accuracies = self.training_record.get_val_accuracy_history()

        # Plotting training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        if self.training_record.early_stopping_epoch:
            es_val_loss = val_losses[self.training_record.early_stopping_epoch - 1]
            plt.axvline(x=self.training_record.early_stopping_epoch, color='gray', linestyle='--', label='Early Stopping')
            plt.axhline(y=es_val_loss, color='gray', linestyle='--')
            plt.scatter(self.training_record.early_stopping_epoch, es_val_loss, color='yellow', edgecolor='black', zorder=5)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        if self.training_record.early_stopping_epoch:
            plt.axvline(x=self.training_record.early_stopping_epoch, color='gray', linestyle='--', label='Early Stopping')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_gradient_norms(self):
        
        if not self.training_record.history:
            print("No data to plot.")
            return

        epochs = range(1, len(self.training_record.history) + 1)
        gradient_norms = self.training_record.get_gradient_norms_history()
        
        plt.figure(figsize=(10, 6))
        plt.plot(gradient_norms, label='Gradient Norms')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.legend()
        plt.show()

    def plot_learning_rate(self):
            
        if not self.training_record.history:
            print("No data to plot.")
            return

        epochs = range(1, len(self.training_record.history) + 1)
        lrs = self.training_record.get_learning_rate_history()
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate During Training')
        plt.legend()
        plt.show()