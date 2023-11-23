import matplotlib.pyplot as plt
import numpy as np

#TODO should also enhance to track learning rate in the future
#TODO also should eventually move plotting functionality to a separate class.
class TrainingRecord:
    """
    A class to store training metrics for each epoch and provide methods to update, retrieve, plot, and save the metrics.
    """
    def __init__(self):
        # Initialize a list to store metrics for each epoch
        self.history = []
        self.early_stopping_epoch = None
        self.current_epoch_data = {}
        
    def update_train_metrics(self, total_train_loss, total_train_acc, dataset_size):
        self.current_epoch_data['train_loss'] = total_train_loss / dataset_size
        self.current_epoch_data['train_accuracy'] = total_train_acc / dataset_size

    def update_val_metrics(self, total_val_loss, total_val_acc, dataset_size):
        self.current_epoch_data['val_loss'] = total_val_loss / dataset_size
        self.current_epoch_data['val_accuracy'] = total_val_acc / dataset_size

    def update_learning_rate(self, lr):
        self.current_epoch_data['lr'] = lr

    #TODO this is not correct!
    def update_gradient_norm(self, grad_norm):
        if 'gradient_norms' not in self.current_epoch_data:
            self.current_epoch_data['gradient_norms'] = []
        self.current_epoch_data['gradient_norms'].append(grad_norm)

    def finalize_epoch(self):
        self.history.append(self.current_epoch_data)
        self.current_epoch_data = {}

    def update(self, total_train_loss, total_train_acc, total_val_loss, total_val_acc, train_data_size, val_data_size, learning_rate):
        """
        Updates the training metrics history with the average loss and accuracy for both training and validation.

        Args:
        - total_train_loss (float): The total training loss.
        - total_train_acc (float): The total training accuracy.
        - total_val_loss (float): The total validation loss.
        - total_val_acc (float): The total validation accuracy.
        - train_data_size (int): The size of the training data.
        - val_data_size (int): The size of the validation data.
        """
        # Calculate average loss and accuracy for both training and validation
        avg_train_loss = total_train_loss / train_data_size
        avg_train_acc = total_train_acc / train_data_size
        avg_val_loss = total_val_loss / val_data_size
        avg_val_acc = total_val_acc / val_data_size

        # Append these averages to the history
        self.history.append({
            "train_loss": avg_train_loss, 
            "train_accuracy": avg_train_acc, 
            "val_loss": avg_val_loss, 
            "val_accuracy": avg_val_acc,
            "lr": learning_rate 
        })

    def set_early_stopping_epoch(self, current_epoch):
        """
        Sets the epoch at which early stopping occurred.

        Args:
            epoch (int): The epoch number at which early stopping occurred.
        """
        self.early_stopping_epoch = current_epoch

    def get_latest_metrics(self):
            """
            Returns the most recent training metrics record.

            Returns:
                dict: A dictionary containing the most recent training metrics record.
            """
            return self.history[-1] if self.history else {}
    
    def get_latest_val_metrics(self):
        if self.history:
            latest = self.history[-1]
            return latest.get("val_loss", 0), latest.get("val_accuracy", 0)
        return 0, 0

    def get_val_loss_history(self):
        """Returns the history of validation loss."""
        return [epoch_data.get('val_loss', 0) for epoch_data in self.history]

    def get_train_loss_history(self):
        """Returns the history of training loss."""
        return [epoch_data.get('train_loss', 0) for epoch_data in self.history]

    def get_train_accuracy_history(self):
        """Returns the history of training accuracy."""
        return [epoch_data.get('train_accuracy', 0) for epoch_data in self.history]

    def get_val_accuracy_history(self):
        """Returns the history of validation accuracy."""
        return [epoch_data.get('val_accuracy', 0) for epoch_data in self.history]

    def get_learning_rate_history(self):
        """Returns the history of learning rate changes."""
        return [epoch_data.get('lr', 0) for epoch_data in self.history]

    def get_gradient_norms_history(self):
        """Returns the history of gradient norms."""
        return [np.mean(epoch_data.get('gradient_norms', [0])) for epoch_data in self.history]
