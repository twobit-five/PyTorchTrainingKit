import matplotlib.pyplot as plt

class TrainingRecord:
    """
    A class to store training metrics for each epoch and provide methods to update, retrieve, plot, and save the metrics.
    """
    def __init__(self):
        # Initialize a list to store metrics for each epoch
        self.history = []
        self.early_stopping_epoch = None

    def update(self, total_train_loss, total_train_acc, total_val_loss, total_val_acc, train_data_size, val_data_size):
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
            "val_accuracy": avg_val_acc
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

    def plot_training_history(self):
        if not self.history:
            print("No data to plot.")
            return

        epochs = range(1, len(self.history) + 1)
        train_losses = [x['train_loss'] for x in self.history]
        val_losses = [x['val_loss'] for x in self.history]
        train_accuracies = [x['train_accuracy'] for x in self.history]
        val_accuracies = [x['val_accuracy'] for x in self.history]

        # Plotting training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        if self.early_stopping_epoch:
            # Get the validation loss at the early stopping point
            es_val_loss = val_losses[self.early_stopping_epoch - 1]
            
            # Vertical line
            plt.axvline(x=self.early_stopping_epoch, color='gray', linestyle='--', label='Early Stopping')
            
            # Horizontal line
            plt.axhline(y=es_val_loss, color='gray', linestyle='--')

            # Highlight the intersection point
            plt.scatter(self.early_stopping_epoch, es_val_loss, color='yellow', edgecolor='black', zorder=5)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        if self.early_stopping_epoch:
            plt.axvline(x=self.early_stopping_epoch, color='gray', linestyle='--', label='Early Stopping')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, filename):
        pass