from torch import nn
import copy
import torch 
import os

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss doesn't improve
    for a specified number of consecutive epochs.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
                                      of the monitored quantity.

    Attributes:
        best_model_state (dict): Model weights from the epoch with the best value of the monitored quantity.
        best_loss (float): Best value of the monitored quantity.
        best_epoch (int): Epoch number corresponding to the best model state.
        counter (int): Number of epochs with no improvement in the monitored quantity.
        status (str): Status message indicating the current state of early stopping.
    """
    def __init__(self, patience:int=5, min_delta: float=0, restore_best_weights: bool=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.best_loss = None
        self.best_epoch = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss:float, epoch: int):
        """
        Check if the monitored quantity has improved and update the best model state accordingly.

        Args:
            model (torch.nn.Module): Model to be evaluated.
            val_loss (float): Value of the monitored quantity on the validation set.
            epoch (int): Current epoch number.

        Returns:
            bool: Whether to stop training or not.
        """
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            self.status = f"Improvement found at epoch {epoch + 1}, counter reset."
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs."
            if self.counter >= self.patience:
                self.status += f" Early stopping triggered at epoch {epoch + 1}."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_state)
                return True
        return False
    
class CheckPointer:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_val_loss = float('inf')

    def save_checkpoint(self, state, is_best):
        """Saves model and training parameters at checkpoint"""
        torch.save(state, os.path.join(self.save_path, 'last_checkpoint.pth'))
        if is_best:
            torch.save(state, os.path.join(self.save_path, 'best_checkpoint.pth'))

    def check_improvement(self, val_loss, model, optimizer, epoch):
        """Check if the validation loss improved and save a checkpoint"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, is_best=True)

# TODO other callbacks, such as Logger etc.
