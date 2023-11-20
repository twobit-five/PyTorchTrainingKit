import torch
import copy
import tqdm
from training_metrics import TrainingRecord
import statistics
from typing import Optional
from torch.utils.data import DataLoader
from torch import nn, optim

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


class ModelTrainer:
    """
    A class for training PyTorch models.

    Args:
        model (nn.Module): The PyTorch model to train.
        optimizer (optim.Optimizer, optional): The optimizer to use for training. Defaults to None.
        criterion (callable, optional): The loss function to use for training. Defaults to None.
        scheduler (optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler to use for training. Defaults to None.
        device (str, optional): The device to use for training. Defaults to 'cpu'.
        early_stopping (EarlyStopping, optional): An instance of the EarlyStopping class to use for early stopping. Defaults to None.
    """
class ModelTrainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: callable, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None, device: str = 'cpu', early_stopping: Optional[EarlyStopping] = None):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = early_stopping
        self.training_record = TrainingRecord()

    #TODO need to start thinking about separating some of the functionality into different methods! This is getting messy!
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int=1, clip_type=None, clip_value=None) -> TrainingRecord:
        """
        Trains the model using the given train_loader and val_loader for the specified number of epochs.

        Args:
            train_loader (DataLoader): The DataLoader for the training set.
            val_loader (DataLoader): The DataLoader for the validation set.
            epochs (int, optional): The number of epochs to train the model for. Defaults to 1.
        Returns:
            TrainingRecord: The TrainingRecord object containing the training history.
        """
        train_data_size = len(train_loader.dataset)
        val_data_size = len(val_loader.dataset)
        epoch = 0
        done = False

        while not done and epoch < epochs:
            epoch += 1
            self.model.train()
            total_train_loss = 0
            total_train_acc = 0
            steps = list(enumerate(train_loader))
            pbar = tqdm.tqdm(steps)

            for i, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # apply gradient clipping, if specified
                if self.clip_type:
                    self._apply_gradient_clipping(self.optimizer, self.clip_type, self.clip_value)

                self.optimizer.step()

                total_train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train_acc += (predicted == targets).sum().item()

                # Scheduler step per batch for schedulers like OneCycleLR or CyclicLR
                if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR) or isinstance(self.scheduler, optim.lr_scheduler.CyclicLR):
                    self.scheduler.step()

                # Epoch finished
                # TODO consider frequency of validation!!
                if (i == len(steps) - 1):
                    total_val_loss, total_val_acc = self._validate(val_loader)

                    # Scheduler step per epoch for most other schedulers
                    #TODO this is WRONG need to update and fix. ONECYCLELR and CYCLICLR are not the only schedulers that need to be stepped per batch.
                    #Which is checked above
                    if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR) and not isinstance(self.scheduler, optim.lr_scheduler.CyclicLR):
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(total_val_loss)
                        elif self.scheduler is not None:
                            self.scheduler.step()

                    # Update TrainingRecord
                    self.training_record.update(total_train_loss, total_train_acc, total_val_loss, total_val_acc, train_data_size, val_data_size)
                    # Retrieve the latest average metrics for progress bar and early stopping
                    latest_metrics = self.training_record.get_latest_metrics()
                    avg_train_loss = latest_metrics.get("train_loss", 0)
                    avg_train_acc = latest_metrics.get("train_accuracy", 0)
                    avg_val_loss = latest_metrics.get("val_loss", 0)
                    avg_val_acc = latest_metrics.get("val_accuracy", 0)

                    # Early stopping logic
                    if self.early_stopping:
                        self.early_stopping(self.model, avg_val_loss, epoch)
                        if self.early_stopping.counter >= self.early_stopping.patience:
                            print(self.early_stopping.status)
                            self.training_record.set_early_stopping_epoch(current_epoch=self.early_stopping.best_epoch)
                            done = True

                    # Update progress bar description
                    description = f"Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
                    if self.early_stopping:
                        description += f", Early Stopping: {self.early_stopping.status}"
                    pbar.set_description(description)

            pbar.close()

        return self.training_record
    
    def _apply_gradient_clipping(optimizer, clip_type='norm', clip_value=1.0):
        """
        Apply gradient clipping to the optimizer.

        Args:
            optimizer: The optimizer with model parameters and gradients.
            clip_type (str): Type of gradient clipping ('norm' or 'value').
            clip_value (float): The clipping threshold.
        """
        if clip_type == 'norm':
            torch.nn.utils.clip_grad_norm_(optimizer.parameters(), clip_value)
        elif clip_type == 'value':
            torch.nn.utils.clip_grad_value_(optimizer.parameters(), clip_value)

    def _validate(self, val_loader: DataLoader) -> tuple:
            """
            Validates the model on the validation set.

            Args:
                val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.

            Returns:
                tuple: A tuple containing the total validation loss and total validation accuracy.
            """
            self.model.eval()
            total_val_loss = 0
            total_val_acc = 0

            with torch.inference_mode():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    total_val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val_acc += (predicted == targets).sum().item()

            return total_val_loss, total_val_acc


