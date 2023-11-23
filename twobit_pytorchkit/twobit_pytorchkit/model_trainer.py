import torch
import tqdm
from training.records import TrainingRecord
from typing import Optional
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.optim as optim
from training.call_backs import EarlyStopping
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from typing import Optional, Union

class ModelTrainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 criterion, scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None, 
                 device: str = 'cpu', early_stopping: Optional[EarlyStopping] = None,
                 clip_type: Optional[str]=None, clip_value: Optional[float]=None):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = early_stopping
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.training_record = TrainingRecord()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 1, validation_frequency: int = 1,
              track_gradients: Optional[bool] = False):
        
        print("Starting training loop ...")
        
        epoch = 0
        done = False


        while not done and epoch < epochs:
            epoch += 1
            self.model.train()

            pbar = self._training_step(train_loader, track_gradients)

            #TODO fix this logic
            if (validation_frequency == 1 or (epoch % validation_frequency) == 0 or epoch == epochs):
                total_val_loss, total_val_acc = self._validation_step(val_loader)
                done = self._update_scheduler_and_early_stopping(total_val_loss, epoch)

            #self.training_record.finalize_epoch()
            self._log_metrics(epoch, epochs)
            latest_metrics = self.training_record.get_latest_metrics()

            # Think aabout returning N/A if no val metrics
            description = f"Epoch: {epoch}/{epochs}, Train Loss: {latest_metrics.get('train_loss', -1):.4f}, Val Loss: {latest_metrics.get('val_loss', -1):.4f}, Val Acc: {latest_metrics.get('val_accuracy', -1):.4f}"
            if self.early_stopping:
                description += f", Early Stopping: {self.early_stopping.status}"
            
            #FIXME this is a hack to get the pbar to update
            print(description)
            pbar.set_description(description)
        
        pbar.close()
        return self.training_record

    def _training_step(self, train_loader: DataLoader, track_gradients: Optional[bool] = False):

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

            if (track_gradients):
                grad_norm = self._calculate_gradient_norm()
                self.training_record.update_gradient_norm(grad_norm)

            if self.clip_type:
                self._apply_gradient_clipping()

            self.optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train_acc += (predicted == targets).sum().item()

            # Update the progress bar
            # We may be able to finilize the epoch here instead of in the train function, then we could update the progress bar correctly here.

        self.training_record.update_train_metrics(total_train_loss, total_train_acc, len(train_loader.dataset))
        return pbar

    def _validation_step(self, val_loader: DataLoader):
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
        
        # Update the training record with validation metrics
        self.training_record.update_val_metrics(total_val_loss, total_val_acc, len(val_loader.dataset))
        return total_val_loss, total_val_acc

    def _update_scheduler_and_early_stopping(self, total_val_loss: float, epoch: int):
        if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR) and not isinstance(self.scheduler, optim.lr_scheduler.CyclicLR):
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        if self.early_stopping and self.early_stopping(self.model, total_val_loss, epoch):
            self.training_record.set_early_stopping_epoch(epoch)
            print(self.early_stopping.status)
            return True
        else:
            return False

    def _log_metrics(self, epoch: int, epochs: int):

        # Could also implement tenrboard logic here for live data updates during training

        # Update the training record with the learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.training_record.update_learning_rate(lr)

        # Finalize the epoch data in the training record
        self.training_record.finalize_epoch()

    def _update_pbar_description(self, pbar, epoch, epochs):
        latest_metrics = self.training_record.get_latest_metrics()
        description = f"Epoch: {epoch}/{epochs}, Train Loss: {latest_metrics.get('train_loss', -1):.4f}, Val Loss: {latest_metrics.get('val_loss', -1):.4f}, Val Acc: {latest_metrics.get('val_accuracy', -1):.4f}"
        if self.early_stopping:
            description += f", Early Stopping: {self.early_stopping.status}"
        pbar.set_description(description)
        pbar.close()

    def _apply_gradient_clipping(self):
        if self.clip_type == 'norm':
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

    def _calculate_gradient_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm