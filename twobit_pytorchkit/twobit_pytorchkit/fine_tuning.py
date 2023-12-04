import torch
import tqdm
from .records import TrainingRecord
from typing import Optional
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.optim as optim
from training.call_backs import EarlyStopping
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from typing import Optional, Union

from .model_trainer import ModelTrainer
try:
    import optuna
    optuna_available = True
except ImportError:
    optuna_available = False

class FineTuningTrainer(ModelTrainer):
    def __init__(self, *args, unfreeze_at_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfrozen_layers_count = 0

    def train(self, train_loader: DataLoader, val_loader: DataLoader, trial=None,
              epochs: int = 1, validation_frequency: int = 1,
              track_gradients: Optional[bool] = False):
        
        print("Starting Fine Tuning Training loop ...")
        
        epoch = 0
        done = False

        #TODO want to define this in objective function.  Look into
        if optuna_available and trial is not None:
            threshold = trial.suggest_float("threshold", 60, 90, log=True)

        while not done and epoch < epochs:
            epoch += 1
            self.model.train()

            pbar = self._training_step(train_loader, track_gradients)

            if (validation_frequency == 1 or (epoch % validation_frequency) == 0 or epoch == epochs):
                total_val_loss, total_val_acc = self._validation_step(val_loader)

                if optuna_available and trial is not None:
                    trial.report(total_val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    

                if self.should_unfreeze_layers(total_val_acc, threshold=threshold):
                    self.unfreeze_next_layer()

                done = self._update_scheduler_and_early_stopping(total_val_loss, epoch)

            #self.training_record.finalize_epoch()
            self._log_metrics(epoch, epochs)
            latest_metrics = self.training_record.get_latest_metrics()
    
            description = self._get_training_description(latest_metrics, epoch, epochs)
            
            #FIXME this is a hack to get the pbar to update
            print(description)
            pbar.set_description(description)

        
        pbar.close()
        return self.training_record

    def _unfreeze_layers(self):
        # Logic to unfreeze layers
        for param in self.model.parameters():
            param.requires_grad = True
        print("Unfroze all layers for fine-tuning.")

    def should_unfreeze_layers(self, val_accuracy, threshold=0.5):
        # Implement logic to decide if layers should be unfrozen based on accuracy
        return val_accuracy >  threshold
    
    def unfreeze_next_layer(self):

        #TODO may need to adjust learning rate. reset to lower after unfreezing?

        # Count the layers backward and unfreeze them one by one
        layers = list(self.model.children())
        total_layers = len(layers)

        if self.unfrozen_layers_count < total_layers:
            layer_to_unfreeze = layers[total_layers - 1 - self.unfrozen_layers_count]
            for param in layer_to_unfreeze.parameters():
                param.requires_grad = True

            self.unfrozen_layers_count += 1
            print(f"Unfroze layer {self.unfrozen_layers_count}/{total_layers}")
        else:
            print("All layers are already unfrozen.")