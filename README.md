# PyTorchTrainingKit
Classes to help train pytorch models

This is still very much a work in progress.  The project is simply tools I have created to help train PyTorch models. There is limited functionality and may not follow best practices. Suggestions are welcome for improvements.


## Installation
[ ] Need to finish installations instruction.
[ ] Eventually publish to PyPi ?

For now directly clone the project and import and use the classes in local project. The classes must be copied to a folder named 'training' with an __init__.py for now.  This will eventually be fixed.

## Example Usage (ModelTrainer):
Note:
```
from training.model_trainer import ModelTrainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training.call_backs import EarlyStopping
from training.records import TrainingRecord


# Define optimizer and loss function
optimizer = torch.optim.SGD(resNet34.parameters(), lr=20, momentum=0.9, weight_decay=1e-4)
loss_function = torch.nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

# Initialize the ModelTrainer
trainer = ModelTrainer(model=resNet34, optimizer=optimizer, 
                       scheduler=scheduler, 
                       #clip_type='norm', clip_value=5.0,
                       criterion=loss_function, 
                       device=device, early_stopping=early_stopping)

# Train the model
training_record = trainer.train(train_loader, val_loader, epochs=10, validation_frequency=1, track_gradients=True)
```

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/1f5c117a-5fee-4d2a-b19d-44f648d4204f)

## Example Usage (TrainingPlotter):

```
from training.plotters import TrainingPlotter

plotter = TrainingPlotter(training_record=training_record)
plotter.plot_training_history()
plotter.plot_gradient_norms()
plotter.plot_learning_rate()
```

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/472dcd6e-e5a3-42af-b7fd-0d5c03a8e172)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/66bec286-8759-4a7f-bf0c-e527c243e7c8)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/1429de92-221f-4d71-b2fa-d11b3ea8c64d)






