# PyTorchTrainingKit
Classes to help train pytorch models

This is still very much a work in progress.  The project is simply tools I have created to help train PyTorch models. There is limited functionality and may not follow best practices. Suggestions are welcome for improvements.

## Features and Bugs
https://github.com/twobit-five/PyTorchTrainingKit/issues

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
training_record = trainer.train(train_loader, val_loader, epochs=20, validation_frequency=1, track_gradients=True)
```

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/a94f4455-28ee-42d5-b0a7-17fb0f7482ff)


## Example Usage (TrainingPlotter):

```
from training.plotters import TrainingPlotter

plotter = TrainingPlotter(training_record=training_record)
plotter.plot_training_history()
plotter.plot_gradient_norms()
plotter.plot_learning_rate()
```
![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/e0e594cc-63d1-48c4-9cde-dbd37d838f5e)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/24a28ab3-2222-45e8-ae4d-b0b55e30f6eb)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/1391ce67-7d9c-40d5-a480-a062969728ba)





