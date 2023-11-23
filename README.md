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
# Define optimizer and loss function
optimizer = torch.optim.SGD(resNet34.parameters(), lr=.001, momentum=0.9, weight_decay=1e-4)
loss_function = torch.nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

# Initialize the ModelTrainer
trainer = ModelTrainer(model=resNet34, optimizer=optimizer, 
                       #scheduler=scheduler, 
                       #clip_type='norm', clip_value=5.0,
                       criterion=loss_function, 
                       device=device, early_stopping=early_stopping)

# Train the model
training_record = trainer.train(train_loader, val_loader, epochs=10, validation_frequency=1)
```

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/1f5c117a-5fee-4d2a-b19d-44f648d4204f)

## Example Usage (TrainingPlotter):

```
plotter = TrainingPlotter(training_record=training_record)
plotter.plot_training_history()
plotter.plot_gradient_norms()
plotter.plot_learning_rate()
```

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/e58a8dd1-a133-46dd-a8e7-abcd386471bc)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/ff111ba4-9b08-4eb3-a344-78257f7ecdd1)

![image](https://github.com/twobit-five/PyTorchTrainingKit/assets/69398054/4c13e738-c34f-497c-b7d5-d1f3e2dd39a9)






