# Artificien Library

This library wraps PyGrid/PySyft functionalities to make the library more intuitive to use. The following is a basic use guide. For more step-by-step documentation, refer to the tutorials under `deploymentExamples` folder

## Installation

Inside the `artificienLibrary` folder, run `pip install -r requirements.txt` to ensure all the needed packages are installed

To compile the library, run `python setup.py bdist_wheel`

To install the compiled library, run `pip install dist/artificienlib-0.1.0-py3-none-any.whl`

## Importation

`artificienlib` can now be imported in a python file with the line `import artificienlib`

## Building a Model

Building a model using `artificienlib` is a three step process.

### Building a PyTorch Model

The first step is specifying a standard PyTorch model of some input and output size. This process is no different than using standard PyTorch. Refer to PyTorch documentation for additional information, and to `model_lib.ipynb` in the `artificien_experimental` folder for an example.

### Defining training plan

Thes seconds step is defining a training plan. First, we must choose some dummy X and Y representative of our input and output parameters respectively. The `X` should be sample size 1 example of inputs, and `Y` the corresponding output. Such values, along with our pytorch model, are passed into our training plan definition. To compile the training plan with default loss function (softmax cross entropy and optimizer (naive stochaistic gradient descent), simply run: `def_training_plan(model, X, Y)`

Additional specifiers can be passed into the training plan definition via a dictionary that allow you to change the loss function and optimizer to whatever is suitable for your model. This is described bellow.

The training plan has three core components:

1. **Loss Function** - The loss function is defaulted to softmax cross entropy. To change it, pass in a dictionary containing item `{"loss": my_loss_function}` where `my_loss_function` is a functional definition of your loss function that takes input `my_loss_function(logits, targets, batch_size)`. `logits` and `targets` must be numpy arrays of size batch_size x C and are the output of model and actual targets respectively. `artificienlib` also has some standard loss functions available out of the box. For instance, `{"loss":sf.mse_with_logits}` will use a standard implementation of mean squared error, no user created function instantiation required.

2. **Optimizer** - The optimizer is defaulted to naive stochaistic gradient descent. To change it, pass in a dictionary (same one as above) containing item `{"optimizer": my_optimizer_function}` where `my_optimizer_function` is a functional definition of your optimizer function that takes input `my_optimizer_function(param, **kwargs)`. `kwargs` is simply a dictionary of basic parameters. For the purposes of defining an optimizer, `kwargs['lr']` will give you the learning rate. param are the coefficients.

3. **Training Steps** - The training plan defines back propagation (i.e. how training happens at each iteration). A `"training_plan"` item and corresponding function in the dictionary would let you change it. It's unlikely this feature needs to be used, but if you'd like more information refer to `syftfunctions.py` in `artificienlib` - the library codebase.

The outputs of a call to `def_training_plan(model, X, Y)` is our model parameters and a training plan object respectively. An example call to `def_training_plan(model, X, Y)` is 

```
from artificienlib import syftfunctions as sf
model_params, training_plan = sf.def_training_plan(model, X, y, {"loss":sf.mse_with_logits})
```

### Defining our Averaging Plan

Federated learning essentially compiles gradients from every dataset it's been trained on, then has to somehow combine these gradients together. For most usecases, these gradients are simply averaged together. This is the default setup, and the function `def_avg_plan(model_params)` does this. Pass in the `model_params` returned by `def_training_plan`.

The outputs of a call to `def_avg_plan(model_params)` is an `avg_plan` object. An example call is:

```
from artificienlib import syftfunctions as sf
avg_plan = sf.def_avg_plan(model_params)
```

### Connecting to artificien infrastructure

Next, you must connect to artificien's infrastructure so the model can be sent via the artificien "node" to relevant consumer devices. This connection allows your model, training plan, and average plan to be used. Simply call:

```
from artificienlib import syftfunctions as sf
grid = sf.artificien_connect()
```

### Sending the Model

Lastly, we must send our model, training plan, and average plan to be trained. Using the function `send_model` we must pass in the `name`, `version`, `batch_size`, `learning_rate`, `max_updates`, `model_params`, `grid`, `training_plan`, and `average_plan`. An example call is as follows:

```
from artificienlib import syftfunctions as sf
sf.send_model(name="perceptron", version="0.3.0", batch_size=1, learning_rate=0.2, max_updates=10, model_params=model_params, grid=grid, training_plan=training_plan, avg_plan=avg_plan)
```






