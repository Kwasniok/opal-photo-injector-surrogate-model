# opal-photo-injector-surrogate-model

Framework to train and evaluate basic neural network surrogate models for low dimensional input/output vectors.

# Requirements/Tested on:
- python v3.13+ on linux


# Start Jupyter Lab
To start the jupyter lab execute
```bash
./start.sh
```
Note: This will automatically [install](#installation) all python packages as well.

⚠️ **IMPORTANT**:Make sure to select the kernel called `<your-project-name> (pythonX.Y))` for all your notebooks. Otherwise they will not find the installed python packages!

# Usage
This project uses torch lightning.

1. store required files in `data` 
2. inspect and use the `.ipynb` in numerical order

🗒️ **NOTE**: The experiment dispatch allows to automatically dispatch the fitting and evaluation of individual models e.g. for a grid scan of hyperparamaters a.k.a. as finetuning.


# License
This project is licensed under [CC-BY-SA-4.0](http://creativecommons.org/licenses/by-sa/4.0).