# Artificien: Experimental

This repo stores Jupyter Notebooks used as examples to test the federated learning workflow of the Artificien platform, and the Library we use to implement in JupyterHub.

*Caution Artificien Library has migrated to new repo [artificien_python_library](https://github.com/dartmouth-cs98/artificien_python_library). Documentation on artificienLibrary here is out of date*
* dataSimulation
    * A Jupyter notebook that writes out fake data files
    * Note that a version of this fake data script that is connected with the DynamoDB and writes out CSVs based on Dynamo inputs is in the infra repo. 
* deploymentExamples
   * Jupyter notebooks that show how to define a model for federated learning using pysyft out-of-the-box, using the artificien library, and training a model on the node as a server
## Architecture

* artificienLibrary
    * Libraries used
        * `PyGrid` the basis of the wrapper library, the core technology behind PySyft
        * 'artificienlib' this is the library that it installs to
        * `pysyft` is the conda environment that contains all the needed packages for the library to run. Requirements can be found in 'requirements.txt'
    * Code Organization
        * `syftfunctions.py` is the artificien library - this is the wrapper around PySyft functions, making it easier for data scientists to build robust federated learning capable pytorch based machine learning models. Functional models can be built and sent for training on the pysyft node in under 5 lines of code + normal pysyft model definition
        * `constants.py` stores authentication protocols for communicating with the node and other pieces of artificien infrastructure
        * `test_myfunctions.py` is several tests of syftfunctions to ensure the artificien library works as expected
* dataSimulation
    * Libraries used
        * Python
    * Code Organization
        * `counting_statistics.ipynb` was the first pass at being able to work with PyGrid to capture data from the node. However, we determined that this wasn't the right way to do it so it was never used in production.
        * `fakedatascript.ipynb` takes in column headers, type of data in each column (int/float), the range for each column, and how many rows as inputs. It then writes a csv file into testdata
        * `regression.ipynb` was another effort at creating a test regression to prove out our use case, again, I did not fully understand how we would be connecting with the actual PyGrid node etc (everything was spun up locally), so this is also not in production.
        * `fakedata` is a folder that stores the CSVs written by the fakedatascript
* deploymentExamples
    * Libraries
        * `pysyft` conda environment with needed libraries. See above
    * Code Organization
        * `model_centric_analysis.ipynb` is jupyter notebook outlining how to launch a pytorch model to the artificien grid (where it is sent off to edge devices to get trained) using pysyft's out-of-the-box library
        * `model_centric_analysis_with_lib.ipynb` is a jupyter notebook outlining how to launch a pytorch model to the artificien grid using the pytorch library
        * `model_centric_test.ipynb` is a jupyter notebook outlining how to train a model that's on the node by pretending to be a server storing data.


## Setup

* artificienLibrary
    * Needed Package Installation:
        * Inside of artificienLibrary folder run `pip install -r requirements.txt`
    * To test:
        * Inside of artificienLibrary folder run `python setup.py pytest`
    * To compile library:
        * Inside of artificienLibrary folder run `python setup.py bdist_wheel`
* dataSimulation
    * Ensure Python installation and that you can run Jupyter Notebooks.
* deploymentExamples
    * Package instalation
        * Inside of artificienLibrary folder run `pip install -r requirements.txt` if you haven't installed the pysyft conda environment already
## Deployment
* artificienLibrary
    * Run
        * After completing setup above, from the artificienLibrary folder
        * Run `pip install dist/artificienlib-0.1.0-py3-none-any.whl`
        * Now you can simply import the artificien library in python as `import artificienlib`
* dataSimulation
    * clone the repo, and just run each successive block in fakedatascript.ipynb
* deploymentExamples
    * Run
        * Open jupyter notebooks using notebook or labs and run cells to see how model building, model deployment, and model training works via federated learning and artificien

## Authors

* Jake Epstein '21
* Tobias Lange '21

## Acknowledgments

* OpenMined Pysyft Tutorials guided the regression buildout
* OpenMined Pygrid examples 
