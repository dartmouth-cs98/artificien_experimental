# Artificien: Experimental

This repo stores Jupyter Notebooks used as examples to test the federated learning workflow of the Artificien platform.

**Note: Artificien Library has migrated to new repo: [artificien_python_library](https://github.com/dartmouth-cs98/artificien_python_library)**
* optimizationModels
    * Series of experimental optimizers to improve various parts of Artificien's infrastructure
* dataSimulation
    * A Jupyter notebook that writes out fake data files
    * Note that a version of this fake data script that is connected with the DynamoDB and writes out CSVs based on Dynamo inputs is in the infra repo. 
* deploymentExamples
   * Jupyter notebooks that show how to define a model for federated learning using pysyft out-of-the-box, using the artificien library, and training a model on the node as a server
## Architecture
* optimizationModels
    * `data_pricing.ipynb` explores how to best price datasets as data has different value based on its use case, and sometimes its profit mazimizing to sell to one buyer over many buyers (hedge fund vs. researcher)
    * `federated_scheduler.ipynd` looks two questions. Given a bunch of models we can train, how do we decide what to train first and after we've decided on models how do we decide what devices we should train the models on.
    * `user_graph.ipynb` looks at how to decide when to spin up and down nodes dynamically based on user need
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
* optimizationModels
    * Run jupyter notebooks
* dataSimulation
    * Ensure Python installation and that you can run Jupyter Notebooks.
* deploymentExamples
    * Package instalation
        * Inside of artificienLibrary folder run `pip install -r requirements.txt` if you haven't installed the pysyft conda environment already
## Deployment
* optimizationModels
    * Clone repos, run jupyter notebooks
* dataSimulation
    * clone the repo, and just run each successive block in fakedatascript.ipynb
* deploymentExamples
    * Run
        * Open jupyter notebooks using notebook or labs and run cells to see how model building, model deployment, and model training works via federated learning and artificien

## Authors

* Jake Epstein '21 (optimizationModels, deploymentExamples)
* Tobias Lange '21 (dataSimulation)

## Acknowledgments

* OpenMined Pysyft Tutorials guided the regression buildout
* OpenMined Pygrid examples 
