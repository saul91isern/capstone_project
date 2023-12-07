# Project: Capstone Project

### Install

This project has been developed in a Anaconda Environment [Anaconda](https://www.anaconda.com/download/). All the installed dependencies are specified
in the `environment.yml` file. Some of the most important dependencies are:

- [NumPy](http://www.numpy.org/)
- [Keras](https://keras.io)
- [h5py](https://www.h5py.org)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Code

Template code is provided in the `capstone-project.ipynb` notebook file.

### Run

In a terminal or command window, navigate to the top-level project directory `capstone-project/` (that contains this README) and run one of the following commands:

```bash
jupyter notebook capstone-project.ipynb
```
This will open the iPython Notebook software and project file in your browser.

Some useful commands to interacte with an Anaconda environment are:

- Create conda virtual env with needed dependencies:
```bash
conda env create -f environment.yml
```
- Update dependecies in the virtual env:
```bash
conda env update -f=environment.yml
```
- Remove conda virtual env:
```bash
conda env remove --name capstone-lab
```
- Activate conda virtual env:
```bash
conda activate capstone-lab
```
- Deactivate conda virtual env:
```bash
conda deactivate
```
