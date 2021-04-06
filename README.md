![](docs/images/ppbig.png)

This repository contains a `Python` implementation of the `MATLAB` software provided by Riccardo Barbieri and Luca Citi [here](http://users.neurostat.mit.edu/barbieri/pphrv).

*Refer to the following papers for more details:*

- [*A point-process model of human heartbeat intervals: new definitions of heart rate and heart rate variability*](https://pubmed.ncbi.nlm.nih.gov/15374824/)
- [*The time-rescaling theorem and its application to neural spike train data analysis*](https://pubmed.ncbi.nlm.nih.gov/11802915/)

# Requirements

Before proceeding with the installation step, make sure you have [nlopt](https://nlopt.readthedocs.io/en/latest/) installed on your machine.

(Note: support for Windows will be available soon)

### On MacOs

```
brew install nlopt
```

### On Linux

```
# refer to the official documentation
```

# Installation

### Install with pip

```
pip install pointprocess
```

### Install from source

```
# clone repo
git clone https://github.com/andreabonvini/pointprocess.git
cd pointprocess
# Install dependencies
pipenv install
```

# Documentation

The technical and scientific *documentation* for this repository can be found [here](https://andreabonvini.github.io/pointprocess/).

# Contributing

If you want to contribute to the project, check the `CONTRIBUTING.md` file to correctly set up your environment.
