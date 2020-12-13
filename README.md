![](docs/images/ppbig.png)

This repository contains a `Python` implementation of the `MATLAB` software provided by Riccardo Barbieri and Luca Citi at [this](http://users.neurostat.mit.edu/barbieri/pphrv) link.

### Installation

##### Install with pip

```
pip install point-process
```

##### Install from source

```
# clone repo
git clone https://github.com/andreabonvini/point-process.git
cd point-process
# Install dependencies
pipenv install --dev
# Update dependencies
pipenv sync
```

### Documentation

The technical and scientific *documentation* for this repository can be found [here](https://andreabonvini.github.io/point-process/).

### Usage example

```python
from pp import InterEventDistribution, PointProcessDataset
from pp import regr_likel
# Suppose we have a np.array inter_events containing inter-event times expressed in seconds.
# Build a dataset object with the specified AR order (p) and hasTheta0 option (if we want to account for the bias)
dataset = PointProcessDataset.load(
    inter_events_times=inter_events,
    p=9,
    hasTheta0=True
)
# We pass to regr_likel the dataset defined above and the distribution we want to fit 
pp_model = regr_likel(dataset, InterEventDistribution.INVERSE_GAUSSIAN)

# We build the same dataset without the hasTheta0 option just to test our model:
dataset = PointProcessDataset.load(
    inter_events_times=inter_events,
    p=9,
    hasTheta0=False
)
test_data = dataset.xn
targets = dataset.wn
predictions = [pp_model(sample).mu for sample in test_data]
# We can then plot our predictions against the actual targets...
```

![](docs/images/plot.png)

### Contributing

If you want to contribute to the project, check the `CONTRIBUTING.md` file to correctly set up your environment.
