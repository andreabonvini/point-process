# `point-process (pp)`

This site contains the *scientific* and `code` documentation for [point-process](https://github.com/andreabonvini/point-process), a `python` library for *Point Process Analysis*.

The *scientific documentation* consists of a series of quick blog-posts explaining the theoretical details behind the algorithmic implementation.

The *code documentation* consists of a series of code snippets showing the usage of the library.

## Scientific Documentation

#### Background knowledge

- [Hazard Function in Survival Analysis](theory_docs/Hazard Function in Survival Analysis.html)
- [Inhomogenous Poisson Process in the Time Domain](theory_docs/Inhomogenous Poisson Process in the Time Domain.html)

#### Model the Inter-Event pdf through the Inverse Gaussian distribution.

- [Inverse Gaussian: How to model the inter-event probability density function and define the likelihood.](theory_docs/Inverse Gaussian - MLE for Point Process Analysis.html)
- [Inverse Gaussian: How to *derive* the *gradient* vector and *hessian* matrix for maximum likelihood estimation.](theory_docs/Inverse Gaussian - Gradient and Hessian - Derivation.html)
- [Inverse Gaussian: How to *compute* the *gradient* vector and *hessian* matrix for maximum likelihood estimation.](theory_docs/Inverse Gaussian - Gradient and Hessian - Computation.html)

## Quick Tour

#### `Usage`

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

![](images/plot.png)

#### `InterEventDistribution`

```python
class InterEventDistribution(Enum):
    INVERSE_GAUSSIAN = "Inverse Gaussian"
#TODO add more distributions...
```

#### `WeightsProducers`

In order to weight the samples of our dataset (e.g. giving more importance to more recent samples) we can supply `regr_likel`  with a third argument `weights_producer`.

```python
# regr_likel signature:
def regr_likel(
        dataset: PointProcessDataset,
        maximizer_distribution: InterEventDistribution,
        weights_producer: WeightsProducer = ExponentialWeightsProducer()
) -> PointProcessModel:
```

We have two types of `WeightsProducers`:

- `ConstantWeightsProducer`: weights the samples by the same amount (i.e. `1`)

- `ExponentialWeightsProducer`: weights the samples with a decreasing exponential function `w(t)=exp(-alpha*t)`

	where `t` is the time distance from the most recent sample's target interval.

	```python
	class ExponentialWeightsProducer(WeightsProducer):
	    def __init__(self, alpha: float = 0.005):
	        """
	        Args:
	            alpha: Weighting time constant that governs the degree of influence
	                    of a previous observation on the local likelihood.
	        """
	        self.alpha = alpha
	
	    def __call__(self, target_intervals: np.ndarray) -> np.ndarray:
	        """
	            Args:
	                target_intervals:
	                    Target intervals vector (as stored in PointProcessDataset.wn)
	        """
	        self.target_intervals = target_intervals
	        return self._compute_weights()
	
	    def _compute_weights(self) -> np.ndarray:
	        target_times = np.cumsum(self.target_intervals) - self.target_intervals[0]
	        return np.exp(-self.alpha * target_times).reshape(-1, 1)[::-1]
	 
	```

```python
from pp import ExponentialWeightsProducer,ConstantWeightsProducer

wp = ExponentialWeightsProducer(alpha = 0.01) # or ConstantWeightsProducer()
pp_model = regr_likel(dataset, InterEventDistribution.INVERSE_GAUSSIAN, wp)
```

#### `PointProcessModel`

The result from the `regr_likel()` function is a `PointProcessModel`, this kind of object contains, other than the actual model, information about the training process and the learnt parameters.

```python
class PointProcessModel:
    def __init__(
        self,
        model: Callable[[np.ndarray], PointProcessResult],
        expected_shape: tuple,
        theta: np.ndarray,
        k: float,
        results: List[float],
        params_history: List[np.ndarray],
        distribution: InterEventDistribution,
        ar_order: int,
        hasTheta0: bool,
    ):
        """
        Args:
            model: actual model which yields a PointProcessResult
            expected_shape: expected input shape to feed the PointProcessModel with
            theta: final AR parameters.
            k: final shape parameter (aka lambda).
            results: negative log-likelihood values obtained during the optimization process (should diminuish in time).
            params_history: list of parameters obtained during the optimization process
            distribution: fitting distribution used to train the model.
            ar_order: AR order used to train the model
            hasTheta0: if the model was trained with theta0 parameter
        """
        self._model = model
        self.expected_input_shape = expected_shape
        self.theta = theta
        self.k = k
        self.results = results
        self.params_history = params_history
        self.distribution = distribution
        self.ar_order = ar_order
        self.hasTheta0 = hasTheta0

    def __repr__(self):
        return (
            f"<PointProcessModel<\n"
            f"\t<model={self._model}>\n"
            f"\t<expected_input_shape={self.expected_input_shape}>\n"
            f"\t<distributuon={self.distribution}>\n"
            f"\t<ar_order={self.ar_order}>\n"
            f"\t<hasTheta0={self.hasTheta0}>\n"
            f">"
        )

    def __call__(self, inter_event_times: np.ndarray) -> PointProcessResult:
        return self._model(inter_event_times)
```

#### `PointProcessResult`

A call to a `PointProcessModel` yields a `PointProcessResult`

```python
class PointProcessResult:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"mu: {self.mu}\nsigma: {self.sigma}"
```

