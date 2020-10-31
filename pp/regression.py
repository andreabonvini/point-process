from pp import ExponentialWeightsProducer
from pp.core.maximizers import InverseGaussianMaximizer
from pp.core.model import (
    InterEventDistribution,
    PointProcessDataset,
    PointProcessModel,
    WeightsProducer,
)

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


def regr_likel(
    dataset: PointProcessDataset,
    maximizer_distribution: InterEventDistribution,
    weights_producer: WeightsProducer = ExponentialWeightsProducer(),
) -> PointProcessModel:
    """
    Args:
        dataset: PointProcessDataset containing the specified AR order (p)
        and hasTheta0 option (if we want to account for the bias)
        maximizer_distribution: log-likelihood maximization function belonging to the Maximizer enum.
        weights_producer: WeightsProducer object to weight the samples of the dataset.

    Returns:
        Traines PointProcessModel

    """

    return maximizers_dict[maximizer_distribution.value](
        dataset, weights_producer
    ).train()
