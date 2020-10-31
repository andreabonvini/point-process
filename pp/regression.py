from pp.core.maximizers import InverseGaussianMaximizer
from pp.core.model import InterEventDistribution, PointProcessDataset, PointProcessModel

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


def regr_likel(
    dataset: PointProcessDataset, maximizer_distribution: InterEventDistribution,
) -> PointProcessModel:
    """

    Args:
        dataset: PointProcessDataset containing the specified AR order (p)
        and hasTheta0 option (if we want to account for the bias)
        maximizer_distribution: log-likelihood maximization function belonging to the Maximizer enum.

    Returns:
        Traines PointProcessModel

    """

    return maximizers_dict[maximizer_distribution.value](dataset).train()
