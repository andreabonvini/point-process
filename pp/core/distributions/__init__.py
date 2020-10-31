import numpy as np


def events2interevents(events: np.array) -> np.array:
    """
    Args:
        events: event-times as returned by the pp.utils.load() function.

    Returns:
         np.array which contains the inter-event intervals.
    """
    # We reset the events s.t. the first event is at time 0.
    # FIXME actually useless
    observ_ev = events - events[0]

    return np.diff(observ_ev)
