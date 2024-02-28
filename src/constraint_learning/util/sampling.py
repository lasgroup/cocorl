"""Some helper functions for sampling."""
import contextlib
import random
from typing import Optional

import numpy as np


@contextlib.contextmanager
def temp_seed(seed: Optional[int]):
    if seed is not None:
        orig_seed = np.random.randint(0, 2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
    yield
    if seed is not None:
        random.seed(orig_seed)
        np.random.seed(orig_seed)


def sample_unit_vectors(*, num_samples: int, num_dims: int) -> np.ndarray:
    """Samples from the unit sphere in `num_dims` dimensions."""
    samples = np.random.uniform(low=0, high=1, size=(num_samples, num_dims))
    samples -= 0.5
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    return samples
