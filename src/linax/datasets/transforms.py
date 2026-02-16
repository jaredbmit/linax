"""Transformations for datasets."""

import numpy as np


class AddGaussianNoise:
    """Add Gaussian noise to input sequences.

    Args:
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility (optional)
    """

    def __init__(self, noise_std: float, seed: int | None = None):
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add noise to input.

        Args:
            x: Input array of shape (seq_len, features)

        Returns:
            Noisy input
        """
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, x.shape)
            return x + noise
        return x

    def __repr__(self):
        """Print w/ noise std."""
        return f"{self.__class__.__name__}(noise_std={self.noise_std})"


class AddUniformNoise:
    """Add uniform noise to input sequences.

    Args:
        noise_scale: Scale of uniform noise (added from [-scale, +scale])
        seed: Random seed for reproducibility (optional)
    """

    def __init__(self, noise_scale: float, seed: int | None = None):
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add noise to input.

        Args:
            x: Input array of shape (seq_len, features)

        Returns:
            Noisy input
        """
        if self.noise_scale > 0:
            noise = self.rng.uniform(-self.noise_scale, self.noise_scale, x.shape)
            return x + noise
        return x

    def __repr__(self):
        """Print w/ noise scale."""
        return f"{self.__class__.__name__}(noise_scale={self.noise_scale})"
