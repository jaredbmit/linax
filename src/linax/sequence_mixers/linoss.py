"""Sequence Mixer for LinOSS-IM, LinOSS-IMEX, and Damped LinOSS-IMEX models.

See: https://openreview.net/pdf?id=GRMfXcAAFh
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import sympy as sp
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray

from linax.sequence_mixers.base import SequenceMixerConfig


@dataclass(frozen=True)
class LinOSSSequenceMixerConfig(SequenceMixerConfig):
    """Configuration for the LinOSS sequence mixer.

    This configuration class defines the hyperparameters for the LinOSS sequence mixer.
    It includes options for the model's architecture, training parameters, and behavior.

    Attributes:
        state_dim: Dimensionality of the state space.
        discretization: Discretization method to use.
        damping: Whether to use damping.
        r_min: Minimum value for the radius.
        theta_max: Maximum value for the theta parameter.
    """

    state_dim: int = 64

    discretization: Literal["IM", "IMEX", "EX"] = "IMEX"
    damping: bool = True
    initialization: Literal["RT", "AG"] = "RT"
    r_min: float = 0.9
    theta_max: float = jnp.pi

    def build(self, in_features: int, key: PRNGKeyArray) -> LinOSSSequenceMixer:
        """Build sequence mixer from config.

        Args:
            in_features: Input dimensionality.
            key: JAX random key for initialization.

        Returns:
            The sequence mixer instance.
        """
        return LinOSSSequenceMixer(in_features=in_features, cfg=self, key=key)


class LinOSSSequenceMixer[ConfigType: LinOSSSequenceMixerConfig](eqx.Module):
    """LinOSS sequence mixer layer.

    This layer implements the LinOSS sequence mixer.

    Attributes:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        C: Output matrix.
        D: Output matrix.
        steps: Step sizes for the sequence mixer.
        discretization: Discretization method to use.
        damping: Whether to use damping.
    """

    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: Literal["IMEX", "IMEX2", "IM", "EX"]
    damping: bool

    def __init__(
        self,
        in_features: int,
        cfg: ConfigType,
        key: PRNGKeyArray,
    ):
        """Initialize the LinOSS sequence mixer layer.

        Args:
            in_features: Input dimensionality.
            cfg: Configuration for the LinOSS sequence mixer.
            key: JAX random key for initialization.
        """
        B_key, C_key, D_key, key = jr.split(key, 4)

        if cfg.damping:
            if cfg.initialization == "RT":
                self.A_diag, self.G_diag, self.steps = _init_damped_linoss_rt(
                    key, cfg.discretization, cfg.state_dim, cfg.r_min, 1.0, 0.0, cfg.theta_max
                )
            elif cfg.initialization == "AG":
                self.A_diag, self.G_diag, self.steps = _init_damped_linoss_ag(
                    key, cfg.state_dim, 0.0, 1.0, 0.0, 1.0
                )
            else:
                raise NotImplementedError(f"Initialization {self.initialization} not implemented")
        else:
            self.A_diag, self.G_diag, self.steps = _init_linoss(key, cfg.state_dim)

        self.B = _simple_uniform_init(
            B_key,
            shape=(cfg.state_dim, in_features, 2),
            std=1.0 / math.sqrt(in_features),
        )
        self.C = _simple_uniform_init(
            C_key,
            shape=(in_features, cfg.state_dim, 2),
            std=1.0 / math.sqrt(cfg.state_dim),
        )
        self.D = normal(stddev=1.0)(D_key, (in_features,))

        self.discretization = cfg.discretization
        self.damping = cfg.damping

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LinOSS sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key for initialization.

        Returns:
            The output of the LinOSS sequence mixer.
        """
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        if self.damping:
            # Project
            A_diag, G_diag, steps = _project_agstep(
                self.discretization, self.A_diag, self.G_diag, self.steps
            )

            # Recurrence
            if self.discretization == "IMEX":
                ys = _apply_damped_linoss_imex(A_diag, G_diag, B_complex, x, steps)
            elif self.discretization == "IMEX2":
                ys = _apply_damped_linoss_imex2(A_diag, G_diag, B_complex, x, steps)
            elif self.discretization == "IM":
                ys = _apply_damped_linoss_im(A_diag, G_diag, B_complex, x, steps)
            elif self.discretization == "EX":
                ys = _apply_damped_linoss_ex(A_diag, G_diag, B_complex, x, steps)
            else:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(
                        self.discretization, self.damping
                    )
                )
        else:
            # Project
            steps = nn.sigmoid(self.steps)
            A_diag = nn.relu(self.A_diag)

            # Recurrence
            if self.discretization == "IMEX":
                ys = _apply_linoss_imex(A_diag, B_complex, x, steps)
            elif self.discretization == "IM":
                ys = _apply_linoss_im(A_diag, B_complex, x, steps)
            else:
                raise NotImplementedError(
                    "Discretization {} and damping = {} not implemented".format(
                        self.discretization, self.damping
                    )
                )

        # Apply SequenceMixer Output Operations Cx + Du
        Cy = jax.vmap(lambda x: (C_complex @ x).real)(ys)
        Du = jax.vmap(lambda u: self.D * u)(x)
        xs = Cy + Du

        return xs


def _simple_uniform_init(rng, shape, std=1.0):
    """Simple uniform initialization.

    This function initializes the weights of a linear layer using a simple uniform distribution.

    Args:
        rng: JAX random key for initialization.
        shape: Shape of the weights.
        std: Standard deviation of the weight initialization.

    Returns:
        Weights initialized using a simple uniform distribution.
    """
    weights = random.uniform(rng, shape) * 2.0 * std - std
    return weights


def _init_linoss(rng, state_dim):
    A_key, step_key = jr.split(rng, 2)
    A_diag = random.uniform(A_key, shape=(state_dim,))
    steps = normal(stddev=0.5)(step_key, (state_dim,))
    return A_diag, None, steps


def _init_damped_linoss_rt(
    rng: PRNGKeyArray,
    discretization: Literal["IMEX", "IMEX2", "IM", "EX"],
    state_dim: int,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
):
    # Solve symbolically
    a, g, step, lam1, lam2 = sp.symbols("a g step lam1 lam2")

    # Characteristic recurrence for 1 decoupled 2x2 system
    if discretization == "IMEX":
        M_i = sp.Matrix(
            [
                [1 / (1 + step * g), -a * step / (1 + step * g)],
                [step / (1 + step * g), 1 - a * step**2 / (1 + step * g)],
            ]
        )
    elif discretization == "IMEX2":
        M_i = sp.Matrix([[1 - step * g, -a * step], [step * (1 - step * g), 1 - step**2 * a]])
    elif discretization == "IM":
        M_i = sp.Matrix(
            [
                [1 / (1 + step * g + step**2 * a), -a * step / (1 + step * g + step**2 * a)],
                [
                    step / (1 + step * g + step**2 * a),
                    (1 + step * g) / (1 + step * g + step**2 * a),
                ],
            ]
        )
    elif discretization == "EX":
        M_i = sp.Matrix([[1 - step * g, -step * a], [step, 1]])
    else:
        raise ValueError

    # Eigenvalue pair expressions
    eigs = list(M_i.eigenvals().keys())
    eqs = [sp.Eq(eigs[0], lam1), sp.Eq(eigs[1], lam2)]
    sol = sp.solve(eqs, (a, g))[0]
    f = sp.lambdify((lam1, lam2, step), sol, "numpy")

    # Sample timesteps
    mag_key, arg_key, step_key = jr.split(rng, 3)
    step_vals = normal(stddev=0.5)(step_key, (state_dim,))
    step_sigmoid = nn.sigmoid(step_vals)

    # Sample eigenvalues in ring
    mag = jnp.sqrt(jr.uniform(mag_key, shape=(state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
    arg = jr.uniform(arg_key, shape=(state_dim,)) * (theta_max - theta_min) + theta_min
    lam1_vals = mag * jnp.cos(arg) + 1j * mag * jnp.sin(arg)
    lam2_vals = mag * jnp.cos(arg) - 1j * mag * jnp.sin(arg)

    # Convert to (A, G) representation
    a_vals, g_vals = f(lam1_vals, lam2_vals, step_sigmoid)

    # Cast to real (imag part is nonzero, ~machine precision)
    a_vals = a_vals.real
    g_vals = g_vals.real

    return a_vals, g_vals, step_vals


def _init_damped_linoss_ag(
    rng: PRNGKeyArray,
    state_dim: int,
    A_min: float,
    A_max: float,
    G_min: float,
    G_max: float,
):
    A_key, G_key, step_key = jr.split(rng, 3)
    A_diag = A_min + random.uniform(A_key, shape=(state_dim,)) * (A_max - A_min)
    G_diag = G_min + random.uniform(G_key, shape=(state_dim,)) * (G_max - G_min)
    steps = normal(stddev=0.5)(step_key, (state_dim,))
    return A_diag, G_diag, steps


def _project_agstep(discretization, A_diag, G_diag, step):
    step = nn.sigmoid(step)

    if discretization == "IMEX":
        G_diag = nn.relu(G_diag)
        A_low = (2 + step * G_diag - 2 * jnp.sqrt(1 + step * G_diag)) / jnp.maximum(step**2, 1e-6)
        A_high = (2 + step * G_diag + 2 * jnp.sqrt(1 + step * G_diag)) / jnp.maximum(step**2, 1e-6)
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
    elif discretization == "IMEX2":
        G_diag = nn.relu(G_diag)
        A_low = (2 - step * G_diag - 2 * jnp.sqrt(1 - step * G_diag)) / jnp.maximum(step**2, 1e-6)
        A_high = (2 - step * G_diag + 2 * jnp.sqrt(1 - step * G_diag)) / jnp.maximum(step**2, 1e-6)
        A_diag = A_low + nn.relu(A_diag - A_low) - nn.relu(A_diag - A_high)
    elif discretization == "IM":
        G_low = -step * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)
        A_low = 1 / 4 * G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)
    elif discretization == "EX":
        G_low = step * A_diag
        G_diag = G_low + nn.relu(G_diag - G_low)
        A_low = 1 / 4 * G_diag**2
        A_diag = A_low + nn.relu(A_diag - A_low)

    return A_diag, G_diag, step


# Parallel scan operations
@jax.vmap
def _binary_operator(q_i, q_j):  # noqa: N802
    """Binary operator for parallel scan of linear recurrence.

    This function implements the binary operator for the parallel scan of the linear recurrence.

    Args:
        q_i: Tuple containing A_i and b_i at position i.
        q_j: Tuple containing A_j and b_j at position j.

    Returns:
        The binary operator applied to the input.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def _apply_linoss_im(A_diag, B, x, step):  # noqa: N802
    """Compute the LinOSS-IM sequence mixer output.

    This function computes the output of the LinOSS-IM sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the LinOSS-IM sequence mixer at a specific time step.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_12 = -1.0 * step * A_diag * schur_comp
    M_21 = step * schur_comp
    M_22 = schur_comp

    M = jnp.concatenate([M_11, M_12, M_21, M_22])

    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = M_11 * Bu_elements * step
    F2 = M_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _apply_linoss_imex(A_diag, B, x, step):  # noqa: N802
    """Compute the LinOSS-IMEX sequence mixer output.

    This function computes the output of the LinOSS-IMEX sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the LinOSS-IMEX sequence mixer.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    M = jnp.concatenate([A_, B_, C_, D_])

    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _apply_damped_linoss_imex(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX sequence mixer output.

    This function computes the output of the Damped LinOSS-IMEX sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the Damped LinOSS-IMEX sequence mixer.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = Identity - step**2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _apply_damped_linoss_imex2(A_diag, G_diag, B, x, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX2 sequence mixer output.

    This function computes the output of the Damped LinOSS-IMEX2 sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the Damped LinOSS-IMEX2 sequence mixer.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    Identity = jnp.ones_like(A_diag)
    M_11 = Identity - step * G_diag
    M_12 = -step * A_diag
    M_21 = step * (Identity - step * G_diag)
    M_22 = Identity - step**2 * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = step * Bu_elements
    F2 = step**2 * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _apply_damped_linoss_im(A_diag, G_diag, B, x, step):
    """Compute the Damped LinOSS-IM sequence mixer output.

    This function computes the output of the Damped LinOSS-IM sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the Damped LinOSS-IM sequence mixer.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag + step**2 * A_diag
    M_11 = 1.0 / S
    M_12 = -step * A_diag / S
    M_21 = step / S
    M_22 = (Identity + step * G_diag) / S

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _apply_damped_linoss_ex(A_diag, G_diag, B, x, step):
    """Compute the Damped LinOSS-EX sequence mixer output.

    This function computes the output of the Damped LinOSS-EX sequence mixer.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        B: Input matrix.
        x: Input sequence of features.
        step: Discretization time-step.

    Returns:
        The output of the Damped LinOSS-EX sequence mixer.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(x)

    Identity = jnp.ones_like(A_diag)
    M_11 = 1.0 - step * G_diag
    M_12 = -step * A_diag
    M_21 = step
    M_22 = Identity

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((x.shape[0], 4 * A_diag.shape[0]))

    F1 = step * Bu_elements
    F2 = jnp.zeros_like(F1)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys
