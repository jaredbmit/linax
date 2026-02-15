"""Training script for SSM models."""

import json
import logging
from pathlib import Path
from typing import Any

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import torch
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from omegaconf import DictConfig, OmegaConf

from linax.encoder import LinearEncoderConfig
from linax.heads.classification import ClassificationHeadConfig
from linax.models.linoss import LinOSSConfig
from linax.models.ssm import SSM
from linax.sequence_mixers import LinOSSSequenceMixerConfig

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value
    """
    # PyTorch seed
    torch.manual_seed(seed)

    # PyTorch CUDA seed (if GPU available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # NumPy seed (used by PyTorch dataloaders)
    import numpy as np

    np.random.seed(seed)


# Dataset registry for extensibility
DATASET_REGISTRY: dict[str, Any] = {
    "mnistseq": {
        "class": "datasets.MNISTSeq",
        "num_classes": 10,
        "input_dim": 4,
        "seq_length": 128,
    }
}


def load_dataset(
    dataset_name: str,
    root: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict[str, Any]]:
    """Load dataset and create dataloaders.

    Args:
        dataset_name: Name of the dataset to load
        root: Root directory for datasets
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes

    Returns:
        Tuple of (trainloader, testloader, dataset_info)
    """
    if dataset_name.lower() == "mnistseq":
        from linax.datasets.MNISTSeq import MNISTSeq

        train_dataset = MNISTSeq(root=root, train=True, download=True)
        test_dataset = MNISTSeq(root=root, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    dataset_info = DATASET_REGISTRY[dataset_name.lower()]
    return trainloader, testloader, dataset_info


def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
    """Cross entropy loss function."""
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


@eqx.filter_jit
def loss(
    model: SSM,
    x: Float[Array, "batch 128 4"],
    y: Int[Array, " batch"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Apply loss function to the model.

    Returns the cross entropy loss given x and y as well as the updated model state.
    """
    batch_keys = jax.random.split(key, x.shape[0])

    # this vmap parallelizes the model over the batch dimension (which is the first dimension).
    pred_y, model_state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, None),
    )(x, state, batch_keys)

    return cross_entropy(y, pred_y), model_state


@eqx.filter_jit
def compute_accuracy(
    model: SSM,
    x: Float[Array, "batch 128 4"],
    y: Int[Array, " batch"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """Computes the average accuracy on a batch."""
    batch_keys = jax.random.split(key, x.shape[0])

    pred_y, _ = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, None),
    )(x, state, batch_keys)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(
    model: SSM,
    testloader: torch.utils.data.DataLoader,
    state: eqx.nn.State,
    key: PRNGKeyArray,
):
    """Evaluates the model on the test dataset."""
    inference_model = eqx.tree_inference(model, value=True)
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(inference_model, x, y, state, key)[0]
        avg_acc += compute_accuracy(inference_model, x, y, state, key)
    return avg_loss / len(testloader), avg_acc / len(testloader)


@eqx.filter_jit
def make_step(
    model: SSM,
    loss_fn: callable,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Float[Array, "batch 128 4"],
    y: Int[Array, " batch"],
    state: eqx.nn.State,
    key: PRNGKeyArray,
):
    """Perform a single training step."""
    (loss_value, new_state), grads = loss_fn(model, x, y, state, key)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, new_state


def train(
    model: SSM,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    state: eqx.nn.State,
    key: PRNGKeyArray,
    output_dir: Path | None = None,
) -> tuple[SSM, dict]:
    """Trains the model on the training dataset.

    Args:
        model: The model to train
        trainloader: Training data loader
        testloader: Test data loader
        optim: Optimizer transformation
        steps: Number of training steps
        print_every: Evaluation frequency
        state: Model state
        key: JAX random key
        output_dir: Directory to save checkpoints (optional)

    Returns:
        Tuple of (trained_model, training_history)
    """
    history = {"steps": [], "train_loss": [], "test_accuracy": []}
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    loss_w_grad = eqx.filter_value_and_grad(loss, has_aux=True)

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    key, train_key = jax.random.split(key, 2)

    for step, (x, y) in zip(range(1, steps + 1), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss, new_state = make_step(
            model, loss_w_grad, optim, opt_state, x, y, state, train_key
        )

        if step % print_every == 0:
            test_loss, test_accuracy = evaluate(model, testloader, new_state, key)
            log.info(
                f"Step {step}: train_loss={train_loss.item():.6f}, "
                f"test_loss={test_loss.item():.6f}, test_accuracy={test_accuracy.item():.4f}"
            )
            history["steps"].append(int(step))
            history["train_loss"].append(float(train_loss.item()))
            history["test_accuracy"].append(float(test_accuracy.item()))
            if output_dir is not None:
                checkpoint_path = output_dir / f"checkpoint_step_{step}.eqx"
                eqx.tree_serialise_leaves(checkpoint_path, model)
                log.debug(f"Saved checkpoint to {checkpoint_path}")

    return model, history


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra configuration."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    # Get hydra output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")

    # Set random seeds for reproducibility
    log.info(f"Setting seed: {cfg.seed}")
    set_seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # Load dataset
    log.info(f"Loading dataset: {cfg.dataset.name}")
    data_dir = Path(__file__).resolve().parent.parent / "data_dir"
    log.info(f"Data directory: {data_dir}")
    trainloader, testloader, dataset_info = load_dataset(
        dataset_name=cfg.dataset.name,
        root=data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    log.info(f"Dataset info: {dataset_info}")

    # Build model based on configuration
    log.info(f"Building model: {cfg.model.type}")
    if cfg.model.type.lower() == "linoss":
        linoss_cfg = LinOSSConfig(
            num_blocks=cfg.model.num_blocks,
            encoder_config=LinearEncoderConfig(
                in_features=dataset_info["input_dim"],
                out_features=cfg.model.encoder.out_features,
            ),
            head_config=ClassificationHeadConfig(out_features=dataset_info["num_classes"]),
            sequence_mixer_config=LinOSSSequenceMixerConfig(
                state_dim=cfg.model.sequence_mixer.state_dim,
                damping=cfg.model.sequence_mixer.damping,
                discretization=cfg.model.sequence_mixer.discretization,
                initialization=cfg.model.sequence_mixer.initialization,
            ),
        )

        key, subkey = jax.random.split(key, 2)
        model = linoss_cfg.build(key=subkey)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    state = eqx.nn.State(model=model)
    log.info("Model built successfully")

    # Setup optimizer
    optim = optax.adamw(cfg.training.learning_rate)
    log.info(f"Using AdamW optimizer with learning rate {cfg.training.learning_rate}")

    # Train
    log.info(f"Starting training for {cfg.training.steps} steps")
    model, history = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optim=optim,
        steps=cfg.training.steps,
        print_every=cfg.training.print_every,
        state=state,
        key=key,
        output_dir=output_dir,
    )
    log.info("Training completed")

    # Save final model
    final_model_path = output_dir / "final_model.eqx"
    eqx.tree_serialise_leaves(final_model_path, model)
    log.info(f"Final model saved to {final_model_path}")

    # Save training metrics
    best_acc = max(history["test_accuracy"]) if history["test_accuracy"] else None
    best_acc_step = (
        history["steps"][history["test_accuracy"].index(best_acc)] if best_acc else None
    )
    metrics_path = output_dir / "metrics.json"
    metrics_data = {
        "final_test_accuracy": history["test_accuracy"][-1] if history["test_accuracy"] else None,
        "best_test_accuracy": float(best_acc) if best_acc else None,
        "best_test_accuracy_step": int(best_acc_step) if best_acc_step else None,
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    log.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
