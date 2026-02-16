"""Generates HTCondor job configurations for hyperparameter sweeps.

Uses Hydra's sweep syntax to generate all parameter combinations.
"""

import argparse
import itertools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate HTCondor jobs for hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_condor_jobs.py \\
      --sweep "training.learning_rate=0.001,0.0001" \\
      --fixed "training.steps=5000" \\
      --fixed "seed=42"
        """,
    )

    parser.add_argument(
        "--sweep",
        action="append",
        required=True,
        help='Sweep parameter in format "key=val1,val2,val3"',
    )

    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        help='Fixed parameter in format "key=value" (applied to all jobs)',
    )

    parser.add_argument(
        "--submit-file",
        default="experiments/condor_sweep.sub",
        help="HTCondor submit file template (default: condor_sweep.sub)",
    )

    parser.add_argument(
        "--output-dir",
        default="experiments/sweeps",
        help="Directory for job configuration files (default: sweep_configs)",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment directory name. Defaults to current date and time.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating files",
    )

    return parser.parse_args()


def parse_sweep_config(sweep_params: list[str]) -> dict[str, list[Any]]:
    """Parse Hydra-style sweep parameters.

    Args:
        sweep_params: List of sweep parameters in format "key=val1,val2,val3"

    Returns:
        Dictionary mapping parameter names to lists of values
    """
    param_dict = {}
    for param in sweep_params:
        if "=" not in param:
            raise ValueError(f"Invalid parameter format: {param}. Expected 'key=val1,val2,val3'")

        key, values_str = param.split("=", 1)

        # Split comma-separated values
        values = [v.strip() for v in values_str.split(",")]
        param_dict[key] = values

    return param_dict


def generate_param_combinations(param_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations of parameters.

    Args:
        param_dict: Dictionary mapping parameter names to lists of values

    Returns:
        List of dictionaries, each representing one parameter combination
    """
    keys = list(param_dict.keys())
    values = list(param_dict.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def format_hydra_overrides(params: dict[str, Any]) -> str:
    """Format parameter dictionary as Hydra command-line overrides.

    Args:
        params: Dictionary of parameter names and values

    Returns:
        String of space-separated Hydra overrides
    """
    overrides = []
    for key, value in params.items():
        overrides.append(f"{key}={value}")

    return " ".join(overrides)


def main():
    """Generate HTCondor jobs for hyperparameter sweep."""
    args = parse_arguments()

    # Parse sweep parameters
    print("Parsing sweep parameters...")
    param_dict = parse_sweep_config(args.sweep)

    # Generate all combinations
    combinations = generate_param_combinations(param_dict)
    num_jobs = len(combinations)
    print(f"\nTotal number of jobs: {num_jobs}")

    if args.dry_run:
        print("\n[DRY RUN] Would generate the following configurations:")
        for i, params in enumerate(combinations[:5]):  # Show first 5
            overrides = format_hydra_overrides(params)
            if args.fixed:
                overrides += " " + " ".join(args.fixed)
            print(f"  Job {i}: {overrides}")
        if num_jobs > 5:
            print(f"  ... and {num_jobs - 5} more jobs")
        return

    # Create output directory (and subdirectories)
    if not args.experiment_name:
        args.experiment_name = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
    output_dir = Path(args.output_dir) / args.experiment_name
    if output_dir.exists():
        print(f"\nRemoving existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "outputs").mkdir()
    (output_dir / "configs").mkdir()
    print(f"\nCreated output directory in {output_dir}/...")

    # Generate configuration files for each job
    for i, params in enumerate(combinations):
        config_file = output_dir / "configs" / f"config_{i}.txt"

        # Combine sweep parameters with fixed parameters
        overrides = format_hydra_overrides(params)
        if args.fixed:
            overrides += " " + " ".join(args.fixed)

        with open(config_file, "w") as f:
            f.write(overrides)

    print(f"Generated {num_jobs} configuration files")

    # Update submit file with NUM_JOBS
    submit_file = Path(args.submit_file)
    if not submit_file.exists():
        print(f"\nError: Submit file {submit_file} not found")
        return

    # Read submit file and update NUM_JOBS
    with open(submit_file) as f:
        submit_content = f.read()

    # Replace queue line with actual number of jobs
    # and replace run argument with multirun folder
    submit_content = submit_content.replace("queue $(NUM_JOBS)", f"queue {num_jobs}")
    submit_content = submit_content.replace("$(MULTIRUN_NAME)", args.experiment_name)

    # Write to temporary submit file
    temp_submit_file = submit_file.with_suffix(".sub.generated")
    with open(temp_submit_file, "w") as f:
        f.write(submit_content)

    print(f"\nGenerated submit file: {temp_submit_file}")
    print("\nTo submit jobs, run:")
    print(f"  condor_submit {temp_submit_file}")
    print("\nOr re-run with --submit flag")


if __name__ == "__main__":
    main()
