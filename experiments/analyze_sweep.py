"""Analyze results from HTCondor hyperparameter sweep.

Aggregates metrics from all jobs and identifies best configurations.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_job_results(sweep_dir: Path) -> list[dict[str, Any]]:
    """Load results from all jobs in a sweep directory.

    Args:
        sweep_dir: Path to sweep directory

    Returns:
        List of dictionaries containing job results
    """
    results = []

    # Iterate through all job subdirectories
    for job_dir in sorted(sweep_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        # Load metrics
        metrics_file = job_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics file found in {job_dir}")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Load config
        config_file = job_dir / ".hydra" / "config.yaml"
        hydra_config = {}
        if config_file.exists():
            import yaml

            with open(config_file) as f:
                hydra_config = yaml.safe_load(f)

        # Combine results
        result = {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "final_test_accuracy": metrics.get("final_test_accuracy"),
            "best_test_accuracy": metrics.get("best_test_accuracy"),
            "best_test_accuracy_step": metrics.get("best_test_accuracy_step"),
        }

        # Add hyperparameters from config
        if hydra_config:
            # Flatten nested config for easier analysis
            def flatten_dict(d, parent_key="", sep="_"):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            flat_config = flatten_dict(hydra_config)
            result.update(flat_config)

        results.append(result)

    return results


def get_config_columns(df: pd.DataFrame) -> list[str]:
    """Identify columns that define unique configurations (excluding seed)."""
    config_cols = []
    for col in df.columns:
        if col.startswith(("training_", "model_", "dataset_")):
            # Exclude seed column
            if "seed" not in col.lower():
                config_cols.append(col)
    return config_cols


def group_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Group results by configuration (across seeds) and compute statistics."""
    config_cols = get_config_columns(df)

    if not config_cols:
        print("Warning: No configuration columns found")
        return df

    # Group by configuration and compute statistics
    agg_dict = {
        "best_test_accuracy": ["mean", "std", "min", "max", "count"],
        "final_test_accuracy": ["mean", "std", "min", "max"],
        "best_test_accuracy_step": ["mean", "std"],
    }

    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    grouped = df.groupby(config_cols, dropna=False).agg(agg_dict)

    # Flatten column names
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Rename for clarity
    rename_dict = {
        "best_test_accuracy_mean": "mean_accuracy",
        "best_test_accuracy_std": "std_accuracy",
        "best_test_accuracy_min": "min_accuracy",
        "best_test_accuracy_max": "max_accuracy",
        "best_test_accuracy_count": "num_seeds",
    }
    grouped = grouped.rename(
        columns={k: v for k, v in rename_dict.items() if k in grouped.columns}
    )

    return grouped


def print_summary(df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
    """Print summary statistics of sweep results."""
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)

    print(f"\nTotal jobs: {len(df)}")
    print(f"Unique configurations: {len(grouped_df)}")

    if "mean_accuracy" in grouped_df.columns:
        print("\nConfiguration Accuracy Statistics (averaged across seeds):")
        print(f"  Mean:   {grouped_df['mean_accuracy'].mean():.4f}")
        print(f"  Std:    {grouped_df['mean_accuracy'].std():.4f}")
        print(f"  Min:    {grouped_df['mean_accuracy'].min():.4f}")
        print(f"  Max:    {grouped_df['mean_accuracy'].max():.4f}")
        print(f"  Median: {grouped_df['mean_accuracy'].median():.4f}")

    if "best_test_accuracy" in df.columns:
        print("\nIndividual Run Statistics:")
        print(f"  Mean:   {df['best_test_accuracy'].mean():.4f}")
        print(f"  Std:    {df['best_test_accuracy'].std():.4f}")
        print(f"  Min:    {df['best_test_accuracy'].min():.4f}")
        print(f"  Max:    {df['best_test_accuracy'].max():.4f}")


def print_top_configs(grouped_df: pd.DataFrame, top_k: int = 10) -> None:
    """Print top performing configurations (averaged across seeds)."""
    print("\n" + "=" * 80)
    print(f"TOP {top_k} CONFIGURATIONS (averaged across seeds)")
    print("=" * 80)

    if "mean_accuracy" not in grouped_df.columns:
        print("No accuracy data available")
        return

    df_sorted = grouped_df.sort_values("mean_accuracy", ascending=False)

    # Identify columns that vary (hyperparameters)
    config_cols = get_config_columns(grouped_df)
    varying_cols = []
    for col in config_cols:
        if grouped_df[col].nunique() > 1:
            varying_cols.append(col)

    print(f"\nVarying parameters: {', '.join(varying_cols)}\n")

    for i, (idx, row) in enumerate(df_sorted.head(top_k).iterrows(), 1):
        mean_acc = row["mean_accuracy"]
        std_acc = row.get("std_accuracy", 0)
        num_seeds = int(row.get("num_seeds", 1))

        print(f"{i}. Accuracy: {mean_acc:.4f} ± {std_acc:.4f} ({num_seeds} seeds)")

        for col in varying_cols:
            if col in row:
                print(f"   {col}: {row[col]}")

        if "min_accuracy" in row and "max_accuracy" in row:
            print(f"   Range: [{row['min_accuracy']:.4f}, {row['max_accuracy']:.4f}]")

        print()


def print_seed_details(df: pd.DataFrame, grouped_df: pd.DataFrame, top_k: int = -1) -> None:
    """Print individual seed results for top configurations."""
    if top_k == -1:
        top_k = len(grouped_df)

    print("\n" + "=" * 80)
    print(f"SEED-LEVEL DETAILS FOR TOP {top_k} CONFIGURATIONS")
    print("=" * 80)

    if "mean_accuracy" not in grouped_df.columns:
        return

    config_cols = get_config_columns(df)
    df_sorted = grouped_df.sort_values("mean_accuracy", ascending=False)

    for i, (idx, config_row) in enumerate(df_sorted.head(top_k).iterrows(), 1):
        print(
            f"\n{i}. Configuration (Mean: {config_row['mean_accuracy']:.4f}"
            " ± {config_row.get('std_accuracy', 0):.4f})"
        )

        # Print config
        varying_cols = [col for col in config_cols if grouped_df[col].nunique() > 1]
        for col in varying_cols:
            if col in config_row:
                print(f"   {col}: {config_row[col]}")

        # Find all runs with this configuration
        mask = pd.Series([True] * len(df))
        for col in config_cols:
            if col in config_row:
                mask &= df[col] == config_row[col]

        matching_runs = df[mask].sort_values("best_test_accuracy", ascending=False)

        print("\n   Individual runs:")
        for _, run in matching_runs.iterrows():
            seed = run.get("training_seed", run.get("seed", "unknown"))
            acc = run.get("best_test_accuracy", "N/A")
            step = run.get("best_test_accuracy_step", "N/A")
            print(f"     Seed {seed}: {acc:.4f} (step {step})")


def analyze_parameter_impact(grouped_df: pd.DataFrame) -> None:
    """Analyze the impact of each hyperparameter on performance."""
    print("\n" + "=" * 80)
    print("PARAMETER IMPACT ANALYSIS")
    print("=" * 80)

    if "mean_accuracy" not in grouped_df.columns:
        print("No accuracy data available")
        return

    # Find varying hyperparameters
    config_cols = get_config_columns(grouped_df)
    varying_params = [col for col in config_cols if grouped_df[col].nunique() > 1]

    if not varying_params:
        print("No varying hyperparameters found")
        return

    print(f"\nAnalyzing {len(varying_params)} varying parameters:\n")

    for param in varying_params:
        print(f"\n{param}:")
        grouped = grouped_df.groupby(param)["mean_accuracy"].agg(["mean", "std", "count"])
        grouped = grouped.sort_values("mean", ascending=False)
        print(grouped.to_string())


def main():
    """Analyze parameter sweep output."""
    parser = argparse.ArgumentParser(description="Analyze HTCondor hyperparameter sweep results")

    parser.add_argument(
        "sweep_dir", type=str, help="Path to sweep directory (e.g., outputs/sweep_12345)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top configurations to display (default: 10)",
    )

    parser.add_argument("--output", type=str, help="Save aggregated results to CSV file")

    parser.add_argument(
        "--output-all", type=str, help="Save all individual run results to CSV file"
    )

    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip parameter impact analysis"
    )

    parser.add_argument(
        "--seed-details",
        type=int,
        default=-1,
        help="Number of top configs to show seed-level details for (default: -1, i.e. all)",
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: Sweep directory {sweep_dir} not found")
        return

    print(f"Loading results from {sweep_dir}...")
    results = load_job_results(sweep_dir)

    if not results:
        print("No results found")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Group by configuration
    grouped_df = group_by_config(df)

    # Print summary
    print_summary(df, grouped_df)

    # Print top configurations
    print_top_configs(grouped_df, top_k=args.top_k)

    # Print seed-level details for top configs
    if args.seed_details != 0:
        print_seed_details(df, grouped_df, top_k=args.seed_details)

    # Analyze parameter impact
    if not args.no_analysis:
        analyze_parameter_impact(grouped_df)

    # Save results
    if args.output:
        output_path = Path(args.output)
        grouped_df.to_csv(output_path, index=False)
        print(f"\nAggregated results saved to {output_path}")

    if args.output_all:
        output_path = Path(args.output_all)
        df.to_csv(output_path, index=False)
        print(f"All individual run results saved to {output_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
