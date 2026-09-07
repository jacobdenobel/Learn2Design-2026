#!/usr/bin/env python3
"""Evaluate one saved UIFO dataset entry with the competition objective."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax.numpy as jnp
from loguru import logger

try:
    from dfbench import Objective
    from dfbench.problems import UIFOProblem
except ModuleNotFoundError as exc:
    raise SystemExit("Install the competition package first: pip install -e .") from exc

from dataset_utils import DATASET_PATH, entry_to_dict, find_entry_index, load_bounded_params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--index", type=int, default=0, help="Entry index to evaluate.")
    parser.add_argument("--hash", dest="unique_hash", help="Entry unique_hash. Overrides --index.")
    parser.add_argument("--max-time", type=float, default=300.0, help="Objective wall-clock budget in seconds.")
    args = parser.parse_args()

    index_arg = None if args.unique_hash else args.index

    with h5py.File(args.dataset, "r") as handle:
        entry_index = find_entry_index(handle, index=index_arg, unique_hash=args.unique_hash)
        entry = handle["entries"][entry_index]
        metadata = entry_to_dict(entry)
        params = load_bounded_params(handle, entry)
        n_frequencies = int(handle["frequency_values"].shape[0])

    problem = UIFOProblem(
        size=metadata["size"],
        topology=metadata["topology_string"],
        n_frequencies=n_frequencies,
    )
    objective = Objective(problem, max_evals=1, max_time=args.max_time, verbose=0)

    expected_params = int(objective.bounds.shape[-1])
    if params.shape[0] != expected_params:
        raise ValueError(
            f"Saved params have length {params.shape[0]}, but this UIFOProblem expects {expected_params}."
        )
    logger.info("warming up")
    objective.warmup_value()
    objective.start_logging()
    logger.info("computing loss")
    evaluated_loss = float(objective.value(jnp.asarray(params)))
    saved_loss = metadata["loss"]
    logger.info("done")
    logger.info(f"entry_index: {entry_index}")
    logger.info(f"unique_hash: {metadata['unique_hash']}")
    logger.info(f"topology_string: {metadata['topology_string']}")
    logger.info(f"size: {metadata['size']}")
    logger.info(f"n_frequencies: {n_frequencies}")
    logger.info(f"n_params: {params.shape[0]}")
    logger.info(f"saved_loss: {saved_loss:.12g}")
    logger.info(f"evaluated_loss: {evaluated_loss:.12g}")
    logger.info(f"absolute_difference: {abs(evaluated_loss - saved_loss):.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
