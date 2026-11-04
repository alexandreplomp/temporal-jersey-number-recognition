"""Explain a trained run: one val sequence per number, with checks that the maps are faithful.
Example: uv run explain.py results/runs/averaged_30

Writes prediction.png and explain.json into the run folder, as train.py does at the end
of a run, so older runs can be explained too. See jersey/explain.py for what they show.
"""

from pathlib import Path
import argparse
import json

from jersey.data import JerseyDataset, filter_numbers, run_splits
from jersey.explain import explain_run
from jersey.model import load_run
from train import pick_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run folder with best.pt")
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, config, _ = load_run(args.run, device)
    splits = run_splits(args.run, config["splits"])
    val_ids = filter_numbers(splits["val"], tuple(config["exclude"]))
    dataset = JerseyDataset(config["data"], val_ids, max_frames=config["max_frames"])
    summary = explain_run(model, dataset, config, args.run, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
