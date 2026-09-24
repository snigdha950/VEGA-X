"""One-command entry point for the RAFTS prototype and result bundle."""

import argparse
from pathlib import Path

from visualize import run_visualization


def main():
    parser = argparse.ArgumentParser(
        description="Run the single-replay RAFTS visual demo (not final learner selection)."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Optional TSRD .h5 path (positional form for convenience)",
    )
    parser.add_argument("--file", help="TSRD .h5 path")
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--min-pulses", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=int, default=250, help="Milliseconds per frame")
    parser.add_argument("--step", type=int, default=6, help="Slots advanced per frame")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--show", action="store_true", help="Open the live window")
    args = parser.parse_args()

    if args.dataset and args.file:
        parser.error("pass the dataset either positionally or with --file, not both")
    dataset = Path(args.file or args.dataset or "data/tsrd/config_0.h5")
    run_visualization(
        file=dataset,
        slots=args.slots,
        bands=args.bands,
        min_pulses=args.min_pulses,
        seed=args.seed,
        speed=args.speed,
        step=args.step,
        output_dir=args.output_dir,
        save_gif=not args.no_gif,
        show=args.show,
    )


if __name__ == "__main__":
    main()
