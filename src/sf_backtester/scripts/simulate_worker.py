"""SLURM afterok finalizer; may also be rerun explicitly after a failed simulation."""

import argparse

from sf_backtester.execution import finalize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    finalize_run(args.manifest)


if __name__ == "__main__":
    main()
