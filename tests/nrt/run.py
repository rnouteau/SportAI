"""Run NRT suites without pytest (same code path as ``tests/nrt/tennis/test_tennis_nrt.py``).

python -m tests.nrt.run                          # every suite, compare to the baselines
python -m tests.nrt.run tennis_players           # one suite
python -m tests.nrt.run tennis_players --update-baseline
"""

from __future__ import annotations

import argparse
import sys

from tests.nrt.harness import run_suite
from tests.nrt.suites import SUITES, SUITES_BY_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suites", nargs="*", help=f"default: all ({', '.join(SUITES_BY_ID)})")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    unknown = [s for s in args.suites if s not in SUITES_BY_ID]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}")

    failed = False
    for suite in [SUITES_BY_ID[s] for s in args.suites] or list(SUITES):
        outcome = run_suite(suite, update_baseline=args.update_baseline)
        if outcome.baseline_written:
            print(f"{suite.id}: baseline written -> review it, then commit it")
        elif outcome.problems:
            failed = True
            print(f"{suite.id}: DIFFERS from the baseline")
            print("\n".join(f"  {problem}" for problem in outcome.problems))
        else:
            print(f"{suite.id}: same as the baseline")
        print(f"  report: {suite.report_dir / 'report.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
