from __future__ import annotations

import argparse
from pathlib import Path

from spider.dashboard import build_probe_dashboard, write_dashboard_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static EXP002 QA explorer payload")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--step-250", type=Path, required=True)
    parser.add_argument("--step-1000", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_probe_dashboard(
        {
            "baseline": args.baseline,
            "step250": args.step_250,
            "step1000": args.step_1000,
        }
    )
    write_dashboard_json(payload, args.output)
    print(
        f"Wrote {payload['qa']['meta']['examples']} QA and "
        f"{payload['grounding']['meta']['examples']} grounding examples to {args.output}"
    )


if __name__ == "__main__":
    main()
