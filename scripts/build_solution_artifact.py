"""Regenerate the backup participant artifact from the reference solution."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solutions.policy_solution import build_customer_scores
from src.workshop_data import load_observed_data


def main() -> None:
    scores = build_customer_scores(load_observed_data())
    out = PROJECT_ROOT / "artifacts" / "customer_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    print(f"Backup artifact: {len(scores):,} rows -> {out}")


if __name__ == "__main__":
    main()
