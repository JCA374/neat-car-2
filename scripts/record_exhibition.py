"""
Record every championship car on the exhibition tracks (random/chosen-track races on the Championship page).

    python scripts/record_exhibition.py            # all raced cars, only seeds not recorded yet
    python scripts/record_exhibition.py neat16

scripts/run_step.py does this automatically for new championship cars.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carracing.exhibition import record  # noqa: E402
from carracing.steps import all_steps  # noqa: E402


def main() -> None:
    wanted = set(sys.argv[1:])
    for step in all_steps():
        if not step.championship or (wanted and step.id not in wanted):
            continue
        driven = record(step)
        print(f"{step.id}: " + (", ".join(f"{cat} {n} tracks" for cat, n in driven.items()) or "nothing new to drive"),
              flush=True)


if __name__ == "__main__":
    main()
