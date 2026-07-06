"""Step 3B: convert generated CP-2 fit_dict to uncertainty scores."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from env_utils import load_quickstart_env


QUICK_START = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICK_START.parent


def main() -> None:
    load_quickstart_env()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    target_dir_name = sys.argv[1] if len(sys.argv) > 1 else "generated"
    
    cmd = [
        sys.executable,
        "-m",
        "matu.fit_to_uncertainty",
        "--fit_dict",
        str(QUICK_START / target_dir_name / "results" / "fit_dict_generated.pkl"),
        "--out",
        str(QUICK_START / target_dir_name / "results" / "uncertainty_generated.pkl"),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
