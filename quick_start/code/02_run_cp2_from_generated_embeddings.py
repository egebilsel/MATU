"""Step 2: run MATU/CP-2 on embeddings generated in Step 1."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from env_utils import load_quickstart_env


QUICK_START = Path(__file__).resolve().parents[1]
REPO_ROOT = QUICK_START.parent
EMB_DIR = QUICK_START / "generated" / "embeddings"
OUT_DIR = QUICK_START / "generated" / "results"


def main() -> None:
    load_quickstart_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "matu.cp2_matu",
        "--embeddings",
        str(EMB_DIR / "user_embedding_matrices.pkl"),
        str(EMB_DIR / "assistant_embedding_matrices.pkl"),
        "--out",
        str(OUT_DIR / "matu_scores.pkl"),
        "--legacy_fit_out",
        str(OUT_DIR / "fit_dict_generated.pkl"),
        "--max_rank",
        os.environ.get("MATU_MAX_RANK", "50"),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()
