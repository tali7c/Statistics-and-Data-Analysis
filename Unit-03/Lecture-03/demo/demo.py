from __future__ import annotations

import sys
from pathlib import Path

lecture_dir = Path(__file__).resolve().parents[1]
subject_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(subject_root / "_shared"))

from demo_runner import run

run("paired_ttest", lecture_dir)
