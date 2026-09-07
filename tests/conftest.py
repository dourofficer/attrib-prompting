import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The representation-based baselines live in a second source root whose
# hyphenated name keeps it from colliding with ``baselines/``; its packages
# (``oat``, ``rb_shared``) import by plain name from there.
RB_ROOT = REPO_ROOT / "baselines-rp"
if str(RB_ROOT) not in sys.path:
    sys.path.insert(0, str(RB_ROOT))
