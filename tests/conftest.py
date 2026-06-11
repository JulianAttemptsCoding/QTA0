import sys
from pathlib import Path

# make `import tactic` work from the src layout without an install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
