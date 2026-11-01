from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATOS = ROOT / 'data'          # una subcarpeta por base (edb, ltst, ...)
DATA = DATOS / 'edb'
CONFIG = ROOT / 'config'
RESULTS = ROOT / 'results'
