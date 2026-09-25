"""Write the synthetic reviews used by the tests to a CSV (for CI and for trying the API
without the real dataset): python mlops/make_sample.py data/processed/sample_reviews.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests'))
from conftest import make_reviews  # noqa: E402

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
make_reviews().to_csv(out, index=False)
print(f'synthetic reviews written to {out}')
