"""Evaluate SF18 on both turns of the complete KPK domain."""

import gzip
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from modern import readout

root = Path(sys.argv[1])
counts = Counter()
with gzip.open(root / "kpk-census.gz") as f:
    data = map(json.loads, f)
    while batch := list(islice(data, 10000)):
        rows = [{"board": b, "wdl": w} for b, w in batch]
        score = readout(rows, root, False)[:, :, :2].sum(axis=(1, 2))
        for r, s in zip(rows, score):
            counts[sum(r["wdl"]), int(s > 0) - int(s < 0)] += 1
print(json.dumps([[*k, v] for k, v in sorted(counts.items())]))
