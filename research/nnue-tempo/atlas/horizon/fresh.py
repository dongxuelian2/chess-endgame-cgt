"""Extract one quiet comparison per game from three later monthly archives."""

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

from games import batches, extract
from validate import init, played


def process(batch):
    return played(extract(batch))


def main(root):
    with (
        ProcessPoolExecutor(8, initializer=init, initargs=(root,)) as pool,
        gzip.open(root / "atlas/horizon/played.gz", "wt") as out,
    ):
        for group in batched(batches(root, range(10, 13)), 64):
            for records in pool.map(process, group):
                for r in records:
                    out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
