"""Exact even/odd interchange spectra of the released SF14.1 integer head."""

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

MASKS = np.arange(256)
ORDER = np.array([int(m).bit_count() for m in MASKS])
PAIR = MASKS[ORDER == 2]


def transform(data, subset_sum=False):
    """Walsh sums, or subset zeta sums, along the last axis."""
    x = np.ascontiguousarray(data).copy()
    for i in range(8):
        t = x.reshape(-1, 256 >> (i + 1), 2, 1 << i)
        left, right = t[:, :, 0, :], t[:, :, 1, :]
        if subset_sum:
            right += left
        else:
            old = left.copy()
            left += right
            right[:] = old - right
    return x


def compile_kernel(root):
    lib = root / "circuits/cube.so"
    lib.parent.mkdir(exist_ok=True)
    temporary = lib.with_suffix(f".{os.getpid()}.so")
    subprocess.run(
        [
            "c++",
            "-O3",
            "-std=c++20",
            "-march=native",
            "-shared",
            "-fPIC",
            str(Path(__file__).with_name("cube.cc")),
            "-o",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(lib)
    return lib


class Cube:
    def __init__(self, path, weights):
        self.b, w, self.out_bias, self.out = weights
        self.w = np.ascontiguousarray(w)
        self.fn = ctypes.CDLL(str(path.resolve())).cube
        self.fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5
        self.fn.restype = None

    def forward(self, h, smooth=False):
        h = np.ascontiguousarray(h, dtype=np.int32)
        cells = np.empty((len(h), 256, 32), dtype=np.uint8)
        unrounded = np.empty_like(cells, dtype=np.uint16) if smooth else None
        self.fn(
            len(h),
            h.ctypes.data,
            self.b.ctypes.data,
            self.w.ctypes.data,
            cells.ctypes.data,
            unrounded.ctypes.data if smooth else None,
        )
        return cells, unrounded

    def verify(self, h):
        # Independent direct evaluation of every mask on reference pairs.
        native, unrounded = self.forward(h, True)
        for m in MASKS:
            z = np.where((m >> np.arange(8)) & 1, h[:, 1], h[:, 0])
            p = np.clip(z @ self.w.T + self.b, 0, 127 * 64)
            assert np.array_equal(unrounded[:, m], p)
            assert np.array_equal(native[:, m], p >> 6)
        return True
