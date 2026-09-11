#!/usr/bin/env python3
"""Import and exercise the pinned numerical stack, reporting the CPU it ran on.

Run via scripts/check-pins.sh or scripts/smoke-image.sh. The flushed line before
each step is the diagnosis, since SIGILL kills silently. --require-no-lse
refuses a core with LSE: a dropped QEMU_CPU falls back to -cpu max, which has
it, and the emulated-Pi run would pass having modelled nothing.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import tempfile
from pathlib import Path

# Not /proc/cpuinfo: this is the bit libgcc dispatches on, and qemu-user < 8.2 lies.
AT_HWCAP = 16
HWCAP_ATOMICS = 1 << 8


def step(message: str) -> None:
    print(f"  ... {message}", flush=True)


def has_lse() -> bool | None:
    """True or False from AT_HWCAP; None where getauxval cannot be called."""
    try:
        libc = ctypes.CDLL(None)
        libc.getauxval.restype = ctypes.c_ulong
        libc.getauxval.argtypes = [ctypes.c_ulong]
        return bool(libc.getauxval(AT_HWCAP) & HWCAP_ATOMICS)
    except (OSError, AttributeError, ValueError):
        return None


def cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    flags: set[str] = set()
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("flags", "features"):
            flags.update(value.split())
    return flags


def describe_cpu() -> tuple[str, list[str]]:
    machine = platform.machine()
    flags = cpu_flags()
    notes = []

    emulated = os.environ.get("QEMU_CPU")
    if emulated:
        notes.append(f"under qemu-user modelling {emulated}, not on real silicon.")

    if machine in ("aarch64", "arm64"):
        lse = has_lse()
        if lse is None:
            notes.append("could not read AT_HWCAP, so LSE here is unknown.")
        elif lse:
            notes.append(
                "this core has ARMv8.1 LSE, so a pass says NOTHING about a Pi 4 "
                "(Cortex-A72 is ARMv8.0 and traps on them). Use QEMU_CPU="
                "cortex-a72, or the Pi itself."
            )
        else:
            notes.append("no LSE here, so this is the ARMv8.0 baseline a Pi 4 has.")
    elif machine in ("x86_64", "AMD64"):
        simd = sorted(f for f in flags
                      if f.startswith("avx512") or f in ("avx2", "avx", "sse4_2"))
        notes.append("SIMD here: " + (", ".join(simd) or "x86-64 baseline only")
                     + ". These wheels dispatch on it at runtime rather than "
                     "requiring it.")
    return f"{machine} ({platform.processor() or 'unknown model'})", notes


def main() -> int:
    description, notes = describe_cpu()
    print(f"Executing the pinned numerical stack on {description}.")
    print(f"Python {sys.version.split()[0]}")
    for note in notes:
        print(f"  note: {note}")
    print()

    if "--require-no-lse" in sys.argv:
        # AT_HWCAP bit 8 means something else on x86.
        if platform.machine() not in ("aarch64", "arm64"):
            print(f"FAILED: --require-no-lse is an aarch64 check, but this is "
                  f"{platform.machine()}.", file=sys.stderr)
            return 2
        lse = has_lse()
        if lse is not False:
            print("FAILED: --require-no-lse, but AT_HWCAP reports LSE "
                  f"{'present' if lse else 'unreadable'}. This run would prove "
                  "nothing about a Cortex-A72.", file=sys.stderr)
            return 2
        print("  ARMv8.0 confirmed: AT_HWCAP carries no atomics.\n")

    # Imported one at a time so the step line above names whichever import dies.
    step("import numpy")
    import numpy as np

    step("import pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    step("import pandas (which imports pyarrow at import time)")
    import pandas as pd

    step("import scikit-learn's HistGradientBoostingRegressor")
    from sklearn.ensemble import HistGradientBoostingRegressor

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.parquet"
        frame = pd.DataFrame({"occupancy": [0.0, 1.0, 0.5]})
        step("write Parquet")
        frame.to_parquet(path)
        step("read Parquet back")
        assert pd.read_parquet(path).equals(frame), "Parquet round-trip differs"
        step("read Parquet metadata")
        assert pq.read_metadata(path).num_rows == 3, "wrong row count"

    step("fit HistGradientBoostingRegressor (needs libgomp)")
    x = np.arange(160, dtype=float).reshape(80, 2)
    y = (x[:, 0] % 8) / 8
    model = HistGradientBoostingRegressor(max_iter=2, max_leaf_nodes=3).fit(x, y)
    step("predict")
    assert np.isfinite(model.predict(x)).all(), "model predicted non-finite values"

    print(f"\nPASSED on {description}: numpy {np.__version__}, "
          f"pandas {pd.__version__}, pyarrow {pa.__version__}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
