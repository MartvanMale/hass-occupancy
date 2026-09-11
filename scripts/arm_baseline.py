#!/usr/bin/env python3
"""Find UNGUARDED ARMv8.1 LSE atomics in the aarch64 wheels, for check-pins.sh.

Not a grep: libgcc's outline atomics put dispatch-guarded LSE in every wheel, and
only the unguarded form matters -- the pyarrow 21.0.0 bug, inlined in bundled
mimalloc. The gate is a baseline diff, so a mis-read is a reviewed bump.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# The wheel tag to resolve against. aarch64 because that is the Pi; cp313
# because build.yaml pins the Debian trixie base, whose python3 is 3.13.
PLATFORM = "manylinux_2_28_aarch64"
PYTHON_VERSION = "313"
ABI = "cp313"

OBJDUMP = "aarch64-linux-gnu-objdump"

# ARMv8.1-A Large System Extensions, undefined on a Cortex-A72 (ARMv8.0-A). The
# suffixes are the acquire/release/byte/halfword variants; `casp` is the pair form.
LSE = re.compile(
    r"^(?:cas[abhlp]*|swp[abhl]*"
    r"|ld(?:add|clr|eor|set|smax|smin|umax|umin)[abhl]*"
    r"|st(?:add|clr|eor|set|smax|smin|umax|umin)[bhl]*)$"
)

# libgcc's outline-atomics helpers, when they still have their symbol.
GUARD_SYMBOL = re.compile(
    r"^__aarch64_(?:cas|swp|ld(?:add|clr|eor|set|smax|smin|umax|umin))"
)

# The ldxr/stxr fallback every dispatch-guarded helper carries: just after an LSE
# instruction, it identifies the helper when the symbol name is stripped.
EXCLUSIVE = re.compile(r"^(?:ld[ax]+r[bh]?|st[lx]+r[bh]?)$")

# How far to look either side of a hit for the dispatch shape. The helpers are
# a dozen instructions long, so this is generous rather than tuned.
LOOK_BACK = 6
LOOK_AHEAD = 16


# ---------------------------------------------------------------------------
# Resolving what would actually be installed
# ---------------------------------------------------------------------------

def resolve(requirements: Path, workdir: Path) -> list[dict]:
    """Resolve the full aarch64 closure, not just the pinned lines: scikit-learn
    drags in scipy, unpinned and bundling its own OpenBLAS. `--dry-run --report`
    yields the wheel URLs without installing.
    """
    report = workdir / "report.json"
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--dry-run", "--ignore-installed", "--quiet",
            "--report", str(report),
            "--only-binary=:all:",
            "--platform", PLATFORM,
            "--python-version", PYTHON_VERSION,
            "--abi", ABI,
            "--implementation", "cp",
            "--target", str(workdir / "target"),
            "-r", str(requirements),
        ],
        check=True,
    )
    with report.open() as handle:
        return json.load(handle)["install"]


def fetch(entries: list[dict], dest: Path) -> dict[str, str]:
    """Download and unpack each wheel into its own directory, so an object can
    be attributed to the distribution it came from. Returns package -> version.
    """
    versions: dict[str, str] = {}
    for entry in entries:
        name = entry["metadata"]["name"]
        version = entry["metadata"]["version"]
        url = entry["download_info"]["url"]
        if not url.endswith(".whl"):
            continue
        versions[name] = version
        target = dest / name
        target.mkdir(parents=True, exist_ok=True)
        wheel = dest / url.rsplit("/", 1)[-1]
        with urllib.request.urlopen(url) as response, wheel.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        wheel.unlink()
    return versions


# ---------------------------------------------------------------------------
# Naming objects so they compare across a version bump
# ---------------------------------------------------------------------------

_SONAME = re.compile(r"\.so(?:\.\d+)+$")
_ABI_TAG = re.compile(r"\.cpython-\d+[a-z]*-[a-z0-9_]+-linux-gnu\.so$")
_AUDITWHEEL_HASH = re.compile(r"-[0-9a-f]{8,}(?=\.so)")


def normalise(relative: Path) -> str:
    """A stable name for a shared object across versions of its wheel: the soname,
    auditwheel's `-<hash>` and the ABI tag all change on a bump that changed no code.
    """
    name = relative.name
    name = _SONAME.sub(".so", name)
    name = _ABI_TAG.sub(".so", name)
    name = _AUDITWHEEL_HASH.sub("", name)
    return str(relative.parent / name)


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

class Hit(collections.namedtuple("Hit", "symbol mnemonic")):
    __slots__ = ()


def scan(path: Path) -> tuple[int, list[Hit]]:
    """Disassemble one object. Returns (guarded count, unguarded hits). Streamed;
    a hit settles once enough following instructions have gone by, since whether
    it is guarded needs a window in both directions.
    """
    process = subprocess.Popen(
        [OBJDUMP, "-d", "--no-show-raw-insn", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="replace",
    )
    assert process.stdout is not None

    symbol = "?"
    recent: collections.deque[str] = collections.deque(maxlen=LOOK_BACK)
    pending: list[dict] = []
    guarded = 0
    unguarded: list[Hit] = []

    def settle(hit: dict) -> None:
        nonlocal guarded
        if hit["guarded"]:
            guarded += 1
        else:
            unguarded.append(Hit(hit["symbol"], hit["mnemonic"]))

    for line in process.stdout:
        # `0000000000182540 <PyInit_algos@@Base>:` -- a symbol boundary.
        if line[:1].isdigit() and line.rstrip().endswith(":") and "<" in line:
            symbol = line.split("<", 1)[1].rsplit(">", 1)[0]
            continue
        if "\t" not in line:
            continue
        text = line.split("\t", 1)[1].strip()
        if not text:
            continue
        mnemonic = text.split(None, 1)[0]

        for hit in pending:
            hit["after"] += 1
            if EXCLUSIVE.match(mnemonic):
                # The ARMv8.0 fallback right after the LSE path: this is a
                # dispatch-guarded helper whose symbol was stripped.
                hit["guarded"] = True
        while pending and pending[0]["after"] >= LOOK_AHEAD:
            settle(pending.pop(0))

        if LSE.match(mnemonic):
            if GUARD_SYMBOL.match(symbol):
                guarded += 1
            else:
                # The dispatch preamble: load the LSE availability byte and branch
                # on it. Both halves must be in the window -- a lone cbz is no guard.
                window = list(recent)
                preamble = any(m.startswith("ldrb") for m in window) and any(
                    m.startswith(("cbz", "cbnz", "tbz", "tbnz")) for m in window
                )
                pending.append(
                    {"symbol": symbol, "mnemonic": mnemonic,
                     "guarded": preamble, "after": 0}
                )
        recent.append(mnemonic)

    for hit in pending:
        settle(hit)
    process.wait()
    return guarded, unguarded


def scan_tree(root: Path) -> dict[str, dict]:
    """Scan every shared object under `root`, keyed by normalised name. Deduped by
    content hash: pyarrow ships byte-identical copies of each bundled lib.
    """
    results: dict[str, dict] = {}
    seen: dict[str, str] = {}
    for package in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(package.rglob("*.so*")):
            if path.is_symlink() or not path.is_file():
                continue
            # Prefixed with the distribution unless the wheel already lays its
            # objects out under that name: `pyarrow/pyarrow/...` reads as a mistake.
            inside = normalise(path.relative_to(package))
            name = inside if inside.startswith(f"{package.name}/") \
                else f"{package.name}/{inside}"
            if name in results:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen[digest] = name
            guarded, unguarded = scan(path)
            if guarded or unguarded:
                results[name] = {"guarded": guarded, "unguarded": unguarded}
    return results


# ---------------------------------------------------------------------------
# Baseline and verdict
# ---------------------------------------------------------------------------

BASELINE_NOTE = (
    "Per-object count of ARMv8.1 LSE instructions NOT behind libgcc's runtime "
    "dispatch. Written by scripts/check-pins.sh --update-baseline. All should be 0."
)


def report(results: dict[str, dict], versions: dict[str, str]) -> None:
    total_guarded = sum(entry["guarded"] for entry in results.values())
    total_unguarded = sum(len(entry["unguarded"]) for entry in results.values())
    print(f"Resolved for {PLATFORM} / {ABI}:")
    for name in sorted(versions):
        print(f"  {name} {versions[name]}")
    print(
        f"\n{len(results)} objects with LSE instructions: "
        f"{total_guarded} dispatch-guarded, {total_unguarded} not."
    )
    for name in sorted(results):
        entry = results[name]
        if not entry["unguarded"]:
            continue
        symbols = collections.Counter(hit.symbol for hit in entry["unguarded"])
        listed = ", ".join(symbol for symbol, _ in symbols.most_common(6))
        if len(symbols) > 6:
            listed += ", ..."
        print(f"  {name}: {len(entry['unguarded'])} unguarded ({listed})")


def compare(results: dict[str, dict], baseline: dict) -> list[str]:
    """Fail only where an object gained unguarded LSE it did not have before --
    the pyarrow 21 -> 22 signature, and it needs no judgement."""
    expected = baseline.get("objects", {})
    problems = []
    for name in sorted(results):
        count = len(results[name]["unguarded"])
        before = expected.get(name, {}).get("unguarded", 0)
        if count > before:
            known = f"{before} in the baseline" if name in expected else "not in the baseline"
            problems.append(f"{name}: {count} unguarded LSE hits, {known}")
    return problems


UNCONFIRMED = "not yet confirmed on a Raspberry Pi 4 (Cortex-A72)"


def build_baseline(results: dict[str, dict], versions: dict[str, str],
                   previous: dict) -> dict:
    objects = {}
    for name in sorted(results):
        package = name.split("/", 1)[0]
        objects[name] = {
            "unguarded": len(results[name]["unguarded"]),
            "version": versions.get(package, "?"),
        }

    # A confirmation is about a set of versions, so any version that moved
    # invalidates it -- otherwise the file claims hardware evidence it lacks.
    confirmed = previous.get("confirmed_on_hardware", UNCONFIRMED)
    before = {name: entry.get("version")
              for name, entry in previous.get("objects", {}).items()}
    moved = sorted({
        name.split("/", 1)[0]
        for name, entry in objects.items()
        if name in before and before[name] != entry["version"]
    })
    if moved and confirmed != UNCONFIRMED:
        confirmed = (f"{UNCONFIRMED} -- the previous confirmation was voided "
                     f"when {', '.join(moved)} moved. It read: {confirmed}")

    return {
        "note": BASELINE_NOTE,
        "platform": PLATFORM,
        "abi": ABI,
        "confirmed_on_hardware": confirmed,
        "objects": objects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    if shutil.which(OBJDUMP) is None:
        print(f"error: {OBJDUMP} not found -- run this through "
              f"scripts/check-pins.sh", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        entries = resolve(args.requirements, workdir)
        wheels = workdir / "wheels"
        wheels.mkdir()
        versions = fetch(entries, wheels)
        results = scan_tree(wheels)

    report(results, versions)

    previous = {}
    if args.baseline.exists():
        with args.baseline.open() as handle:
            previous = json.load(handle)

    if args.update_baseline:
        with args.baseline.open("w") as handle:
            json.dump(build_baseline(results, versions, previous), handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.baseline}.")
        return 0

    if not previous:
        print(f"\nerror: {args.baseline} does not exist -- nothing to compare "
              f"against.\n       Run with --update-baseline once these wheels "
              f"are known good.", file=sys.stderr)
        return 2

    problems = compare(results, previous)
    if problems:
        sys.stdout.flush()   # else the verdict prints above its own report
        print("\nFAIL: unguarded ARMv8.1 LSE where the baseline had none.",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("\nThese instructions are undefined on a Cortex-A72, so this pin "
              "may abort with\n`Illegal instruction` on a Raspberry Pi 4. Read "
              "which object and which symbols:\nLSE inlined into a library's "
              "own code is the bug, and confirming it on the Pi\nis the next "
              "step. If it is genuinely fine, record it with "
              "--update-baseline.", file=sys.stderr)
        return 1

    print(f"\nOK: no object gained unguarded LSE against {args.baseline}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
