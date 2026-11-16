from __future__ import annotations

import hashlib
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

FREEZE_REL_PATH = Path("outputs") / "env" / "eval_stack_freeze.txt"
FREEZE_SHA256 = "838ebea44991b7a1cd385d976ebe545ba6a253341421dfd73c3e4a3b8ad320d8"
TRAIN_FREEZE_REL_PATH = Path("outputs") / "env" / "train_stack_freeze.txt"
TRAIN_FREEZE_SHA256 = "eebf2290140b148aa738f19065d21c63470f3aea4fabed12b30959432daecae2"
ANCHOR_FREEZE_REL_PATH = TRAIN_FREEZE_REL_PATH
ANCHOR_FREEZE_SHA256 = TRAIN_FREEZE_SHA256


def load_verified_freeze(
    repo_root: Path,
    *,
    rel_path: Path = FREEZE_REL_PATH,
    expected_sha256: str = FREEZE_SHA256,
) -> dict[str, str]:
    path = repo_root / rel_path
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha256:
        raise SystemExit(
            f"[envcheck] {path} sha256 {got[:16]}… != committed pin "
            f"{expected_sha256[:16]}… — the recorded freeze was modified"
        )
    pins: dict[str, str] = {}
    for line in data.decode().splitlines():
        line = line.strip()
        if "==" in line and not line.startswith(("#", "-")):
            name, ver = line.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins


def check_freeze_complete(
    repo_root: Path,
    *,
    rel_path: Path = FREEZE_REL_PATH,
    expected_sha256: str = FREEZE_SHA256,
) -> dict[str, str]:
    pins = load_verified_freeze(repo_root, rel_path=rel_path, expected_sha256=expected_sha256)
    drift: dict[str, dict[str, str | None]] = {}
    checked: dict[str, str] = {}
    for name, want in pins.items():
        try:
            got = version(name)
        except PackageNotFoundError:
            drift[name] = {"recorded": want, "installed": None}
            continue
        if got != want:
            drift[name] = {"recorded": want, "installed": got}
        checked[name] = got
    if drift:
        raise SystemExit(
            f"[envcheck] env drift vs recorded freeze ({len(drift)} packages): {drift}"
        )
    return checked


def run_nvidia_smi() -> str:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"[envcheck] nvidia-smi did not run: {e}") from None
    if smi.returncode != 0:
        raise SystemExit(f"[envcheck] nvidia-smi failed: {smi.stderr.strip()}")
    return smi.stdout.strip()
