"""Download the two signed FSR SDK Vulkan DLLs (and their license) into `bin/`, or check the ones there.

Source: the AMD FidelityFX SDK repository on GitHub, tag v2.4.0 (FSR Upscaling 4.1.2). The files are
redistributable binaries under the SDK license in `bin/license.md`. Every file has a pinned SHA-256: a download
goes to `bin/<name>.part` and replaces `bin/<name>` only once its hash matches, so an interrupted or altered
download never takes the place of a good file. Files that already match are not downloaded again.

    python tools/fetch_sdk.py           download what is missing or does not match
    python tools/fetch_sdk.py --check   re-hash bin/; exit status 1 when a file is missing or does not match
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

TAG = "v2.4.0"
REPOSITORY = "GPUOpen-LibrariesAndSDKs/FidelityFX-SDK"
BASE = f"https://raw.githubusercontent.com/{REPOSITORY}/{TAG}"
FILES = {
    "amd_fidelityfx_loader_vk.dll": "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll",
    "amd_fidelityfx_upscaler_vk.dll": "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
    "license.md": "Kits/FidelityFX/docs/license.md",
}
SHA256 = {
    "amd_fidelityfx_loader_vk.dll": "38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9",
    "amd_fidelityfx_upscaler_vk.dll": "4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7",
    "license.md": "32357259b965d6a8001e9e508ba60eda6034027c5923a7d226ffbaff813c2bfb",
}
DESTINATION = Path(__file__).resolve().parent.parent / "bin"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 20


def pinned_sha256(name: str) -> str:
    """The SHA-256 `name` must have: every file this script fetches needs one."""
    if name not in SHA256:
        raise KeyError(f"{name} has no pinned SHA-256 in tools/fetch_sdk.py")
    return SHA256[name]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def problem(name: str) -> str | None:
    """Why `bin/<name>` cannot be used (missing, or another hash), or None when it matches its pin."""
    path = DESTINATION / name
    if not path.is_file():
        return f"{path} is missing"
    digest = file_sha256(path)
    if digest != pinned_sha256(name):
        return f"{path} has SHA-256 {digest}, pinned {pinned_sha256(name)}"
    return None


def download(url: str, destination: Path) -> int:
    """Write the file at `url` to `destination`; returns its size in bytes."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response, open(destination, "wb") as output:
        total = 0
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
            total += len(chunk)
    return total


def fetch(name: str, remote: str) -> None:
    """Download one file to `bin/<name>.part`, verify it, then move it into place."""
    url = f"{BASE}/{remote}"
    target = DESTINATION / name
    partial = DESTINATION / f"{name}.part"
    print(f"downloading {url}")
    try:
        size = download(url, partial)
        digest = file_sha256(partial)
        if digest != pinned_sha256(name):
            raise ValueError(f"{name}: the download has SHA-256 {digest}, pinned {pinned_sha256(name)}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"  {name}: {size / 1048576:.2f} MB, SHA-256 verified")


def check() -> int:
    """Print every file that is missing or does not match its pin; 1 when there is one."""
    problems = [text for text in map(problem, FILES) if text is not None]
    for text in problems:
        print(text, file=sys.stderr)
    if problems:
        return 1
    print(f"{DESTINATION}: {len(FILES)} files match their pinned SHA-256")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AMD's FSR DLLs into bin/, or check the ones there.")
    parser.add_argument("--check", action="store_true",
                        help="only re-hash bin/: exit status 1 when a file is missing or does not match")
    arguments = parser.parse_args()
    for name in FILES:
        pinned_sha256(name)
    if arguments.check:
        return check()
    print(f"source: AMD FidelityFX SDK on GitHub ({REPOSITORY}), tag {TAG}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    try:
        for name, remote in FILES.items():
            if problem(name) is None:
                print(f"  {name}: already there, SHA-256 verified")
                continue
            fetch(name, remote)
    except (OSError, ValueError) as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1
    print(f"done: {DESTINATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
