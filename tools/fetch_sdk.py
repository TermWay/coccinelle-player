"""Download the two signed FSR SDK Vulkan DLLs (and their license) into `bin/`.

Source: the AMD FidelityFX SDK repository on GitHub, tag v2.4.0 (FSR Upscaling 4.1.2). The files are
redistributable binaries under the SDK license in `bin/license.md`. SHA-256 hashes are pinned after the
first verified download; a mismatch aborts.

    python tools/fetch_sdk.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

TAG = "v2.4.0"
BASE = f"https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/{TAG}"
FILES = {
    "amd_fidelityfx_loader_vk.dll": "Kits/FidelityFX/signedbin/amd_fidelityfx_loader_vk.dll",
    "amd_fidelityfx_upscaler_vk.dll": "Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll",
    "license.md": "Kits/FidelityFX/docs/license.md",
}
SHA256 = {
    # filled in after the first download; see README
}
DESTINATION = Path(__file__).resolve().parent.parent / "bin"


def download(url: str, destination: Path) -> int:
    with urllib.request.urlopen(url) as response, open(destination, "wb") as output:
        total = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            output.write(chunk)
            total += len(chunk)
    return total


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        target = DESTINATION / name
        url = f"{BASE}/{remote}"
        print(f"downloading {url}")
        size = download(url, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"  {name}: {size / 1048576:.2f} MB, sha256 {digest}")
        if name in SHA256 and SHA256[name] != digest:
            target.unlink()
            print(f"  hash mismatch for {name}: expected {SHA256[name]}", file=sys.stderr)
            return 1
    print(f"done: {DESTINATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
