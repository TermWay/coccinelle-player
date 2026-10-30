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
    "amd_fidelityfx_loader_vk.dll": "38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9",
    "amd_fidelityfx_upscaler_vk.dll": "4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7",
    "license.md": "32357259b965d6a8001e9e508ba60eda6034027c5923a7d226ffbaff813c2bfb",
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
