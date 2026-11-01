"""Write the third-party notices of the packaged app into `<folder>/licenses/`, one file per component.

build_exe.bat runs it on the folder PyInstaller made, with the same Python (so the notices are those of the
packages it bundled):

    python tools/collect_licenses.py dist/fsr4

The sources are the Python install, the installed packages' dist-info folders and the repository's
`third_party/`; nothing is copied into the repository. A missing source fails the build. AMD's license for the
FSR DLLs is not written here: build_exe.bat copies it next to them (`bin/license.md`).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
import tkinter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_LICENSE = Path(sys.base_prefix) / "LICENSE.txt"
TK_LICENSE = Path(sys.base_prefix) / "tcl" / f"tk{tkinter.TkVersion}" / "license.terms"
KHRONOS_LICENSE = ROOT / "third_party" / "LICENSE-Khronos.txt"
VULKAN_HEADER = ROOT / "third_party" / "vulkan_core.h"
FIDELITYFX_HEADERS = ROOT / "third_party" / "ffx"
# Python's Windows build ships Tcl without its license.terms; its LICENSE.txt carries that text verbatim
# (followed by Tk's, which names Apple Inc. as well).
TCL_TERMS_START = ("This software is copyrighted by the Regents of the University of\n"
                   "California, Sun Microsystems, Inc., Scriptics Corporation, ActiveState\n"
                   "Corporation and other parties.")
TCL_TERMS_END = "terms specified in this license.\n"


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    return path.read_text(encoding="utf-8")


def existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing")
    return path


def metadata_file(package: str, file_name: str) -> Path:
    """`file_name` in the installed `package`'s dist-info folder, where its wheel puts its licence files."""
    distribution = importlib.metadata.distribution(package)
    for entry in distribution.files or ():
        if entry.name == file_name and entry.parts[0].endswith(".dist-info"):
            return existing(Path(distribution.locate_file(entry)))
    raise FileNotFoundError(f"{package} {distribution.version} has no {file_name} in its dist-info folder")


def tcl_terms() -> str:
    """Tcl's license.terms, cut out of Python's LICENSE.txt."""
    text = read(PYTHON_LICENSE)
    start = text.find(TCL_TERMS_START)
    end = text.find(TCL_TERMS_END, start)
    if start < 0 or end < 0:
        raise ValueError(f"Tcl's license terms are not in {PYTHON_LICENSE}")
    return text[start:end + len(TCL_TERMS_END)]


def leading_comment(path: Path) -> str:
    """The `//` comment block a C header starts with, without the slashes."""
    lines = []
    for line in read(path).splitlines():
        if not line.startswith("//"):
            break
        lines.append(line[2:].strip())
    return "\n".join(lines).strip() + "\n"


def fidelityfx_notice() -> str:
    """The MIT notice of AMD's FidelityFX API headers, which fsr4/ffx.py mirrors; every header must carry it."""
    headers = sorted(FIDELITYFX_HEADERS.glob("*.h"))
    notices = {leading_comment(header) for header in headers}
    if not headers or len(notices) != 1:
        raise ValueError(f"expected one MIT notice shared by the headers in {FIDELITYFX_HEADERS}, "
                         f"found {len(notices)} in {len(headers)} headers")
    names = ", ".join(header.name for header in headers)
    return f"AMD FidelityFX SDK API headers ({names}), mirrored by fsr4/ffx.py:\n\n{notices.pop()}"


def khronos_license() -> Path:
    """third_party/LICENSE-Khronos.txt, once it names the same copyright as the vendored header."""
    copyright_line = next((line[3:].strip() for line in read(VULKAN_HEADER).splitlines()
                           if line.startswith("** Copyright ")), None)
    if copyright_line is None or copyright_line not in read(KHRONOS_LICENSE):
        raise ValueError(f"{KHRONOS_LICENSE} must carry the header's line {copyright_line!r}")
    return KHRONOS_LICENSE


def python_incorporated_software() -> str:
    """Where Python publishes the notices of the code built into its DLL and modules, which its Windows
    LICENSE.txt leaves out: a pointer, as in Python's own Windows install."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (f"python{sys.version_info.major}{sys.version_info.minor}.dll and the Python modules bundled here "
            "incorporate third-party software (among them Expat in pyexpat.pyd and libmpdec in _decimal.pyd). "
            "Their copyright and license notices are in Python's documentation, \"Licenses and Acknowledgements "
            f"for Incorporated Software\":\n\n    https://docs.python.org/{version}/license.html\n")


def notices() -> dict[str, Path | str]:
    """File name in licenses/ -> the file to copy or the text to write."""
    return {
        "Python-LICENSE.txt": existing(PYTHON_LICENSE),
        "Python-incorporated-software.txt": python_incorporated_software(),
        "Tcl-LICENSE.txt": tcl_terms(),
        "Tk-LICENSE.txt": existing(TK_LICENSE),
        "numpy-LICENSE.txt": metadata_file("numpy", "LICENSE.txt"),
        "opencv-python-LICENSE.txt": metadata_file("opencv-python", "LICENSE.txt"),
        "opencv-python-LICENSE-3RD-PARTY.txt": metadata_file("opencv-python", "LICENSE-3RD-PARTY.txt"),
        "FidelityFX-API-headers-LICENSE.txt": fidelityfx_notice(),
        "Vulkan-headers-LICENSE.txt": khronos_license(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the third-party notices into <folder>/licenses/.")
    parser.add_argument("folder", type=Path, help="the packaged app's folder (dist/fsr4)")
    folder = parser.parse_args().folder
    try:
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} is not a folder")
        sources = notices()
        target = folder / "licenses"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        for name, source in sources.items():
            if isinstance(source, Path):
                shutil.copyfile(source, target / name)
            else:
                (target / name).write_text(source, encoding="utf-8")
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"collect_licenses: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(sources)} notices into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
