# PyInstaller build: two executables sharing one folder (dist/fsr4).
#   "FSR4 Upscaler.exe"  the window, no console (double-click it)
#   fsr4.exe             the command line (fsr4 upscale clip.mp4 ...); with no arguments it opens the window too
# Build with build_exe.bat (it also copies the FSR DLLs next to the executables).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH)
sys.path.insert(0, str(ROOT))
from fsr4.ui_art import ico_bytes  # noqa: E402

ICON = ROOT / "build" / "fsr4.ico"
ICON.parent.mkdir(exist_ok=True)
ICON.write_bytes(ico_bytes())

analysis = Analysis(
    [str(ROOT / "run_fsr4.py")],
    pathex=[str(ROOT)],
    binaries=collect_dynamic_libs("cv2"),
    excludes=["torch", "torchvision", "matplotlib", "PIL", "pytest"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
command_line = EXE(archive, analysis.scripts, [], exclude_binaries=True, name="fsr4", console=True,
                   icon=str(ICON))
window = EXE(archive, analysis.scripts, [], exclude_binaries=True, name="FSR4 Upscaler", console=False,
             icon=str(ICON))
COLLECT(command_line, window, analysis.binaries, analysis.datas, name="fsr4")
