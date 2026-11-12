"""What the window shows about the machine (Windows): the processor's name, how much of the main memory and of
the graphics card's memory this program uses, and the card's driver (for bug reports).

The processor's name comes from the registry; memory in use from the process counters (working set) and the
physical total; the graphics card's from DXGI, which counts every allocation of this process on that card,
Vulkan included, against its dedicated memory.
"""
from __future__ import annotations

import ctypes
import re
import winreg
from ctypes import wintypes

GIGABYTE = 1024 ** 3


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "GUID":
        parts = text.split("-")
        tail = bytes.fromhex(parts[3] + parts[4])
        return cls(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), (ctypes.c_ubyte * 8)(*tail))


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_int64), ("Flags", wintypes.UINT)]


class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]


IID_FACTORY1 = GUID.parse("770aae78-f26f-4dba-a829-253c83d1b387")
IID_ADAPTER3 = GUID.parse("645967a4-1392-4310-a798-8053ce3e93fd")
DXGI_ERROR_NOT_FOUND = 0x887A0002
SEGMENT_LOCAL = 0
DISPLAY_ADAPTERS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
# COM method slots (vtable indices) of the DXGI interfaces used.
RELEASE, QUERY_INTERFACE = 2, 0
ENUM_ADAPTERS1, GET_DESC1, QUERY_VIDEO_MEMORY_INFO = 12, 10, 14


def processor_name() -> str:
    """The processor's name without its maker's trademarks and core count: "Ryzen 9 9950X"."""
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    name = re.sub(r"\((R|TM)\)|\bCPU\b|\b\d+-Core\b|\bProcessor\b|@.*$|^(AMD|Intel)\b", " ", name)
    return " ".join(name.split())


def display_driver(card: str) -> str | None:
    """The driver of the graphics card named `card` as people know it: "Adrenalin 26.10.44, Windows driver
    32.0.31044.16" (AMD writes its version beside Windows' entry), None when no display adapter has that name."""
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DISPLAY_ADAPTERS_KEY) as adapters:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(adapters, index)
            except OSError:
                return None  # no more adapters
            index += 1
            try:
                with winreg.OpenKey(adapters, name) as adapter:
                    if winreg.QueryValueEx(adapter, "DriverDesc")[0] != card:
                        continue
                    version = winreg.QueryValueEx(adapter, "DriverVersion")[0]
                    try:
                        radeon = winreg.QueryValueEx(adapter, "RadeonSoftwareVersion")[0]
                    except FileNotFoundError:
                        return f"Windows driver {version}"
                    return f"Adrenalin {radeon}, Windows driver {version}"
            except OSError:
                continue  # "Properties" and other keys that are not adapters, or not readable


def main_memory() -> tuple[int, int]:
    """(bytes this process uses: its working set, bytes of memory installed)."""
    counters = PROCESS_MEMORY_COUNTERS(cb=ctypes.sizeof(PROCESS_MEMORY_COUNTERS))
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(process), ctypes.byref(counters),
                                                     counters.cb):
        raise ctypes.WinError()
    installed = ctypes.c_ulonglong()
    if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(installed)):
        return counters.WorkingSetSize, installed.value * 1024
    status = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))  # a virtual machine: what Windows can use
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return counters.WorkingSetSize, status.ullTotalPhys


def _method(interface: ctypes.c_void_p, slot: int, *argtypes):
    """The COM method in `slot` of `interface`'s vtable, callable with its arguments after `this`."""
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    function = prototype(vtable[slot])
    return lambda *arguments: function(interface, *arguments)


def _check(result: int, what: str) -> None:
    if result < 0:
        raise OSError(f"cannot {what} (DXGI error {result & 0xFFFFFFFF:#010x})")


class VideoMemory:
    """The dedicated memory of the graphics card whose name contains `name` (the largest one when none does):
    `read()` gives (bytes this process uses on it, its size)."""

    def __init__(self, name: str):
        factory = ctypes.c_void_p()
        _check(ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(IID_FACTORY1), ctypes.byref(factory)),
               "open DXGI")
        best = None
        try:
            index = 0
            while True:
                adapter = ctypes.c_void_p()
                result = _method(factory, ENUM_ADAPTERS1, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(
                    index, ctypes.byref(adapter))
                if result & 0xFFFFFFFF == DXGI_ERROR_NOT_FOUND:
                    break
                _check(result, "list the graphics cards")
                description = DXGI_ADAPTER_DESC1()
                _check(_method(adapter, GET_DESC1, ctypes.POINTER(DXGI_ADAPTER_DESC1))(ctypes.byref(description)),
                       "read a graphics card's description")
                score = (name.lower() in description.Description.lower(), description.DedicatedVideoMemory)
                if best is None or score > best[0]:
                    if best is not None:
                        _method(best[1], RELEASE)()
                    best = (score, adapter, description.DedicatedVideoMemory, description.Description)
                else:
                    _method(adapter, RELEASE)()
                index += 1
        finally:
            _method(factory, RELEASE)()
        if best is None:
            raise OSError("DXGI lists no graphics card")
        _, adapter, self.total, self.description = best
        self._adapter3 = ctypes.c_void_p()
        try:
            _check(_method(adapter, QUERY_INTERFACE, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))(
                ctypes.byref(IID_ADAPTER3), ctypes.byref(self._adapter3)), "query the graphics card's memory")
        finally:
            _method(adapter, RELEASE)()

    def read(self) -> tuple[int, int]:
        info = DXGI_QUERY_VIDEO_MEMORY_INFO()
        _check(_method(self._adapter3, QUERY_VIDEO_MEMORY_INFO, wintypes.UINT, ctypes.c_int,
                       ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO))(0, SEGMENT_LOCAL, ctypes.byref(info)),
               "read the graphics card's memory use")
        return info.CurrentUsage, self.total

    def close(self) -> None:
        if self._adapter3:
            _method(self._adapter3, RELEASE)()
            self._adapter3 = ctypes.c_void_p()
