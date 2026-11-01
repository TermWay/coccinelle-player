"""Minimal ctypes binding to vulkan-1.dll: only the entry points this tool needs.

Structures and enum constants come from the generated `vk_types` module. Every call that returns a
`VkResult` goes through `check`, which raises `VkError` on failure: nothing is ignored.
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, byref, c_char_p, c_float, c_int32, c_uint32, c_uint64, c_void_p

from . import vk_types as T

VkResult = c_int32
VkBool32 = c_uint32
VkDeviceSize = c_uint64

_library = None
_load_lock = threading.Lock()
# name -> (restype, argtypes) of every entry point, bound as module attributes by `load`.
_SIGNATURES: dict[str, tuple] = {}

_RESULT_NAMES = {
    value: name for name, value in vars(T).items()
    if name.startswith("VK_ERROR_") or name in ("VK_SUCCESS", "VK_NOT_READY", "VK_TIMEOUT", "VK_INCOMPLETE")
}


class VkError(RuntimeError):
    """A Vulkan call returned an error code."""

    def __init__(self, result: int, what: str):
        super().__init__(f"{what} failed: {_RESULT_NAMES.get(result, result)} ({result})")
        self.result = result


def check(result: int, what: str) -> None:
    """Raise `VkError` unless `result` is VK_SUCCESS."""
    if result != T.VK_SUCCESS:
        raise VkError(result, what)


def _function(name: str, restype, *argtypes) -> None:
    _SIGNATURES[name] = (restype, argtypes)


def load() -> None:
    """Load vulkan-1.dll and bind the entry points. It happens on first use, so importing this module (the
    window, `--version`) works on a PC without a Vulkan driver, which then gets this error where the GPU is
    needed."""
    global _library
    with _load_lock:
        if _library is not None:
            return
        try:
            library = ctypes.WinDLL("vulkan-1")
        except OSError as error:
            raise OSError(f"cannot load vulkan-1.dll, the Vulkan runtime ({error}); install or update the "
                          f"graphics driver") from error
        for name, (restype, argtypes) in _SIGNATURES.items():
            function = getattr(library, name)
            function.restype = restype
            function.argtypes = argtypes
            globals()[name] = function
        _library = library


def __getattr__(name: str):
    """The entry points, bound on first use (see `load`)."""
    if name in _SIGNATURES:
        load()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


Pointer = c_void_p

_function("vkCreateInstance", VkResult, Pointer, Pointer, POINTER(c_void_p))
_function("vkDestroyInstance", None, c_void_p, Pointer)
_function("vkEnumerateInstanceLayerProperties", VkResult, POINTER(c_uint32), Pointer)
_function("vkEnumeratePhysicalDevices", VkResult, c_void_p, POINTER(c_uint32), Pointer)
_function("vkGetPhysicalDeviceProperties", None, c_void_p, Pointer)
_function("vkGetPhysicalDeviceFeatures2", None, c_void_p, Pointer)
_function(
    "vkEnumerateDeviceExtensionProperties", VkResult, c_void_p, c_char_p, POINTER(c_uint32), Pointer)
_function(
    "vkGetPhysicalDeviceQueueFamilyProperties", None, c_void_p, POINTER(c_uint32), Pointer)
_function("vkGetPhysicalDeviceMemoryProperties", None, c_void_p, Pointer)
_function("vkGetPhysicalDeviceFormatProperties", None, c_void_p, c_int32, Pointer)
_function("vkCreateDevice", VkResult, c_void_p, Pointer, Pointer, POINTER(c_void_p))
_function("vkDestroyDevice", None, c_void_p, Pointer)
_function("vkDeviceWaitIdle", VkResult, c_void_p)
_function("vkGetDeviceQueue", None, c_void_p, c_uint32, c_uint32, POINTER(c_void_p))
_function("vkGetDeviceProcAddr", c_void_p, c_void_p, c_char_p)

_function("vkCreateImage", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
_function("vkDestroyImage", None, c_void_p, c_uint64, Pointer)
_function("vkGetImageMemoryRequirements", None, c_void_p, c_uint64, Pointer)
_function("vkBindImageMemory", VkResult, c_void_p, c_uint64, c_uint64, VkDeviceSize)
_function("vkCreateBuffer", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
_function("vkDestroyBuffer", None, c_void_p, c_uint64, Pointer)
_function("vkGetBufferMemoryRequirements", None, c_void_p, c_uint64, Pointer)
_function("vkBindBufferMemory", VkResult, c_void_p, c_uint64, c_uint64, VkDeviceSize)
_function("vkAllocateMemory", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
_function("vkFreeMemory", None, c_void_p, c_uint64, Pointer)
_function("vkMapMemory", VkResult, c_void_p, c_uint64, VkDeviceSize, VkDeviceSize, c_uint32, POINTER(c_void_p))
_function("vkUnmapMemory", None, c_void_p, c_uint64)

_function("vkCreateCommandPool", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
_function("vkDestroyCommandPool", None, c_void_p, c_uint64, Pointer)
_function("vkAllocateCommandBuffers", VkResult, c_void_p, Pointer, POINTER(c_void_p))
_function("vkBeginCommandBuffer", VkResult, c_void_p, Pointer)
_function("vkEndCommandBuffer", VkResult, c_void_p)
_function("vkResetCommandBuffer", VkResult, c_void_p, c_uint32)
_function(
    "vkCmdPipelineBarrier", None, c_void_p, c_uint32, c_uint32, c_uint32,
    c_uint32, Pointer, c_uint32, Pointer, c_uint32, Pointer)
_function("vkCmdCopyBufferToImage", None, c_void_p, c_uint64, c_uint64, c_int32, c_uint32, Pointer)
_function("vkCmdCopyImageToBuffer", None, c_void_p, c_uint64, c_int32, c_uint64, c_uint32, Pointer)
_function("vkCmdCopyImage", None, c_void_p, c_uint64, c_int32, c_uint64, c_int32, c_uint32, Pointer)
_function("vkCmdClearColorImage", None, c_void_p, c_uint64, c_int32, Pointer, c_uint32, Pointer)
_function("vkQueueSubmit", VkResult, c_void_p, c_uint32, Pointer, c_uint64)
_function("vkQueueWaitIdle", VkResult, c_void_p)
_function("vkCreateFence", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
_function("vkDestroyFence", None, c_void_p, c_uint64, Pointer)
_function("vkWaitForFences", VkResult, c_void_p, c_uint32, POINTER(c_uint64), VkBool32, c_uint64)
_function("vkResetFences", VkResult, c_void_p, c_uint32, POINTER(c_uint64))


def address_of_vkGetDeviceProcAddr() -> int:
    """The raw address of the loader's vkGetDeviceProcAddr, handed to the FSR backend."""
    load()
    return ctypes.cast(globals()["vkGetDeviceProcAddr"], c_void_p).value


def pointer(structure) -> int:
    """Address of a ctypes structure, for `pNext` chains and `Pointer` arguments."""
    return ctypes.addressof(structure)


def c_string_array(strings: list[str]):
    """A `const char* const*` array for extension and layer name lists."""
    array = (c_char_p * max(len(strings), 1))()
    for index, text in enumerate(strings):
        array[index] = text.encode("ascii")
    return array


__all__ = list(_SIGNATURES) + [name for name in dir() if name.startswith("Vk") or name in (
    "check", "load", "VkError", "pointer", "c_string_array", "address_of_vkGetDeviceProcAddr", "T", "c_float", "byref")]
