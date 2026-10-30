"""Minimal ctypes binding to vulkan-1.dll: only the entry points this tool needs.

Structures and enum constants come from the generated `vk_types` module. Every call that returns a
`VkResult` goes through `check`, which raises `VkError` on failure: nothing is ignored.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_float, c_int32, c_uint32, c_uint64, c_void_p

from . import vk_types as T

VkResult = c_int32
VkBool32 = c_uint32
VkDeviceSize = c_uint64

_library = ctypes.WinDLL("vulkan-1")

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


def _function(name: str, restype, *argtypes):
    function = getattr(_library, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


Pointer = c_void_p

vkCreateInstance = _function("vkCreateInstance", VkResult, Pointer, Pointer, POINTER(c_void_p))
vkDestroyInstance = _function("vkDestroyInstance", None, c_void_p, Pointer)
vkEnumerateInstanceLayerProperties = _function("vkEnumerateInstanceLayerProperties", VkResult, POINTER(c_uint32), Pointer)
vkEnumeratePhysicalDevices = _function("vkEnumeratePhysicalDevices", VkResult, c_void_p, POINTER(c_uint32), Pointer)
vkGetPhysicalDeviceProperties = _function("vkGetPhysicalDeviceProperties", None, c_void_p, Pointer)
vkGetPhysicalDeviceFeatures2 = _function("vkGetPhysicalDeviceFeatures2", None, c_void_p, Pointer)
vkEnumerateDeviceExtensionProperties = _function(
    "vkEnumerateDeviceExtensionProperties", VkResult, c_void_p, c_char_p, POINTER(c_uint32), Pointer)
vkGetPhysicalDeviceQueueFamilyProperties = _function(
    "vkGetPhysicalDeviceQueueFamilyProperties", None, c_void_p, POINTER(c_uint32), Pointer)
vkGetPhysicalDeviceMemoryProperties = _function("vkGetPhysicalDeviceMemoryProperties", None, c_void_p, Pointer)
vkGetPhysicalDeviceFormatProperties = _function("vkGetPhysicalDeviceFormatProperties", None, c_void_p, c_int32, Pointer)
vkCreateDevice = _function("vkCreateDevice", VkResult, c_void_p, Pointer, Pointer, POINTER(c_void_p))
vkDestroyDevice = _function("vkDestroyDevice", None, c_void_p, Pointer)
vkDeviceWaitIdle = _function("vkDeviceWaitIdle", VkResult, c_void_p)
vkGetDeviceQueue = _function("vkGetDeviceQueue", None, c_void_p, c_uint32, c_uint32, POINTER(c_void_p))
vkGetDeviceProcAddr = _function("vkGetDeviceProcAddr", c_void_p, c_void_p, c_char_p)

vkCreateImage = _function("vkCreateImage", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
vkDestroyImage = _function("vkDestroyImage", None, c_void_p, c_uint64, Pointer)
vkGetImageMemoryRequirements = _function("vkGetImageMemoryRequirements", None, c_void_p, c_uint64, Pointer)
vkBindImageMemory = _function("vkBindImageMemory", VkResult, c_void_p, c_uint64, c_uint64, VkDeviceSize)
vkCreateBuffer = _function("vkCreateBuffer", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
vkDestroyBuffer = _function("vkDestroyBuffer", None, c_void_p, c_uint64, Pointer)
vkGetBufferMemoryRequirements = _function("vkGetBufferMemoryRequirements", None, c_void_p, c_uint64, Pointer)
vkBindBufferMemory = _function("vkBindBufferMemory", VkResult, c_void_p, c_uint64, c_uint64, VkDeviceSize)
vkAllocateMemory = _function("vkAllocateMemory", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
vkFreeMemory = _function("vkFreeMemory", None, c_void_p, c_uint64, Pointer)
vkMapMemory = _function("vkMapMemory", VkResult, c_void_p, c_uint64, VkDeviceSize, VkDeviceSize, c_uint32, POINTER(c_void_p))
vkUnmapMemory = _function("vkUnmapMemory", None, c_void_p, c_uint64)

vkCreateCommandPool = _function("vkCreateCommandPool", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
vkDestroyCommandPool = _function("vkDestroyCommandPool", None, c_void_p, c_uint64, Pointer)
vkAllocateCommandBuffers = _function("vkAllocateCommandBuffers", VkResult, c_void_p, Pointer, POINTER(c_void_p))
vkBeginCommandBuffer = _function("vkBeginCommandBuffer", VkResult, c_void_p, Pointer)
vkEndCommandBuffer = _function("vkEndCommandBuffer", VkResult, c_void_p)
vkResetCommandBuffer = _function("vkResetCommandBuffer", VkResult, c_void_p, c_uint32)
vkCmdPipelineBarrier = _function(
    "vkCmdPipelineBarrier", None, c_void_p, c_uint32, c_uint32, c_uint32,
    c_uint32, Pointer, c_uint32, Pointer, c_uint32, Pointer)
vkCmdCopyBufferToImage = _function("vkCmdCopyBufferToImage", None, c_void_p, c_uint64, c_uint64, c_int32, c_uint32, Pointer)
vkCmdCopyImageToBuffer = _function("vkCmdCopyImageToBuffer", None, c_void_p, c_uint64, c_int32, c_uint64, c_uint32, Pointer)
vkCmdCopyImage = _function("vkCmdCopyImage", None, c_void_p, c_uint64, c_int32, c_uint64, c_int32, c_uint32, Pointer)
vkCmdClearColorImage = _function("vkCmdClearColorImage", None, c_void_p, c_uint64, c_int32, Pointer, c_uint32, Pointer)
vkQueueSubmit = _function("vkQueueSubmit", VkResult, c_void_p, c_uint32, Pointer, c_uint64)
vkQueueWaitIdle = _function("vkQueueWaitIdle", VkResult, c_void_p)
vkCreateFence = _function("vkCreateFence", VkResult, c_void_p, Pointer, Pointer, POINTER(c_uint64))
vkDestroyFence = _function("vkDestroyFence", None, c_void_p, c_uint64, Pointer)
vkWaitForFences = _function("vkWaitForFences", VkResult, c_void_p, c_uint32, POINTER(c_uint64), VkBool32, c_uint64)
vkResetFences = _function("vkResetFences", VkResult, c_void_p, c_uint32, POINTER(c_uint64))


def address_of_vkGetDeviceProcAddr() -> int:
    """The raw address of the loader's vkGetDeviceProcAddr, handed to the FSR backend."""
    return ctypes.cast(vkGetDeviceProcAddr, c_void_p).value


def pointer(structure) -> int:
    """Address of a ctypes structure, for `pNext` chains and `Pointer` arguments."""
    return ctypes.addressof(structure)


def c_string_array(strings: list[str]):
    """A `const char* const*` array for extension and layer name lists."""
    array = (c_char_p * max(len(strings), 1))()
    for index, text in enumerate(strings):
        array[index] = text.encode("ascii")
    return array


__all__ = [name for name in dir() if name.startswith("vk") or name in ("check", "VkError", "pointer", "c_string_array",
                                                                       "address_of_vkGetDeviceProcAddr", "T", "c_float", "byref")]
