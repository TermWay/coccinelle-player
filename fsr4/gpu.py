"""Vulkan device, images, staging buffers and one-shot command submission on top of `vk`.

`Gpu` creates an instance and a logical device with every feature the GPU reports (as the FSR SDK sample
does), including cooperative matrix and 8-bit float support when the driver offers them, which the FSR 4
FP8 model needs on RDNA 4. Images are plain 2D single-mip textures; uploads and readbacks go through
persistently mapped host-coherent staging buffers.
"""
from __future__ import annotations

import ctypes
from ctypes import byref, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass

from . import log
from . import vk
from . import vk_types as T

VENDOR_AMD = 0x1002

WANTED_DEVICE_EXTENSIONS = (
    "VK_KHR_cooperative_matrix",
    "VK_EXT_shader_float8",
)

VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation"

BYTES_PER_PIXEL = {
    T.VK_FORMAT_R8G8B8A8_UNORM: 4,
    T.VK_FORMAT_R8G8B8A8_SRGB: 4,
    T.VK_FORMAT_R16G16_SFLOAT: 4,
    T.VK_FORMAT_R32G32_SFLOAT: 8,
    T.VK_FORMAT_R32_SFLOAT: 4,
    T.VK_FORMAT_R16G16B16A16_SFLOAT: 8,
}


@dataclass
class Image:
    """A 2D GPU image with its memory and the layout the last barrier left it in."""
    handle: int
    memory: int
    width: int
    height: int
    format: int
    usage: int
    layout: int = T.VK_IMAGE_LAYOUT_UNDEFINED

    @property
    def byte_size(self) -> int:
        return self.width * self.height * BYTES_PER_PIXEL[self.format]


@dataclass
class Buffer:
    """A host-visible, host-coherent, persistently mapped buffer."""
    handle: int
    memory: int
    size: int
    mapped: int

    def write(self, data) -> None:
        """Copy `data` (bytes, bytearray, memoryview or a C-contiguous numpy array) into the buffer."""
        view = memoryview(data).cast("B")
        if view.nbytes > self.size:
            raise ValueError(f"{view.nbytes} bytes do not fit in a {self.size}-byte buffer")
        ctypes.memmove(self.mapped, (ctypes.c_char * view.nbytes).from_buffer(view), view.nbytes)

    def read_into(self, destination) -> None:
        """Copy the buffer into `destination` (a writable buffer of at most `size` bytes)."""
        view = memoryview(destination).cast("B")
        if view.nbytes > self.size:
            raise ValueError(f"cannot read {view.nbytes} bytes from a {self.size}-byte buffer")
        ctypes.memmove((ctypes.c_char * view.nbytes).from_buffer(view), self.mapped, view.nbytes)


class Gpu:
    """A Vulkan instance, physical device choice, logical device, queue and command buffer."""

    def __init__(self, gpu_index: int | None = None, validate: bool = False, verbose: bool = False):
        self.verbose = verbose
        self.instance = self._create_instance(validate)
        self.physical_device, self.properties = self._pick_physical_device(gpu_index)
        self.device_name = self.properties.deviceName.decode("utf-8", "replace")
        self.extensions = self._device_extensions()
        self.enabled_extensions = [name for name in WANTED_DEVICE_EXTENSIONS if name in self.extensions]
        self.queue_family = self._pick_queue_family()
        self.memory_properties = T.VkPhysicalDeviceMemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device, byref(self.memory_properties))
        self.features = self._query_features()
        self.device = self._create_device()
        queue = c_void_p()
        vk.vkGetDeviceQueue(self.device, self.queue_family, 0, byref(queue))
        self.queue = queue.value
        self.command_pool = self._create_command_pool()
        self.command_buffer = self._allocate_command_buffer()
        self.fence = self._create_fence()
        self._images: list[Image] = []
        self._buffers: list[Buffer] = []
        if verbose:
            log.emit(f"[gpu] {self.device_name}, Vulkan {self.api_version_text}, driver {self.driver_version_text}")
            log.emit(f"[gpu] device extensions enabled: {', '.join(self.enabled_extensions) or 'none'}")
            log.emit(f"[gpu] cooperative matrix: {self.supports_cooperative_matrix}, shader float8: {self.supports_float8}")

    # ---- creation -------------------------------------------------------------------------------

    def _create_instance(self, validate: bool) -> int:
        application = T.VkApplicationInfo(
            sType=T.VK_STRUCTURE_TYPE_APPLICATION_INFO, pApplicationName=b"fsr4", applicationVersion=1,
            pEngineName=b"fsr4", engineVersion=1, apiVersion=T.VK_MAKE_API_VERSION(0, 1, 3, 0))
        layers = []
        if validate:
            if VALIDATION_LAYER not in self._instance_layers():
                raise RuntimeError(f"{VALIDATION_LAYER} is not installed (it ships with the Vulkan SDK)")
            layers.append(VALIDATION_LAYER)
        layer_names = vk.c_string_array(layers)
        info = T.VkInstanceCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=vk.pointer(application),
            enabledLayerCount=len(layers), ppEnabledLayerNames=layer_names)
        instance = c_void_p()
        vk.check(vk.vkCreateInstance(byref(info), None, byref(instance)), "vkCreateInstance")
        return instance.value

    @staticmethod
    def _instance_layers() -> set[str]:
        count = c_uint32()
        vk.check(vk.vkEnumerateInstanceLayerProperties(byref(count), None), "vkEnumerateInstanceLayerProperties")
        layers = (T.VkLayerProperties * max(count.value, 1))()
        vk.check(vk.vkEnumerateInstanceLayerProperties(byref(count), layers), "vkEnumerateInstanceLayerProperties")
        return {layers[index].layerName.decode("ascii", "replace") for index in range(count.value)}

    def physical_devices(self) -> list[tuple[int, T.VkPhysicalDeviceProperties]]:
        count = c_uint32()
        vk.check(vk.vkEnumeratePhysicalDevices(self.instance, byref(count), None), "vkEnumeratePhysicalDevices")
        if count.value == 0:
            raise RuntimeError("no Vulkan device found")
        handles = (c_void_p * count.value)()
        vk.check(vk.vkEnumeratePhysicalDevices(self.instance, byref(count), handles), "vkEnumeratePhysicalDevices")
        devices = []
        for index in range(count.value):
            properties = T.VkPhysicalDeviceProperties()
            vk.vkGetPhysicalDeviceProperties(handles[index], byref(properties))
            devices.append((handles[index], properties))
        return devices

    def _pick_physical_device(self, gpu_index: int | None):
        devices = self.physical_devices()
        if gpu_index is not None:
            if gpu_index >= len(devices):
                raise RuntimeError(f"--gpu {gpu_index}: only {len(devices)} Vulkan device(s) present")
            return devices[gpu_index]
        ranked = sorted(
            range(len(devices)),
            key=lambda index: (
                devices[index][1].deviceType != T.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                devices[index][1].vendorID != VENDOR_AMD,
                index))
        return devices[ranked[0]]

    def _device_extensions(self) -> set[str]:
        count = c_uint32()
        vk.check(vk.vkEnumerateDeviceExtensionProperties(self.physical_device, None, byref(count), None),
                 "vkEnumerateDeviceExtensionProperties")
        extensions = (T.VkExtensionProperties * max(count.value, 1))()
        vk.check(vk.vkEnumerateDeviceExtensionProperties(self.physical_device, None, byref(count), extensions),
                 "vkEnumerateDeviceExtensionProperties")
        return {extensions[index].extensionName.decode("ascii", "replace") for index in range(count.value)}

    def _pick_queue_family(self) -> int:
        count = c_uint32()
        vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device, byref(count), None)
        families = (T.VkQueueFamilyProperties * max(count.value, 1))()
        vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device, byref(count), families)
        needed = T.VK_QUEUE_GRAPHICS_BIT | T.VK_QUEUE_COMPUTE_BIT
        for index in range(count.value):
            if families[index].queueFlags & needed == needed:
                return index
        raise RuntimeError("no queue family with graphics and compute support")

    def _query_features(self) -> dict:
        """Query every feature struct the device supports; the same chain is passed to vkCreateDevice."""
        features = {
            "base": T.VkPhysicalDeviceFeatures2(sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2),
            "vulkan11": T.VkPhysicalDeviceVulkan11Features(sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES),
            "vulkan12": T.VkPhysicalDeviceVulkan12Features(sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES),
            "vulkan13": T.VkPhysicalDeviceVulkan13Features(sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES),
        }
        if "VK_KHR_cooperative_matrix" in self.enabled_extensions:
            features["cooperative_matrix"] = T.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(
                sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
        if "VK_EXT_shader_float8" in self.enabled_extensions:
            features["float8"] = T.VkPhysicalDeviceShaderFloat8FeaturesEXT(
                sType=T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT)
        chain = list(features.values())
        for current, following in zip(chain, chain[1:]):
            current.pNext = vk.pointer(following)
        vk.vkGetPhysicalDeviceFeatures2(self.physical_device, byref(features["base"]))
        if "cooperative_matrix" in features:
            # Robust buffer access on cooperative matrices is optional and costs performance.
            features["cooperative_matrix"].cooperativeMatrixRobustBufferAccess = 0
        features["base"].features.robustBufferAccess = 0
        return features

    @property
    def supports_cooperative_matrix(self) -> bool:
        return "cooperative_matrix" in self.features and bool(self.features["cooperative_matrix"].cooperativeMatrix)

    @property
    def supports_float8(self) -> bool:
        return "float8" in self.features and bool(self.features["float8"].shaderFloat8CooperativeMatrix)

    def _create_device(self) -> int:
        priority = vk.c_float(1.0)
        queue_info = T.VkDeviceQueueCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=self.queue_family,
            queueCount=1, pQueuePriorities=vk.pointer(priority))
        extension_names = vk.c_string_array(self.enabled_extensions)
        info = T.VkDeviceCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, pNext=vk.pointer(self.features["base"]),
            queueCreateInfoCount=1, pQueueCreateInfos=vk.pointer(queue_info),
            enabledExtensionCount=len(self.enabled_extensions), ppEnabledExtensionNames=extension_names)
        device = c_void_p()
        vk.check(vk.vkCreateDevice(self.physical_device, byref(info), None, byref(device)), "vkCreateDevice")
        return device.value

    def _create_command_pool(self) -> int:
        info = T.VkCommandPoolCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=T.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=self.queue_family)
        pool = c_uint64()
        vk.check(vk.vkCreateCommandPool(self.device, byref(info), None, byref(pool)), "vkCreateCommandPool")
        return pool.value

    def _allocate_command_buffer(self) -> int:
        info = T.VkCommandBufferAllocateInfo(
            sType=T.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=self.command_pool,
            level=T.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1)
        buffer = c_void_p()
        vk.check(vk.vkAllocateCommandBuffers(self.device, byref(info), byref(buffer)), "vkAllocateCommandBuffers")
        return buffer.value

    def _create_fence(self) -> int:
        info = T.VkFenceCreateInfo(sType=T.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
        fence = c_uint64()
        vk.check(vk.vkCreateFence(self.device, byref(info), None, byref(fence)), "vkCreateFence")
        return fence.value

    # ---- info -----------------------------------------------------------------------------------

    @property
    def api_version_text(self) -> str:
        version = self.properties.apiVersion
        return f"{T.VK_API_VERSION_MAJOR(version)}.{T.VK_API_VERSION_MINOR(version)}.{T.VK_API_VERSION_PATCH(version)}"

    @property
    def driver_version_text(self) -> str:
        version = self.properties.driverVersion
        return f"{T.VK_API_VERSION_MAJOR(version)}.{T.VK_API_VERSION_MINOR(version)}.{T.VK_API_VERSION_PATCH(version)}"

    def format_supports(self, format: int, feature_bits: int) -> bool:
        properties = T.VkFormatProperties()
        vk.vkGetPhysicalDeviceFormatProperties(self.physical_device, format, byref(properties))
        return properties.optimalTilingFeatures & feature_bits == feature_bits

    # ---- resources ------------------------------------------------------------------------------

    def _memory_type(self, type_bits: int, required: int) -> int:
        for index in range(self.memory_properties.memoryTypeCount):
            flags = self.memory_properties.memoryTypes[index].propertyFlags
            if type_bits & (1 << index) and flags & required == required:
                return index
        raise RuntimeError(f"no memory type matches bits {type_bits:#x} with properties {required:#x}")

    def _allocate(self, requirements: T.VkMemoryRequirements, properties: int) -> int:
        info = T.VkMemoryAllocateInfo(
            sType=T.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, allocationSize=requirements.size,
            memoryTypeIndex=self._memory_type(requirements.memoryTypeBits, properties))
        memory = c_uint64()
        vk.check(vk.vkAllocateMemory(self.device, byref(info), None, byref(memory)), "vkAllocateMemory")
        return memory.value

    def create_image(self, width: int, height: int, format: int, usage: int) -> Image:
        """Create a 2D optimal-tiling image in device-local memory. Layout starts UNDEFINED."""
        if format not in BYTES_PER_PIXEL:
            raise ValueError(f"unsupported image format {format}")
        info = T.VkImageCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, imageType=T.VK_IMAGE_TYPE_2D, format=format,
            extent=T.VkExtent3D(width, height, 1), mipLevels=1, arrayLayers=1,
            samples=T.VK_SAMPLE_COUNT_1_BIT, tiling=T.VK_IMAGE_TILING_OPTIMAL, usage=usage,
            sharingMode=T.VK_SHARING_MODE_EXCLUSIVE, initialLayout=T.VK_IMAGE_LAYOUT_UNDEFINED)
        handle = c_uint64()
        vk.check(vk.vkCreateImage(self.device, byref(info), None, byref(handle)), "vkCreateImage")
        requirements = T.VkMemoryRequirements()
        vk.vkGetImageMemoryRequirements(self.device, handle.value, byref(requirements))
        memory = self._allocate(requirements, T.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        vk.check(vk.vkBindImageMemory(self.device, handle.value, memory, 0), "vkBindImageMemory")
        image = Image(handle.value, memory, width, height, format, usage)
        self._images.append(image)
        return image

    def create_staging_buffer(self, size: int, readback: bool = False) -> Buffer:
        """Create a host-visible, host-coherent buffer usable as a copy source and destination.

        A readback buffer asks for host-cached memory: reading uncached (write-combined) memory from the
        CPU is an order of magnitude slower than reading cached system memory.
        """
        info = T.VkBufferCreateInfo(
            sType=T.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=size,
            usage=T.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | T.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=T.VK_SHARING_MODE_EXCLUSIVE)
        handle = c_uint64()
        vk.check(vk.vkCreateBuffer(self.device, byref(info), None, byref(handle)), "vkCreateBuffer")
        requirements = T.VkMemoryRequirements()
        vk.vkGetBufferMemoryRequirements(self.device, handle.value, byref(requirements))
        properties = T.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | T.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        if readback:
            try:
                memory = self._allocate(requirements, properties | T.VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
            except RuntimeError:
                memory = self._allocate(requirements, properties)
        else:
            memory = self._allocate(requirements, properties)
        vk.check(vk.vkBindBufferMemory(self.device, handle.value, memory, 0), "vkBindBufferMemory")
        mapped = c_void_p()
        vk.check(vk.vkMapMemory(self.device, memory, 0, T.VK_WHOLE_SIZE, 0, byref(mapped)), "vkMapMemory")
        buffer = Buffer(handle.value, memory, size, mapped.value)
        self._buffers.append(buffer)
        return buffer

    # ---- commands -------------------------------------------------------------------------------

    def begin(self) -> int:
        """Reset and begin the command buffer; returns its handle."""
        vk.check(vk.vkResetCommandBuffer(self.command_buffer, 0), "vkResetCommandBuffer")
        info = T.VkCommandBufferBeginInfo(
            sType=T.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, flags=T.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
        vk.check(vk.vkBeginCommandBuffer(self.command_buffer, byref(info)), "vkBeginCommandBuffer")
        return self.command_buffer

    def submit_and_wait(self) -> None:
        """End the command buffer, submit it and block until the GPU has finished."""
        vk.check(vk.vkEndCommandBuffer(self.command_buffer), "vkEndCommandBuffer")
        command_buffer = c_void_p(self.command_buffer)
        submit = T.VkSubmitInfo(
            sType=T.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=vk.pointer(command_buffer))
        vk.check(vk.vkQueueSubmit(self.queue, 1, byref(submit), self.fence), "vkQueueSubmit")
        fence = c_uint64(self.fence)
        vk.check(vk.vkWaitForFences(self.device, 1, byref(fence), 1, 0xFFFFFFFFFFFFFFFF), "vkWaitForFences")
        vk.check(vk.vkResetFences(self.device, 1, byref(fence)), "vkResetFences")

    def cmd_transition(self, image: Image, new_layout: int, src_stage: int, src_access: int,
                       dst_stage: int, dst_access: int) -> None:
        """Record a layout transition for the whole image and remember the new layout."""
        barrier = T.VkImageMemoryBarrier(
            sType=T.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, srcAccessMask=src_access, dstAccessMask=dst_access,
            oldLayout=image.layout, newLayout=new_layout,
            srcQueueFamilyIndex=T.VK_QUEUE_FAMILY_IGNORED, dstQueueFamilyIndex=T.VK_QUEUE_FAMILY_IGNORED,
            image=image.handle,
            subresourceRange=T.VkImageSubresourceRange(T.VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
        vk.vkCmdPipelineBarrier(self.command_buffer, src_stage, dst_stage, 0, 0, None, 0, None, 1, byref(barrier))
        image.layout = new_layout

    def _copy_region(self, image: Image) -> T.VkBufferImageCopy:
        return T.VkBufferImageCopy(
            bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=T.VkImageSubresourceLayers(T.VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            imageOffset=T.VkOffset3D(0, 0, 0), imageExtent=T.VkExtent3D(image.width, image.height, 1))

    def cmd_upload(self, staging: Buffer, image: Image, final_layout: int, dst_stage: int, dst_access: int) -> None:
        """Record: transition to TRANSFER_DST, copy the staging buffer in, transition to `final_layout`."""
        self.cmd_transition(image, T.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            T.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                            T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT)
        region = self._copy_region(image)
        vk.vkCmdCopyBufferToImage(self.command_buffer, staging.handle, image.handle,
                                  T.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, byref(region))
        self.cmd_transition(image, final_layout, T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT,
                            dst_stage, dst_access)

    def cmd_readback(self, image: Image, staging: Buffer, src_stage: int, src_access: int) -> None:
        """Record: transition to TRANSFER_SRC and copy the image into the staging buffer."""
        self.cmd_transition(image, T.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src_stage, src_access,
                            T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_READ_BIT)
        region = self._copy_region(image)
        vk.vkCmdCopyImageToBuffer(self.command_buffer, image.handle, T.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  staging.handle, 1, byref(region))

    def cmd_clear(self, image: Image, red: float, green: float, blue: float, alpha: float, final_layout: int,
                  dst_stage: int, dst_access: int) -> None:
        """Record: clear the whole image to a constant colour and leave it in `final_layout`."""
        self.cmd_transition(image, T.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            T.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                            T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT)
        value = T.VkClearColorValue()
        value.float32[0], value.float32[1], value.float32[2], value.float32[3] = red, green, blue, alpha
        range_ = T.VkImageSubresourceRange(T.VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        vk.vkCmdClearColorImage(self.command_buffer, image.handle, T.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                byref(value), 1, byref(range_))
        self.cmd_transition(image, final_layout, T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT,
                            dst_stage, dst_access)

    # ---- teardown -------------------------------------------------------------------------------

    def destroy(self) -> None:
        """Wait for the device and free everything created through this object."""
        if self.device is None:
            return
        vk.check(vk.vkDeviceWaitIdle(self.device), "vkDeviceWaitIdle")
        for buffer in self._buffers:
            vk.vkUnmapMemory(self.device, buffer.memory)
            vk.vkDestroyBuffer(self.device, buffer.handle, None)
            vk.vkFreeMemory(self.device, buffer.memory, None)
        for image in self._images:
            vk.vkDestroyImage(self.device, image.handle, None)
            vk.vkFreeMemory(self.device, image.memory, None)
        vk.vkDestroyFence(self.device, self.fence, None)
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)
        self.device = None
