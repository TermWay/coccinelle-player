"""GPU self-test: upload a synthetic image, copy it through a second image and read it back.

Exercises the Vulkan binding end to end (device creation, images, staging buffers, barriers, copies,
submission) without touching the FSR DLLs. Returns 0 when the readback matches the upload.
"""
from __future__ import annotations

import time

import numpy as np

from . import vk
from . import vk_types as T
from .gpu import Gpu


def run(validate: bool = False, verbose: bool = True, width: int = 1920, height: int = 1080) -> int:
    gpu = Gpu(validate=validate, verbose=verbose)
    try:
        usage = T.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | T.VK_IMAGE_USAGE_TRANSFER_DST_BIT | T.VK_IMAGE_USAGE_SAMPLED_BIT
        source = gpu.create_image(width, height, T.VK_FORMAT_R8G8B8A8_UNORM, usage)
        copy = gpu.create_image(width, height, T.VK_FORMAT_R8G8B8A8_UNORM, usage | T.VK_IMAGE_USAGE_STORAGE_BIT)
        upload = gpu.create_staging_buffer(source.byte_size)
        readback = gpu.create_staging_buffer(copy.byte_size)

        rows = np.arange(height, dtype=np.uint32)[:, None]
        columns = np.arange(width, dtype=np.uint32)[None, :]
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[..., 0] = columns & 0xFF
        pixels[..., 1] = rows & 0xFF
        pixels[..., 2] = (columns * 7 + rows * 13) & 0xFF
        pixels[..., 3] = 255

        started = time.perf_counter()
        upload.write(pixels)
        command_buffer = gpu.begin()
        gpu.cmd_upload(upload, source, T.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_READ_BIT)
        gpu.cmd_transition(copy, T.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, T.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                           T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT)
        region = T.VkImageCopy(
            srcSubresource=T.VkImageSubresourceLayers(T.VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            dstSubresource=T.VkImageSubresourceLayers(T.VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            extent=T.VkExtent3D(width, height, 1))
        vk.vkCmdCopyImage(command_buffer, source.handle, source.layout, copy.handle, copy.layout, 1, vk.byref(region))
        gpu.cmd_readback(copy, readback, T.VK_PIPELINE_STAGE_TRANSFER_BIT, T.VK_ACCESS_TRANSFER_WRITE_BIT)
        gpu.submit_and_wait()
        result = np.empty_like(pixels)
        readback.read_into(result)
        elapsed = (time.perf_counter() - started) * 1000

        mismatches = int(np.count_nonzero(result != pixels))
        if verbose:
            print(f"[selftest] {width}x{height} upload + copy + readback: {elapsed:.1f} ms, mismatching bytes: {mismatches}")
        if mismatches:
            print("[selftest] FAILED: readback differs from upload")
            return 1
        print("[selftest] OK")
        return 0
    finally:
        gpu.destroy()


if __name__ == "__main__":
    import sys
    sys.exit(run(validate="--validate" in sys.argv))
