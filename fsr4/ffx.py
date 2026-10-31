"""ctypes binding to the AMD FSR API (`amd_fidelityfx_loader_vk.dll`) and the FSR Upscaling context.

Structures mirror `third_party/ffx/*.h` field for field; `tests/test_ffx_structs.py` pins their sizes to the
MSVC x64 layout. Only the upscaler entry points are bound.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_bool, c_char_p, c_float, c_int32, c_uint32, c_uint64, c_void_p, c_wchar_p
from pathlib import Path

from . import log
from . import vk
from . import vk_types as T
from .gpu import Gpu, Image

LOADER_DLL = "amd_fidelityfx_loader_vk.dll"
UPSCALER_DLL = "amd_fidelityfx_upscaler_vk.dll"

# ---- constants (ffx_api.h, ffx_api_types.h, ffx_upscale.h, ffx_api_vk.h) ------------------------

FFX_API_RETURN_OK = 0
RETURN_CODE_NAMES = {
    0: "OK", 1: "ERROR", 2: "ERROR_UNKNOWN_DESCTYPE", 3: "ERROR_RUNTIME_ERROR", 4: "NO_PROVIDER",
    5: "ERROR_MEMORY", 6: "ERROR_PARAMETER", 7: "PROVIDER_NO_SUPPORT_NEW_DESCTYPE",
}

FFX_API_EFFECT_ID_UPSCALE = 0x00010000
FFX_API_BACKEND_ID_VK = 0x01000000
FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE = FFX_API_EFFECT_ID_UPSCALE | 0x00
FFX_API_DISPATCH_DESC_TYPE_UPSCALE = FFX_API_EFFECT_ID_UPSCALE | 0x01
FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE = FFX_API_EFFECT_ID_UPSCALE | 0x02
FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE = FFX_API_EFFECT_ID_UPSCALE | 0x03
FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT = FFX_API_EFFECT_ID_UPSCALE | 0x04
FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET = FFX_API_EFFECT_ID_UPSCALE | 0x05
FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE = FFX_API_EFFECT_ID_UPSCALE | 0x08
FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE = FFX_API_EFFECT_ID_UPSCALE | 0x07
FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION = FFX_API_EFFECT_ID_UPSCALE | 0x0B
FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK = FFX_API_BACKEND_ID_VK | 0x03
FFX_API_QUERY_DESC_TYPE_GET_VERSIONS = 4
FFX_API_DESC_TYPE_OVERRIDE_VERSION = 5
FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION = 6
FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG = 7
FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_SILENCE = 0
FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_ERRORS = 1
FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS = 2
FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE = 0xFFFFFFF

FFX_UPSCALER_VERSION = (4 << 22) | (1 << 12) | 2
# ffxConfigureDescUpscaleKeyValue keys (ffx_upscale.h); the header documents them for FSR 3.1.
UPSCALE_KEYS = {"velocity_factor": 0, "reactiveness_scale": 1, "shading_change_scale": 2,
                "accumulation_added_per_frame": 3, "min_disocclusion_accumulation": 4}

FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE = 1 << 0
FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS = 1 << 1
FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION = 1 << 2
FFX_UPSCALE_ENABLE_DEPTH_INVERTED = 1 << 3
FFX_UPSCALE_ENABLE_DEPTH_INFINITE = 1 << 4
FFX_UPSCALE_ENABLE_AUTO_EXPOSURE = 1 << 5
FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION = 1 << 6
FFX_UPSCALE_ENABLE_DEBUG_CHECKING = 1 << 7
FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE = 1 << 8
FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION = 1 << 9

FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW = 1 << 0
FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB = 1 << 1
FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ = 1 << 2

QUALITY_MODES = {"native": 0, "quality": 1, "balanced": 2, "performance": 3, "ultra": 4}
QUALITY_MODE_SCALES = {"native": 1.0, "quality": 1.5, "balanced": 1.7, "performance": 2.0, "ultra": 3.0}

FFX_API_SURFACE_FORMAT = {
    T.VK_FORMAT_R16G16B16A16_SFLOAT: 4,
    T.VK_FORMAT_R32G32_SFLOAT: 6,
    T.VK_FORMAT_R8G8B8A8_UNORM: 10,
    T.VK_FORMAT_R8G8B8A8_SRGB: 12,
    T.VK_FORMAT_R16G16_SFLOAT: 18,
    T.VK_FORMAT_R32_SFLOAT: 28,
}
FFX_API_RESOURCE_TYPE_TEXTURE2D = 2
FFX_API_RESOURCE_USAGE_READ_ONLY = 0
FFX_API_RESOURCE_USAGE_UAV = 1 << 1
FFX_API_RESOURCE_FLAGS_NONE = 0
FFX_API_RESOURCE_STATE_UNORDERED_ACCESS = 1 << 1
FFX_API_RESOURCE_STATE_COMPUTE_READ = 1 << 2
FFX_API_RESOURCE_STATE_PIXEL_READ = 1 << 3
FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ = FFX_API_RESOURCE_STATE_PIXEL_READ | FFX_API_RESOURCE_STATE_COMPUTE_READ

# ---- structures ---------------------------------------------------------------------------------


class ffxApiHeader(ctypes.Structure):
    _fields_ = [("type", c_uint64), ("pNext", c_void_p)]


class FfxApiDimensions2D(ctypes.Structure):
    _fields_ = [("width", c_uint32), ("height", c_uint32)]


class FfxApiFloatCoords2D(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class FfxApiResourceDescription(ctypes.Structure):
    _fields_ = [
        ("type", c_uint32), ("format", c_uint32), ("width", c_uint32), ("height", c_uint32),
        ("depth", c_uint32), ("mipCount", c_uint32), ("flags", c_uint32), ("usage", c_uint32),
    ]


class FfxApiResource(ctypes.Structure):
    _fields_ = [("resource", c_void_p), ("description", FfxApiResourceDescription), ("state", c_uint32)]


class FfxApiEffectMemoryUsage(ctypes.Structure):
    _fields_ = [("totalUsageInBytes", c_uint64), ("aliasableUsageInBytes", c_uint64)]


class VkDeviceContext(ctypes.Structure):
    _fields_ = [("vkDevice", c_void_p), ("vkPhysicalDevice", c_void_p), ("vkDeviceProcAddr", c_void_p)]


class ffxCreateBackendVKDesc(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("deviceContext", VkDeviceContext)]


ffxApiMessage = ctypes.CFUNCTYPE(None, c_uint32, c_wchar_p)


class ffxCreateContextDescUpscale(ctypes.Structure):
    _fields_ = [
        ("header", ffxApiHeader), ("flags", c_uint32), ("maxRenderSize", FfxApiDimensions2D),
        ("maxUpscaleSize", FfxApiDimensions2D), ("fpMessage", ffxApiMessage),
    ]


class ffxCreateContextDescUpscaleVersion(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("version", c_uint32)]


class ffxDispatchDescUpscale(ctypes.Structure):
    _fields_ = [
        ("header", ffxApiHeader), ("commandList", c_void_p),
        ("color", FfxApiResource), ("depth", FfxApiResource), ("motionVectors", FfxApiResource),
        ("exposure", FfxApiResource), ("reactive", FfxApiResource), ("transparencyAndComposition", FfxApiResource),
        ("output", FfxApiResource),
        ("jitterOffset", FfxApiFloatCoords2D), ("motionVectorScale", FfxApiFloatCoords2D),
        ("renderSize", FfxApiDimensions2D), ("upscaleSize", FfxApiDimensions2D),
        ("enableSharpening", c_bool), ("sharpness", c_float), ("frameTimeDelta", c_float),
        ("preExposure", c_float), ("reset", c_bool), ("cameraNear", c_float), ("cameraFar", c_float),
        ("cameraFovAngleVertical", c_float), ("viewSpaceToMetersFactor", c_float), ("flags", c_uint32),
    ]


class ffxQueryDescGetVersions(ctypes.Structure):
    _fields_ = [
        ("header", ffxApiHeader), ("createDescType", c_uint64), ("device", c_void_p),
        ("outputCount", POINTER(c_uint64)), ("versionIds", POINTER(c_uint64)), ("versionNames", POINTER(c_char_p)),
    ]


class ffxQueryGetProviderVersion(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("versionId", c_uint64), ("versionName", c_char_p)]


class ffxOverrideVersion(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("versionId", c_uint64)]


class ffxConfigureDescUpscaleKeyValue(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("key", c_uint64), ("u64", c_uint64), ("ptr", c_void_p)]


class ffxConfigureDescGlobalDebug(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("effectId", c_uint64), ("fpMessage", ffxApiMessage), ("debugLevel", c_uint32)]


class ffxQueryDescUpscaleGetJitterPhaseCount(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("renderWidth", c_uint32), ("displayWidth", c_uint32),
                ("pOutPhaseCount", POINTER(c_int32))]


class ffxQueryDescUpscaleGetJitterOffset(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("index", c_int32), ("phaseCount", c_int32),
                ("pOutX", POINTER(c_float)), ("pOutY", POINTER(c_float))]


class ffxQueryDescUpscaleGetGPUMemoryUsage(ctypes.Structure):
    _fields_ = [("header", ffxApiHeader), ("gpuMemoryUsageUpscaler", POINTER(FfxApiEffectMemoryUsage))]


# ---- API ----------------------------------------------------------------------------------------


class FfxError(RuntimeError):
    """An FSR API call returned an error code."""

    def __init__(self, code: int, what: str):
        super().__init__(f"{what} failed: FFX_API_RETURN_{RETURN_CODE_NAMES.get(code, code)} ({code})")
        self.code = code


def find_sdk_dir(explicit: str | None = None) -> Path:
    """Locate the folder holding the two FSR DLLs: `--sdk`, then `bin/` next to the exe or the package."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    import sys
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "bin")
        candidates.append(Path(sys.executable).parent)
    candidates.append(Path(__file__).resolve().parent.parent / "bin")
    for candidate in candidates:
        if (candidate / LOADER_DLL).is_file() and (candidate / UPSCALER_DLL).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"{LOADER_DLL} and {UPSCALER_DLL} not found (searched {searched}); run `python tools/fetch_sdk.py`")


class FfxApi:
    """The five FSR API entry points, loaded from the loader DLL."""

    def __init__(self, sdk_dir: Path):
        self.sdk_dir = sdk_dir
        self.library = ctypes.WinDLL(str(sdk_dir / LOADER_DLL))
        self.create_context = self._function("ffxCreateContext", POINTER(c_void_p), c_void_p, c_void_p)
        self.destroy_context = self._function("ffxDestroyContext", POINTER(c_void_p), c_void_p)
        self.configure = self._function("ffxConfigure", POINTER(c_void_p), c_void_p)
        self.query = self._function("ffxQuery", POINTER(c_void_p), c_void_p)
        self.dispatch = self._function("ffxDispatch", POINTER(c_void_p), c_void_p)

    def _function(self, name: str, *argtypes):
        function = getattr(self.library, name)
        function.restype = c_uint32
        function.argtypes = argtypes
        return function


def check(code: int, what: str) -> None:
    if code != FFX_API_RETURN_OK:
        raise FfxError(code, what)


def _resource(image: Image | None, state: int) -> FfxApiResource:
    """Describe a `gpu.Image` for the FSR API; `None` gives an empty (unused) resource."""
    if image is None:
        return FfxApiResource()
    usage = FFX_API_RESOURCE_USAGE_READ_ONLY
    if image.usage & T.VK_IMAGE_USAGE_STORAGE_BIT:
        usage |= FFX_API_RESOURCE_USAGE_UAV
    description = FfxApiResourceDescription(
        type=FFX_API_RESOURCE_TYPE_TEXTURE2D, format=FFX_API_SURFACE_FORMAT[image.format],
        width=image.width, height=image.height, depth=1, mipCount=1, flags=FFX_API_RESOURCE_FLAGS_NONE, usage=usage)
    return FfxApiResource(resource=image.handle, description=description, state=state)


class Upscaler:
    """One FSR Upscaling context on a `Gpu`, for fixed render and output sizes."""

    def __init__(self, gpu: Gpu, render_size: tuple[int, int], upscale_size: tuple[int, int], flags: int,
                 sdk_dir: Path, verbose: bool = False, debug_checking: bool = False, provider: str | None = None):
        self.gpu = gpu
        self.verbose = verbose
        self.render_size = render_size
        self.upscale_size = upscale_size
        self.api = FfxApi(sdk_dir)
        self.messages: list[str] = []
        self._message_callback = ffxApiMessage(self._on_message)
        self.device_context = VkDeviceContext(
            vkDevice=gpu.device, vkPhysicalDevice=gpu.physical_device, vkDeviceProcAddr=vk.address_of_vkGetDeviceProcAddr())
        self.versions = self._query_versions()
        self.context = c_void_p()
        self._create_context(flags | (FFX_UPSCALE_ENABLE_DEBUG_CHECKING if debug_checking else 0), provider)
        self.version_name = self._provider_version()
        if verbose:
            log.emit(f"[fsr] available upscaler versions: {', '.join(name for _, name in self.versions)}")
            log.emit(f"[fsr] context created with {self.version_name}, render {render_size[0]}x{render_size[1]} "
                  f"-> {upscale_size[0]}x{upscale_size[1]}, GPU memory {self.gpu_memory_bytes() / 1048576:.0f} MB")

    def _on_message(self, message_type: int, message: str) -> None:
        kind = "error" if message_type == 0 else "warning"
        self.messages.append(f"{kind}: {message}")
        log.emit(f"[fsr {kind}] {message}")

    def _query_versions(self) -> list[tuple[int, str]]:
        count = c_uint64(0)
        query = ffxQueryDescGetVersions(
            header=ffxApiHeader(type=FFX_API_QUERY_DESC_TYPE_GET_VERSIONS),
            createDescType=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
            device=ctypes.addressof(self.device_context), outputCount=ctypes.pointer(count))
        check(self.api.query(None, byref(query.header)), "ffxQuery(GetVersions count)")
        ids = (c_uint64 * max(count.value, 1))()
        names = (c_char_p * max(count.value, 1))()
        query.versionIds = ctypes.cast(ids, POINTER(c_uint64))
        query.versionNames = ctypes.cast(names, POINTER(c_char_p))
        check(self.api.query(None, byref(query.header)), "ffxQuery(GetVersions)")
        return [(ids[index], names[index].decode("ascii", "replace")) for index in range(count.value)]

    def _create_context(self, flags: int, provider: str | None) -> None:
        """Create the context; `provider` names one of the DLL's upscaler versions ("3.1.6"), else the newest."""
        self._backend = ffxCreateBackendVKDesc(
            header=ffxApiHeader(type=FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK), deviceContext=self.device_context)
        chain = ctypes.addressof(self._backend)
        if provider is not None:
            ids = [version_id for version_id, name in self.versions if name == provider]
            if not ids:
                raise ValueError(f"the FSR DLL has no upscaler {provider}; it has "
                                 f"{', '.join(name for _, name in self.versions)}")
            self._override = ffxOverrideVersion(
                header=ffxApiHeader(type=FFX_API_DESC_TYPE_OVERRIDE_VERSION, pNext=chain), versionId=ids[0])
            chain = ctypes.addressof(self._override)
        self._version = ffxCreateContextDescUpscaleVersion(
            header=ffxApiHeader(type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION, pNext=chain),
            version=FFX_UPSCALER_VERSION)
        self._create = ffxCreateContextDescUpscale(
            header=ffxApiHeader(type=FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, pNext=ctypes.addressof(self._version)),
            flags=flags, maxRenderSize=FfxApiDimensions2D(*self.render_size),
            maxUpscaleSize=FfxApiDimensions2D(*self.upscale_size), fpMessage=self._message_callback)
        check(self.api.create_context(byref(self.context), byref(self._create.header), None), "ffxCreateContext")

    def _provider_version(self) -> str:
        query = ffxQueryGetProviderVersion(header=ffxApiHeader(type=FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION))
        check(self.api.query(byref(self.context), byref(query.header)), "ffxQuery(GetProviderVersion)")
        return query.versionName.decode("ascii", "replace") if query.versionName else f"id {query.versionId:#x}"

    def gpu_memory_bytes(self) -> int:
        usage = FfxApiEffectMemoryUsage()
        query = ffxQueryDescUpscaleGetGPUMemoryUsage(
            header=ffxApiHeader(type=FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE),
            gpuMemoryUsageUpscaler=ctypes.pointer(usage))
        check(self.api.query(byref(self.context), byref(query.header)), "ffxQuery(GPUMemoryUsage)")
        return usage.totalUsageInBytes

    def jitter_phase_count(self) -> int:
        count = c_int32(0)
        query = ffxQueryDescUpscaleGetJitterPhaseCount(
            header=ffxApiHeader(type=FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT),
            renderWidth=self.render_size[0], displayWidth=self.upscale_size[0], pOutPhaseCount=ctypes.pointer(count))
        check(self.api.query(byref(self.context), byref(query.header)), "ffxQuery(GetJitterPhaseCount)")
        return count.value

    def jitter_offset(self, index: int, phase_count: int) -> tuple[float, float]:
        """The Halton sub-pixel offset for frame `index`, in pixels."""
        x = c_float(0.0)
        y = c_float(0.0)
        query = ffxQueryDescUpscaleGetJitterOffset(
            header=ffxApiHeader(type=FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET), index=index,
            phaseCount=phase_count, pOutX=ctypes.pointer(x), pOutY=ctypes.pointer(y))
        check(self.api.query(byref(self.context), byref(query.header)), "ffxQuery(GetJitterOffset)")
        return x.value, y.value

    def dispatch(self, command_buffer: int, color: Image, depth: Image, motion_vectors: Image, output: Image,
                 jitter: tuple[float, float], frame_time_ms: float, reset: bool, sharpness: float | None,
                 flags: int) -> None:
        """Record the upscale into `command_buffer` (inputs in SHADER_READ_ONLY, output in GENERAL)."""
        description = ffxDispatchDescUpscale(
            header=ffxApiHeader(type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE), commandList=command_buffer,
            color=_resource(color, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ),
            depth=_resource(depth, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ),
            motionVectors=_resource(motion_vectors, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ),
            exposure=_resource(None, 0), reactive=_resource(None, 0), transparencyAndComposition=_resource(None, 0),
            output=_resource(output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS),
            jitterOffset=FfxApiFloatCoords2D(*jitter), motionVectorScale=FfxApiFloatCoords2D(1.0, 1.0),
            renderSize=FfxApiDimensions2D(color.width, color.height),
            upscaleSize=FfxApiDimensions2D(output.width, output.height),
            enableSharpening=sharpness is not None, sharpness=sharpness or 0.0, frameTimeDelta=frame_time_ms,
            preExposure=1.0, reset=reset, cameraNear=0.1, cameraFar=1000.0, cameraFovAngleVertical=1.0,
            viewSpaceToMetersFactor=1.0, flags=flags)
        check(self.api.dispatch(byref(self.context), byref(description.header)), "ffxDispatch(Upscale)")

    def configure(self, key: str, value: float) -> None:
        """Set one tuning key (UPSCALE_KEYS) for the following dispatches."""
        number = c_float(value)
        description = ffxConfigureDescUpscaleKeyValue(
            header=ffxApiHeader(type=FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE), key=UPSCALE_KEYS[key],
            ptr=ctypes.cast(ctypes.pointer(number), c_void_p))
        check(self.api.configure(byref(self.context), byref(description.header)), f"ffxConfigure({key})")

    def destroy(self) -> None:
        if self.context.value:
            check(self.api.destroy_context(byref(self.context), None), "ffxDestroyContext")
            self.context = c_void_p()
