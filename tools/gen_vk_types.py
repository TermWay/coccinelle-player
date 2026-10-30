"""Generate `fsr4/vk_types.py`: ctypes structures and enum constants for the Vulkan structs the tool uses.

The source is the vendored `third_party/vulkan_core.h` (Khronos, Apache-2.0). Only a whitelist of structs
and enums is emitted, nested by-value structs are pulled in automatically. Run from the repository root:

    python tools/gen_vk_types.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "third_party" / "vulkan_core.h"
OUTPUT = ROOT / "fsr4" / "vk_types.py"

STRUCTS = [
    "VkApplicationInfo",
    "VkInstanceCreateInfo",
    "VkPhysicalDeviceProperties",
    "VkPhysicalDeviceProperties2",
    "VkPhysicalDeviceFeatures",
    "VkPhysicalDeviceFeatures2",
    "VkPhysicalDeviceVulkan11Features",
    "VkPhysicalDeviceVulkan12Features",
    "VkPhysicalDeviceVulkan13Features",
    "VkPhysicalDeviceCooperativeMatrixFeaturesKHR",
    "VkPhysicalDeviceShaderFloat8FeaturesEXT",
    "VkPhysicalDeviceSubgroupSizeControlFeatures",
    "VkExtensionProperties",
    "VkLayerProperties",
    "VkQueueFamilyProperties",
    "VkDeviceQueueCreateInfo",
    "VkDeviceCreateInfo",
    "VkPhysicalDeviceMemoryProperties",
    "VkImageCreateInfo",
    "VkMemoryRequirements",
    "VkMemoryAllocateInfo",
    "VkBufferCreateInfo",
    "VkCommandPoolCreateInfo",
    "VkCommandBufferAllocateInfo",
    "VkCommandBufferBeginInfo",
    "VkImageMemoryBarrier",
    "VkBufferMemoryBarrier",
    "VkMemoryBarrier",
    "VkBufferImageCopy",
    "VkImageCopy",
    "VkSubmitInfo",
    "VkFenceCreateInfo",
    "VkMappedMemoryRange",
    "VkFormatProperties",
    "VkClearColorValue",
    "VkImageSubresourceRange",
]

ENUMS = [
    "VkResult",
    "VkStructureType",
    "VkFormat",
    "VkImageLayout",
    "VkImageType",
    "VkImageTiling",
    "VkSharingMode",
    "VkPhysicalDeviceType",
    "VkCommandBufferLevel",
    "VkQueueFlagBits",
    "VkImageUsageFlagBits",
    "VkBufferUsageFlagBits",
    "VkMemoryPropertyFlagBits",
    "VkAccessFlagBits",
    "VkPipelineStageFlagBits",
    "VkImageAspectFlagBits",
    "VkSampleCountFlagBits",
    "VkCommandPoolCreateFlagBits",
    "VkCommandBufferUsageFlagBits",
    "VkFenceCreateFlagBits",
    "VkDependencyFlagBits",
    "VkFormatFeatureFlagBits",
    "VkImageCreateFlagBits",
]

BASE_TYPES = {
    "uint8_t": "ctypes.c_uint8",
    "int8_t": "ctypes.c_int8",
    "uint16_t": "ctypes.c_uint16",
    "int16_t": "ctypes.c_int16",
    "uint32_t": "ctypes.c_uint32",
    "int32_t": "ctypes.c_int32",
    "uint64_t": "ctypes.c_uint64",
    "int64_t": "ctypes.c_int64",
    "float": "ctypes.c_float",
    "double": "ctypes.c_double",
    "size_t": "ctypes.c_size_t",
    "char": "ctypes.c_char",
}


def parse_header(text: str) -> dict:
    """Collect handles, scalar typedefs, enums, static const flag values, defines and struct bodies."""
    handles: dict[str, str] = {}
    for name in re.findall(r"VK_DEFINE_HANDLE\((\w+)\)", text):
        handles[name] = "ctypes.c_void_p"
    for name in re.findall(r"VK_DEFINE_NON_DISPATCHABLE_HANDLE\((\w+)\)", text):
        handles[name] = "ctypes.c_uint64"

    typedefs: dict[str, str] = {}
    for base, name in re.findall(r"^typedef (\w+) (Vk\w+);", text, re.M):
        typedefs[name] = base

    enums: dict[str, list[tuple[str, str]]] = {}
    for enum_name, body in re.findall(r"typedef enum (\w+) \{(.*?)\} \1;", text, re.S):
        values = []
        for line in body.splitlines():
            match = re.match(r"\s*(\w+) = ([^,]+),?", line)
            if match:
                values.append((match.group(1), match.group(2).strip()))
        enums[enum_name] = values

    flag_consts: dict[str, list[tuple[str, str]]] = {}
    for type_name, name, value in re.findall(r"^static const (\w+) (\w+) = ([^;]+);", text, re.M):
        flag_consts.setdefault(type_name, []).append((name, value.strip()))

    defines: dict[str, int] = {}
    for name, value in re.findall(r"^#define (VK_\w+)\s+(\d+)U?\s*$", text, re.M):
        defines[name] = int(value)

    structs: dict[str, tuple[str, list[tuple[str, str, int | None]]]] = {}
    for kind, name, body in re.findall(r"typedef (struct|union) (\w+) \{(.*?)\} \2;", text, re.S):
        fields = []
        for line in body.splitlines():
            line = line.split("//")[0].strip().rstrip(";").strip()
            if not line:
                continue
            match = re.match(r"(.+?)\s+(\w+)((?:\[\w+\])*)$", line)
            if not match:
                # Bitfields and other exotic members only occur in structs this tool never uses.
                fields = None
                break
            type_text, field_name, array_text = match.groups()
            size = None
            for dimension in re.findall(r"\[(\w+)\]", array_text):
                dimension_size = int(dimension) if dimension.isdigit() else defines[dimension]
                size = dimension_size if size is None else size * dimension_size
            fields.append((" ".join(type_text.split()), field_name, size))
        if fields is not None:
            structs[name] = (kind, fields)

    version = int(re.search(r"#define VK_HEADER_VERSION (\d+)", text).group(1))
    return {
        "handles": handles,
        "typedefs": typedefs,
        "enums": enums,
        "flag_consts": flag_consts,
        "defines": defines,
        "structs": structs,
        "version": version,
    }


def resolve_scalar(type_name: str, info: dict) -> str | None:
    """Map a scalar C type name to a ctypes expression, or None when it is a struct."""
    if type_name in BASE_TYPES:
        return BASE_TYPES[type_name]
    if type_name in info["handles"]:
        return info["handles"][type_name]
    if type_name in info["enums"]:
        return "ctypes.c_int32"
    if type_name.startswith("PFN_"):
        return "ctypes.c_void_p"
    if type_name in info["typedefs"]:
        return resolve_scalar(info["typedefs"][type_name], info)
    if type_name in info["structs"]:
        return None
    raise KeyError(f"unknown type {type_name}")


def field_ctype(type_text: str, info: dict, emit_struct) -> str:
    if type_text == "const char* const*":
        return "ctypes.POINTER(ctypes.c_char_p)"
    if type_text == "const char*":
        return "ctypes.c_char_p"
    if type_text.endswith("*"):
        return "ctypes.c_void_p"
    type_name = type_text.replace("const ", "").strip()
    scalar = resolve_scalar(type_name, info)
    if scalar is not None:
        return scalar
    emit_struct(type_name)
    return type_name


def generate(info: dict) -> str:
    lines = [
        '"""Generated by tools/gen_vk_types.py from third_party/vulkan_core.h '
        f'(VK_HEADER_VERSION {info["version"]}). Do not edit."""',
        "import ctypes",
        "",
    ]
    emitted: set[str] = set()

    def emit_struct(name: str) -> None:
        if name in emitted:
            return
        emitted.add(name)
        kind, fields = info["structs"][name]
        field_lines = []
        for type_text, field_name, array_size in fields:
            ctype = field_ctype(type_text, info, emit_struct)
            if array_size is not None:
                ctype = f"{ctype} * {array_size}"
            field_lines.append(f'        ("{field_name}", {ctype}),')
        base = "ctypes.Structure" if kind == "struct" else "ctypes.Union"
        lines.append(f"class {name}({base}):")
        lines.append("    _fields_ = [")
        lines.extend(field_lines)
        lines.append("    ]")
        lines.append("")

    for name in STRUCTS:
        emit_struct(name)

    lines.append("")
    for enum_name in ENUMS:
        lines.append(f"# {enum_name}")
        known: dict[str, str] = {}
        for value_name, value in info["enums"][enum_name]:
            if value_name.endswith("_MAX_ENUM"):
                continue
            expression = known.get(value, value)
            known[value_name] = value_name
            lines.append(f"{value_name} = {expression}")
        lines.append("")
    for type_name in ("VkPipelineStageFlagBits2", "VkAccessFlagBits2"):
        lines.append(f"# {type_name} (64-bit flags)")
        for value_name, value in info["flag_consts"][type_name]:
            lines.append(f"{value_name} = {value.replace('ULL', '')}")
        lines.append("")
    for name in ("VK_MAX_PHYSICAL_DEVICE_NAME_SIZE", "VK_UUID_SIZE", "VK_MAX_EXTENSION_NAME_SIZE",
                 "VK_MAX_DESCRIPTION_SIZE", "VK_MAX_MEMORY_TYPES", "VK_MAX_MEMORY_HEAPS",
                 "VK_QUEUE_FAMILY_IGNORED", "VK_WHOLE_SIZE"):
        if name in info["defines"]:
            lines.append(f"{name} = {info['defines'][name]}")
    lines.append("VK_QUEUE_FAMILY_IGNORED = 0xFFFFFFFF")
    lines.append("VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF")
    lines.append("VK_SUBPASS_EXTERNAL = 0xFFFFFFFF")
    lines.append("")
    lines.append("def VK_MAKE_API_VERSION(variant, major, minor, patch):")
    lines.append("    return (variant << 29) | (major << 22) | (minor << 12) | patch")
    lines.append("")
    lines.append("")
    lines.append("def VK_API_VERSION_MAJOR(version):")
    lines.append("    return (version >> 22) & 0x7F")
    lines.append("")
    lines.append("")
    lines.append("def VK_API_VERSION_MINOR(version):")
    lines.append("    return (version >> 12) & 0x3FF")
    lines.append("")
    lines.append("")
    lines.append("def VK_API_VERSION_PATCH(version):")
    lines.append("    return version & 0xFFF")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    info = parse_header(HEADER.read_text(encoding="utf-8"))
    OUTPUT.write_text(generate(info), encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT} from header version {info['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
