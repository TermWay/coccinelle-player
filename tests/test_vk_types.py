"""The generated Vulkan structures must match the C layout the driver expects."""
import ctypes
import unittest

from fsr4 import vk_types as T


class VulkanTypesTest(unittest.TestCase):
    def test_sizes(self):
        expected = {
            T.VkApplicationInfo: 48,
            T.VkInstanceCreateInfo: 64,
            T.VkPhysicalDeviceProperties: 824,
            T.VkPhysicalDeviceLimits: 504,
            T.VkPhysicalDeviceFeatures2: 240,
            T.VkPhysicalDeviceVulkan11Features: 64,
            T.VkPhysicalDeviceVulkan12Features: 208,
            T.VkPhysicalDeviceVulkan13Features: 80,
            T.VkPhysicalDeviceCooperativeMatrixFeaturesKHR: 24,
            T.VkPhysicalDeviceShaderFloat8FeaturesEXT: 24,
            T.VkDeviceQueueCreateInfo: 40,
            T.VkDeviceCreateInfo: 72,
            T.VkPhysicalDeviceMemoryProperties: 520,
            T.VkImageCreateInfo: 88,
            T.VkBufferCreateInfo: 56,
            T.VkMemoryAllocateInfo: 32,
            T.VkImageMemoryBarrier: 72,
            T.VkBufferImageCopy: 56,
            T.VkImageCopy: 68,
            T.VkSubmitInfo: 72,
            T.VkClearColorValue: 16,
        }
        for structure, size in expected.items():
            with self.subTest(structure=structure.__name__):
                self.assertEqual(ctypes.sizeof(structure), size)

    def test_enum_values(self):
        self.assertEqual(T.VK_FORMAT_R8G8B8A8_UNORM, 37)
        self.assertEqual(T.VK_FORMAT_R16G16_SFLOAT, 83)
        self.assertEqual(T.VK_FORMAT_R32_SFLOAT, 100)
        self.assertEqual(T.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 5)
        self.assertEqual(T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, 1000567000)
        self.assertEqual(T.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, 1000506000)
        self.assertEqual(T.VK_ERROR_DEVICE_LOST, -4)


if __name__ == "__main__":
    unittest.main()
