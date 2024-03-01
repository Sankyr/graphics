#include "DeviceQueries.h"

#include <iostream>
#include <set>

namespace {
    void printDeviceProperties(const vk::PhysicalDeviceProperties& properties) {
        std::cout << "Device properties: " << std::endl;
        std::cout << "    apiVersion: " << properties.apiVersion << std::endl;
        std::cout << "    deviceID: " << properties.deviceID << std::endl;
        std::cout << "    deviceName: " << properties.deviceName << std::endl;
        // std::cout << "    deviceType: " << properties.deviceType << std::endl;
        std::cout << "    driverVersion: " << properties.driverVersion << std::endl;
        // std::cout << "limits" << properties.limits << std::endl;
        // std::cout << "    pipelineCacheUUID: " << properties.pipelineCacheUUID << std::endl;
        // std::cout << "sparseProperties" << properties.sparseProperties << std::endl;
        std::cout << "    vendorID: " << properties.vendorID << std::endl;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device, const std::vector<const char*>& deviceExtensions) {
        std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

        std::vector<std::string> availableExtentionNames;
        std::transform(availableExtensions.begin(), availableExtensions.end(), std::back_inserter(availableExtentionNames), [](const vk::ExtensionProperties& extension) {return std::string(extension.extensionName.data()); });
        std::set<std::string> availableExtentionNamesSet(availableExtentionNames.begin(), availableExtentionNames.end());
        return std::all_of(deviceExtensions.begin(), deviceExtensions.end(), [availableExtentionNamesSet](std::string extension) { return availableExtentionNamesSet.contains(extension); });
    }
}  // namespace



SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface) {
    SwapChainSupportDetails details;

    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);
    return details;
}

bool isDeviceSuitable(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface, const std::vector<const char*>& deviceExtensions) {
    vk::PhysicalDeviceProperties deviceProperties;
    vk::PhysicalDeviceFeatures deviceFeatures;
    device.getProperties(&deviceProperties);
    device.getFeatures(&deviceFeatures);
    printDeviceProperties(deviceProperties);

    QueueFamilyIndices indices = findQueueFamilies(device, surface);

    bool extensionsSupported = checkDeviceExtensionSupport(device, deviceExtensions);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface) {
    QueueFamilyIndices queueFamilyIndices;

    uint32_t queueFamilyCount = 0;
    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            queueFamilyIndices.graphicsFamily = i;
        }
        if (device.getSurfaceSupportKHR(i, surface)) {
            queueFamilyIndices.presentFamily = i;
        }
        if (queueFamilyIndices.isComplete()) {
            break;
        }

        i++;
    }

    return queueFamilyIndices;
}

vk::Format findSupportedFormat(const vk::PhysicalDevice& physicalDevice, const std::vector<vk::Format>& candidates, const vk::ImageTiling& tiling, const vk::FormatFeatureFlags& features) {
    for (vk::Format format : candidates) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);
        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

vk::Format findDepthFormat(const vk::PhysicalDevice& physicalDevice) {
    return findSupportedFormat(
        physicalDevice,
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice, uint32_t typeFilter, const vk::MemoryPropertyFlags& properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}