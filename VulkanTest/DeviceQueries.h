#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

#include <optional>

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface);

bool isDeviceSuitable(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface, const std::vector<const char*>& deviceExtensions);

QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface);

vk::Format findSupportedFormat(const vk::PhysicalDevice& physicalDevice, const std::vector<vk::Format>& candidates, const vk::ImageTiling& tiling, const vk::FormatFeatureFlags& features);

vk::Format findDepthFormat(const vk::PhysicalDevice& physicalDevice);

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice, uint32_t typeFilter, const vk::MemoryPropertyFlags& properties);

vk::SampleCountFlagBits getMaxUsableSampleCount(const vk::PhysicalDevice& physicalDevice);