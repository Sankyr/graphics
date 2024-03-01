#pragma once

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

struct SwapChainSetupInfo {
    SwapChainSetupInfo() : presentMode() {}
    SwapChainSetupInfo(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface);
    vk::SurfaceCapabilitiesKHR capabilities;
    vk::SurfaceFormatKHR surfaceFormat;
    vk::PresentModeKHR presentMode;
};

