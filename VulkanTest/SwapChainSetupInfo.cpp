#include "SwapChainSetupInfo.h"

namespace {
vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}
}  // namespace

SwapChainSetupInfo::SwapChainSetupInfo(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface) {
    capabilities = device.getSurfaceCapabilitiesKHR(surface);
    surfaceFormat = chooseSwapSurfaceFormat(device.getSurfaceFormatsKHR(surface));
    presentMode = chooseSwapPresentMode(device.getSurfacePresentModesKHR(surface));
}