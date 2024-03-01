#pragma once

#include "AllocatedGPUMemory.h"

#include <vulkan/vulkan.hpp>

struct FrameRenderingInfo {
    FrameRenderingInfo(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::DescriptorPool descriptorPool, const vk::DescriptorSetLayout& descriptorSetLayout, const vk::ImageView& textureImageView, const vk::Sampler& textureSampler);
    vk::CommandBuffer commandBuffer;
    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;
    vk::Fence inFlightFence;
    std::unique_ptr<AllocatedBuffer> uniformBuffer;
    void* uniformBufferMapped;
    vk::DescriptorSet descriptorSet;

    ~FrameRenderingInfo();
private:
    vk::Device device_;
};

