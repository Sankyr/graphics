#pragma once

#include "AllocatedGPUMemory.h"

#include <vulkan/vulkan.hpp>

struct FrameRenderingInfo {
    FrameRenderingInfo(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::DescriptorPool descriptorPool,
        const vk::DescriptorSetLayout& descriptorSetLayout, const vk::DescriptorSetLayout& computeDescriptorSetLayout, const vk::ImageView& textureImageView, const vk::Sampler& textureSampler,
        const std::unique_ptr<AllocatedBuffer>& prevParticleBuffer, const std::unique_ptr<AllocatedBuffer>& currParticleBuffer);
    vk::CommandBuffer commandBuffer;
    vk::CommandBuffer computeCommandBuffer;
    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;
    vk::Semaphore computeFinishedSemaphore;
    vk::Fence computeInFlightFence;
    vk::Fence inFlightFence;
    std::unique_ptr<AllocatedBuffer> uniformBuffer;
    void* uniformBufferMapped;
    std::unique_ptr<AllocatedBuffer> timeBuffer;
    void* timeBufferMapped;
    std::unique_ptr<AllocatedBuffer> particleBuffer;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSet computeDescriptorSet;

    ~FrameRenderingInfo();
private:
    vk::Device device_;
};
