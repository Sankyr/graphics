#include "FrameRenderingInfo.h"
#include "UniformBufferObject.h"

FrameRenderingInfo::FrameRenderingInfo(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::DescriptorPool descriptorPool, const vk::DescriptorSetLayout& descriptorSetLayout, const vk::ImageView &textureImageView, const vk::Sampler &textureSampler) :
        device_(device),
        uniformBufferMapped(nullptr) {
    // Uniform buffer
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    vk::BufferCreateInfo bufferCreateInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eUniformBuffer,
        .sharingMode = vk::SharingMode::eExclusive
    };
    uniformBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, bufferCreateInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    uniformBufferMapped = device.mapMemory(uniformBuffer->memory, 0, bufferSize);

    // Descriptor Set
    vk::DescriptorSetAllocateInfo allocInfo = {
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo bufferInfo = {
        .buffer = uniformBuffer->buffer,
        .offset = 0,
        .range = sizeof(UniformBufferObject)
    };

    vk::DescriptorImageInfo imageInfo = {
        .sampler = textureSampler,
        .imageView = textureImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {
        vk::WriteDescriptorSet{
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &bufferInfo
        },
        {
            .dstSet = descriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imageInfo
        }
    };

    device.updateDescriptorSets(descriptorWrites, 0);

    // Command Buffer
    commandBuffer = device.allocateCommandBuffers({
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
        })[0];

    // Sync objects
    imageAvailableSemaphore = device.createSemaphore({});
    renderFinishedSemaphore = device.createSemaphore({});
    inFlightFence = device.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled });
}

FrameRenderingInfo::~FrameRenderingInfo() {
    device_.destroySemaphore(imageAvailableSemaphore);
    device_.destroySemaphore(renderFinishedSemaphore);
    device_.destroyFence(inFlightFence);
}