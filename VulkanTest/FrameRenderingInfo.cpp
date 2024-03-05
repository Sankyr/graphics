#include "FrameRenderingInfo.h"
#include "UniformBufferObject.h"
#include "Particle.h"
#include "SingleTimeCommands.h"

namespace {
const int PARTICLE_COUNT = 1000;

vk::DescriptorSet createDescriptorSet(const vk::Device& device, const vk::DescriptorPool descriptorPool, const vk::DescriptorSetLayout& descriptorSetLayout, const vk::Buffer &uniformBuffer, const vk::ImageView& textureImageView, const vk::Sampler& textureSampler) {
    vk::DescriptorSetAllocateInfo allocInfo = {
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo bufferInfo = {
        .buffer = uniformBuffer,
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
    return descriptorSet;
}

vk::DescriptorSet createComputeDescriptorSet(const vk::Device& device, const vk::DescriptorPool descriptorPool, const vk::DescriptorSetLayout& descriptorSetLayout, const vk::Buffer &uniformBuffer, const vk::DeviceSize& bufferSize, const vk::Buffer& prevParticleBuffer, const vk::Buffer& currParticleBuffer) {
    vk::DescriptorSetAllocateInfo allocInfo = {
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };

    vk::DescriptorSet descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

    vk::DescriptorBufferInfo uniformBufferInfo = {
        .buffer = uniformBuffer,
        .offset = 0,
        .range = sizeof(TimeBufferObject),
    };

    vk::DescriptorBufferInfo storageBufferInfoLastFrame = {
        .buffer = prevParticleBuffer,
        .offset = 0,
        .range = bufferSize,
    };

    vk::DescriptorBufferInfo storageBufferInfoCurrentFrame = {
        .buffer = currParticleBuffer,
        .offset = 0,
        .range = bufferSize
    };

    std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {
        vk::WriteDescriptorSet{
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uniformBufferInfo
        },
        {
            .dstSet = descriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &storageBufferInfoLastFrame
        },
        {
            .dstSet = descriptorSet,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &storageBufferInfoCurrentFrame
        }
    };

    device.updateDescriptorSets(descriptorWrites, nullptr);
    return descriptorSet;
}
}  // namespace

FrameRenderingInfo::FrameRenderingInfo(
    const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::DescriptorPool descriptorPool, 
    const vk::DescriptorSetLayout& descriptorSetLayout, const vk::DescriptorSetLayout& computeDescriptorSetLayout, const vk::ImageView &textureImageView, const vk::Sampler &textureSampler,
    const std::unique_ptr<AllocatedBuffer>& prevParticleBuffer, const std::unique_ptr<AllocatedBuffer>& currParticleBuffer) :
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

    // Time buffer
    vk::DeviceSize timeBufferSize = sizeof(TimeBufferObject);
    vk::BufferCreateInfo timeBufferCreateInfo = {
        .size = timeBufferSize,
        .usage = vk::BufferUsageFlagBits::eUniformBuffer,
        .sharingMode = vk::SharingMode::eExclusive
    };
    timeBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, timeBufferCreateInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    timeBufferMapped = device.mapMemory(timeBuffer->memory, 0, timeBufferSize);

    // Descriptor Set
    descriptorSet = createDescriptorSet(device, descriptorPool, descriptorSetLayout, uniformBuffer->buffer, textureImageView, textureSampler);
    computeDescriptorSet = createComputeDescriptorSet(device, descriptorPool, computeDescriptorSetLayout, timeBuffer->buffer, prevParticleBuffer->bufferInfo().size, prevParticleBuffer->buffer, currParticleBuffer->buffer);

    // Command Buffer
    commandBuffer = device.allocateCommandBuffers({
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
        })[0];
    computeCommandBuffer = device.allocateCommandBuffers({
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
        })[0];

    // Sync objects
    imageAvailableSemaphore = device.createSemaphore({});
    renderFinishedSemaphore = device.createSemaphore({});
    computeFinishedSemaphore = device.createSemaphore({});
    computeInFlightFence = device.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled });
    inFlightFence = device.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled });
}

FrameRenderingInfo::~FrameRenderingInfo() {
    device_.destroySemaphore(imageAvailableSemaphore);
    device_.destroySemaphore(renderFinishedSemaphore);
    device_.destroySemaphore(computeFinishedSemaphore);
    device_.destroyFence(computeInFlightFence);
    device_.destroyFence(inFlightFence);
}