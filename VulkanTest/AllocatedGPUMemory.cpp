#include "AllocatedGPUMemory.h"

#include "DeviceQueries.h"

namespace {
vk::DeviceMemory AllocateGPUMemory(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, vk::MemoryPropertyFlags properties, const vk::MemoryRequirements &memRequirements) {
    vk::MemoryAllocateInfo allocInfo = {
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties)
    };

    // NOTE: there's a limit on how many individual allocataions that can be made on the GPU!
    return device.allocateMemory(allocInfo);
}
}  // namespace

AllocatedBuffer::AllocatedBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::BufferCreateInfo &bufferCreateInfo, const vk::MemoryPropertyFlags &properties) : device_(device) {
    buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

    memory = AllocateGPUMemory(device, physicalDevice, properties, memRequirements);

    device.bindBufferMemory(buffer, memory, 0);
}

AllocatedBuffer::~AllocatedBuffer() {
    device_.destroyBuffer(buffer);
    device_.freeMemory(memory);
}

AllocatedImage::AllocatedImage(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::ImageCreateInfo& imageCreateInfo, const vk::MemoryPropertyFlags &properties, vk::ImageViewCreateInfo& imageViewInfo) : device_(device), imageInfo_(imageCreateInfo) {
    image = device.createImage(imageInfo_);

    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

    memory = AllocateGPUMemory(device, physicalDevice, properties, memRequirements);

    device.bindImageMemory(image, memory, 0);

    imageViewInfo.image = image;
    imageView = device.createImageView(imageViewInfo);
}

AllocatedImage::~AllocatedImage() {
    device_.destroyImageView(imageView);
    device_.destroyImage(image);
    device_.freeMemory(memory);
}