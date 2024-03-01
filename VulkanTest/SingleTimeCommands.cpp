#include "SingleTimeCommands.h"

SingleTimeCommands::SingleTimeCommands(
    const vk::Device& device, const vk::CommandPool& commandPool, const vk::Queue& queue) : 
        device_(device), commandPool_(commandPool), queue_(queue) {
    vk::CommandBufferAllocateInfo allocInfo = {
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    commandBuffer.begin(beginInfo);
}

void SingleTimeCommands::submit() {
    commandBuffer.end();

    vk::SubmitInfo submitInfo = {
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer
    };

    queue_.submit(submitInfo);
    queue_.waitIdle();

    device_.freeCommandBuffers(commandPool_, commandBuffer);
}