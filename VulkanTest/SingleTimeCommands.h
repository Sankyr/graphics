#pragma once

#include <vulkan/vulkan.hpp>

class SingleTimeCommands {
public:
	SingleTimeCommands(const vk::Device& device, const vk::CommandPool& commandPool, const vk::Queue& queue);
	void submit();
	vk::CommandBuffer commandBuffer;
private:
	vk::Device device_;
	vk::CommandPool commandPool_;
	vk::Queue queue_;
};

