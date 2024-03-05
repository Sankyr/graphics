#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>

struct AllocatedBuffer {
	AllocatedBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::BufferCreateInfo& bufferInfo, const vk::MemoryPropertyFlags &properties);
	~AllocatedBuffer();
	const vk::BufferCreateInfo& bufferInfo() const { return bufferInfo_; }
	vk::Buffer buffer;
	vk::DeviceMemory memory;
private:
	vk::Device device_;
	vk::BufferCreateInfo bufferInfo_;
};

struct AllocatedImage {
	AllocatedImage(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::ImageCreateInfo& imageInfo, const vk::MemoryPropertyFlags& properties, vk::ImageViewCreateInfo& imageViewInfo);
	~AllocatedImage();
	const vk::ImageCreateInfo& imageInfo() const { return imageInfo_;  }
	vk::Image image;
	vk::DeviceMemory memory;
	vk::ImageView imageView;
private:
	vk::Device device_;
	vk::ImageCreateInfo imageInfo_;
};