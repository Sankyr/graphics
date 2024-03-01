#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
class DebugMessenger {
public:
	DebugMessenger(const vk::Instance& instance);
	static vk::DebugUtilsMessengerCreateInfoEXT CreateDebugMessengerCreateInfo();
	~DebugMessenger();
private:
	vk::Instance instance_;
	vk::DebugUtilsMessengerEXT debugMessenger_;
};

