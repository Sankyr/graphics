#pragma once

#include <glm/gtx/hash.hpp>

#include <vulkan/vulkan.h>

#include <vulkan/vulkan.hpp>

#include <array>

struct Particle {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 velocity;
    alignas(16) glm::vec3 acceleration;
    alignas(16) glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
};