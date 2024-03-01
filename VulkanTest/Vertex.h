#pragma once

#include <glm/gtx/hash.hpp>

#include <vulkan/vulkan.h>

#include <vulkan/vulkan.hpp>

#include <array>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions();

    bool operator==(const Vertex& other) const;
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            std::size_t seed = 0;
            seed ^= hash<glm::vec3>()(vertex.pos) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hash<glm::vec3>()(vertex.color) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hash<glm::vec3>()(vertex.normal) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hash<glm::vec2>()(vertex.texCoord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}