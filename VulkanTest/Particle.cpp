#include "Particle.h"

vk::VertexInputBindingDescription Particle::getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription = {
        .binding = 0,
        .stride = sizeof(Particle),
        .inputRate = vk::VertexInputRate::eVertex,
    };

    return bindingDescription;
}

std::array<vk::VertexInputAttributeDescription, 2> Particle::getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Particle, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Particle, color)
        },
        /*{
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Particle, velocity),
        },*/
        
    };

    return attributeDescriptions;
}