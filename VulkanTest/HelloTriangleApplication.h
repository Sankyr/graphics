#pragma once

#include "AllocatedGPUMemory.h"
#include "DeviceQueries.h"
#include "DebugMessenger.h"
#include "FrameRenderingInfo.h"
#include "SwapChainSetupInfo.h"
#include "UniformBufferObject.h"
#include "Vertex.h"
#include "FirstPersonController.h"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <optional>

class HelloTriangleApplication {
public:
    void run();
    ~HelloTriangleApplication();
    float scale = 1;

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    static void windowPosCallback(GLFWwindow* window, int xpos, int ypos);

    static void windowSizeCallback(GLFWwindow* window, int width, int height);

    static void windowRefreshCallback(GLFWwindow* window);

    GLFWwindow* initWindow();

    void initVulkan();

    void loadModel();

    void recreateSwapChain();

    void mainLoop();

    void cleanupSwapChain();

    GLFWwindow* window_;
    vk::Instance instance_;
    std::unique_ptr<DebugMessenger> debugMessenger_;
    vk::SurfaceKHR surface_;

    vk::PhysicalDevice physicalDevice_;
    vk::Device device_;
    QueueFamilyIndices queueFamilyIndices_;
    vk::Queue graphicsComputeQueue_;
    vk::Queue presentQueue_;

    vk::Format swapChainImageFormat_;
    vk::Extent2D swapChainExtent_;
    SwapChainSetupInfo swapChainSetupInfo_;
    vk::SwapchainKHR swapChain_;
    std::vector<vk::Image> swapChainImages_;
    std::vector<vk::ImageView> swapChainImageViews_;

    vk::RenderPass renderPass_;
    vk::DescriptorSetLayout descriptorSetLayout_;
    vk::PipelineLayout pipelineLayout_;
    vk::Pipeline graphicsPipeline_;
    vk::Pipeline particleGraphicsPipeline_;

    vk::DescriptorSetLayout computeDescriptorSetLayout_;
    vk::PipelineLayout computePipelineLayout_;
    vk::Pipeline computePipeline_;

    std::vector<vk::Framebuffer> swapChainFramebuffers_;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unique_ptr<AllocatedBuffer> vertexBuffer_;
    std::unique_ptr<AllocatedBuffer> indexBuffer_;
    vk::DescriptorPool descriptorPool_;
    vk::CommandPool commandPool_;
    std::vector<std::unique_ptr<FrameRenderingInfo>> frameRenderingInfos_;
    
    uint32_t mipLevels;
    std::unique_ptr<AllocatedImage> textureImage_;
    vk::Sampler textureSampler_;
    std::unique_ptr<AllocatedImage> depthImage_;
    vk::Format depthFormat_;
    std::unique_ptr<AllocatedImage> colorImage_;
    vk::SampleCountFlagBits msaaSamples_;
    std::vector<std::unique_ptr<AllocatedBuffer>> particleBuffers_;

    size_t frameNum_ = 0;
    bool framebufferResized_ = false;
    int count = 0;
    // FirstPersonController fpcon_;
};
