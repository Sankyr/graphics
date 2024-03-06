#include "HelloTriangleApplication.h"

#include "SingleTimeCommands.h"
#include "Particle.h"

#include <stb_image.h>
#include <tiny_obj_loader.h>

#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include <set>
#include <cstdint> // Necessary for UINT32_MAX
#include <algorithm> // Necessary for std::min/std::max
#include <fstream>
#include <random>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

#ifdef NDEBUG
    const bool ENABLE_VALIDATION_LAYERS = false;
#else
    const bool ENABLE_VALIDATION_LAYERS = true;
#endif

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

const int MAX_FRAMES_IN_FLIGHT = 2;

const int PARTICLE_COUNT = 1000000;

const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

std::chrono::steady_clock::time_point prevTime;

FirstPersonController fpcon;

void displayExtensions() {
    std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();
    std::cout << "available extensions:\n";

    for (const auto& extension : extensions) {
        std::cout << "    " << extension.extensionName << '\n';
    }
}

bool checkValidationLayerSupport() {
    std::vector<vk::LayerProperties> layerProperties = vk::enumerateInstanceLayerProperties();
    for (const char* layerName : VALIDATION_LAYERS) {
        bool layerFound = false;

        auto hasLayer = [layerName](vk::LayerProperties layerProperties) { return strcmp(layerName, layerProperties.layerName) == 0; };
        if (std::find_if(layerProperties.begin(), layerProperties.end(), hasLayer) == std::end(layerProperties)) {
            return false;
        }
    }

    return true;
}

std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (ENABLE_VALIDATION_LAYERS) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

vk::Instance createInstance() {
    if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo{
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
    };

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    
    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();

        vk::DebugUtilsMessengerCreateInfoEXT debugMessengerInfo = DebugMessenger::CreateDebugMessengerCreateInfo();
        createInfo.pNext = &debugMessengerInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    return vk::createInstance(createInfo);
}

vk::SurfaceKHR createSurface(const vk::Instance& instance, GLFWwindow* window) {
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    return surface;
}

vk::PhysicalDevice pickPhysicalDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface) {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    for (const auto& device : devices) {
        if (isDeviceSuitable(device, surface, deviceExtensions)) {
            return device;
        }
    }

    throw std::runtime_error("failed to find a suitable GPU!");
}

vk::Device createLogicalDevice(const vk::PhysicalDevice& physicalDevice, const QueueFamilyIndices& queueFamilyIndices) {
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        queueCreateInfos.push_back({
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
            });
    }

    vk::PhysicalDeviceFeatures deviceFeatures = {
        .samplerAnisotropy = vk::True
    };

    vk::DeviceCreateInfo createInfo = {
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),

        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),

        .pEnabledFeatures = &deviceFeatures,
    };

    if (ENABLE_VALIDATION_LAYERS) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    return physicalDevice.createDevice(createInfo);
}

vk::Extent2D chooseSwapExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        vk::Extent2D actualExtent = {
            .width = static_cast<uint32_t>(width),
            .height = static_cast<uint32_t>(height)
        };
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        
        return actualExtent;
    }
}

vk::SwapchainKHR createSwapChain(const vk::Device& device, const vk::SurfaceKHR& surface, const SwapChainSetupInfo& swapChainSetupInfo, const vk::Extent2D& extent, const QueueFamilyIndices& queueFamilyIndices) {
    uint32_t imageCount = swapChainSetupInfo.capabilities.minImageCount + 1;
    if (swapChainSetupInfo.capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, swapChainSetupInfo.capabilities.maxImageCount);
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = swapChainSetupInfo.surfaceFormat.format;
    createInfo.imageColorSpace = swapChainSetupInfo.surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    if (queueFamilyIndices.graphicsComputeFamily != queueFamilyIndices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        uint32_t indices[] = { queueFamilyIndices.graphicsComputeFamily.value(), queueFamilyIndices.presentFamily.value() };
        createInfo.pQueueFamilyIndices = indices;
    }
    else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = swapChainSetupInfo.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

    createInfo.presentMode = swapChainSetupInfo.presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    return device.createSwapchainKHR(createInfo);
}

std::vector<vk::ImageView> createImageViews(const vk::Device& device, const std::vector<vk::Image>& swapChainImages, const vk::Format& swapChainImageFormat) {
    std::vector<vk::ImageView> swapChainImageViews(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vk::ImageViewCreateInfo viewInfo{
            .image = swapChainImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainImageFormat,
            .components = {
                .r = vk::ComponentSwizzle::eIdentity,
                .g = vk::ComponentSwizzle::eIdentity,
                .b = vk::ComponentSwizzle::eIdentity,
                .a = vk::ComponentSwizzle::eIdentity,
            },
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        swapChainImageViews[i] = device.createImageView(viewInfo);
    }
    return swapChainImageViews;
}

vk::ShaderModule createShaderModule(const vk::Device& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo = {
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    return device.createShaderModule(createInfo);
}

vk::RenderPass createRenderPass(const vk::Device& device, const vk::Format& swapChainImageFormat, const vk::Format &depthFormat, const vk::SampleCountFlagBits &msaaSamples) {
    vk::AttachmentDescription colorAttachment = {
        .format = swapChainImageFormat,
        .samples = msaaSamples,

        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,

        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,

        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::AttachmentReference colorAttachmentRef = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::AttachmentDescription depthAttachment = {
        .format = depthFormat,
        .samples = msaaSamples,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };

    vk::AttachmentReference depthAttachmentRef = {
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };

    vk::AttachmentDescription colorAttachmentResolve = {
        .format = swapChainImageFormat,
        .samples = vk::SampleCountFlagBits::e1,

        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,

        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,

        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR,
    };

    vk::AttachmentReference colorAttachmentResolveRef = {
        .attachment = 2,
        .layout = vk::ImageLayout::eColorAttachmentOptimal,
    };

    vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = &colorAttachmentResolveRef,
        .pDepthStencilAttachment = &depthAttachmentRef,
    };

    vk::SubpassDependency dependency = {
        .srcSubpass = vk::SubpassExternal,
        .dstSubpass = 0,

        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,

        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
    };
    std::array<vk::AttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
    vk::RenderPassCreateInfo renderPassInfo = {
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,

        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    return device.createRenderPass(renderPassInfo);
}

vk::PipelineLayout createPipelineLayout(const vk::Device &device, const vk::DescriptorSetLayout &descriptorSetLayout) {
    //setup push constants
    vk::PushConstantRange push_constant = {
        //this push constant range is accessible only in the vertex shader
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        //this push constant range starts at the beginning
        .offset = 0,
        //this push constant range takes up the size of a MeshPushConstants struct
        .size = sizeof(float),
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1, // Optional
            .pPushConstantRanges = &push_constant, // Optional
    };

    return device.createPipelineLayout(pipelineLayoutInfo);
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

vk::Pipeline createGraphicsPipeline(const vk::Device& device, const vk::Extent2D& swapChainExtent, const vk::PipelineLayout& pipelineLayout, const vk::RenderPass& renderPass, const vk::SampleCountFlagBits &msaaSamples) {
    auto vertShaderCode = readFile("compiledShaders/shader_vert.spv");
    auto fragShaderCode = readFile("compiledShaders/shader_frag.spv");
    // auto vertShaderCode = readFile("compiledShaders/rayTracer_vert.spv");
    // auto fragShaderCode = readFile("compiledShaders/rayTracer_frag.spv");

    vk::ShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    vk::ShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)swapChainExtent.width,
        .height = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vk::Rect2D scissor = {
        .offset = { 0, 0 },
        .extent = swapChainExtent,
    };

    vk::PipelineViewportStateCreateInfo viewportState = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer = {
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasConstantFactor = 0.0f, // Optional
        .depthBiasClamp = 0.0f, // Optional
        .depthBiasSlopeFactor = 0.0f, // Optional
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::False,
        .minSampleShading = 1.0f, // Optional
        .pSampleMask = nullptr, // Optional
        .alphaToCoverageEnable = vk::False, // Optional
        .alphaToOneEnable = vk::False, // Optional
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eOne, // Optional
        .dstColorBlendFactor = vk::BlendFactor::eZero, // Optional
        .colorBlendOp = vk::BlendOp::eAdd, // Optional
        .srcAlphaBlendFactor = vk::BlendFactor::eOne, // Optional
        .dstAlphaBlendFactor = vk::BlendFactor::eZero, // Optional
        .alphaBlendOp = vk::BlendOp::eAdd, // Optional
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending = {
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy, // Optional
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f} // Optional
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil = {
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,

        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
        .front = {}, // Optional
        .back = {}, // Optional
        .minDepthBounds = 0.0f, // Optional
        .maxDepthBounds = 1.0f, // Optional
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState = {
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo = {
        .stageCount = 2,
        .pStages = shaderStages,

        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil, // Optional
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState, // Optional

        .layout = pipelineLayout,

        .renderPass = renderPass,
        .subpass = 0,

        .basePipelineHandle = VK_NULL_HANDLE, // Optional
        .basePipelineIndex = -1, // Optional
    };

    vk::ResultValue<vk::Pipeline> graphicsPipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);
    if (graphicsPipeline.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    device.destroyShaderModule(fragShaderModule);
    device.destroyShaderModule(vertShaderModule);

    return graphicsPipeline.value;
}


vk::Pipeline createParticleGraphicsPipeline(const vk::Device& device, const vk::Extent2D& swapChainExtent, const vk::PipelineLayout& pipelineLayout, const vk::RenderPass& renderPass, const vk::SampleCountFlagBits& msaaSamples) {
    auto vertShaderCode = readFile("compiledShaders/particles_vert.spv");
    auto fragShaderCode = readFile("compiledShaders/particles_frag.spv");

    vk::ShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
    vk::ShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {
        // .topology = vk::PrimitiveTopology::ePointList,
        .topology = vk::PrimitiveTopology::eLineStrip,
        .primitiveRestartEnable = vk::False,
    };

    vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)swapChainExtent.width,
        .height = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    vk::Rect2D scissor = {
        .offset = { 0, 0 },
        .extent = swapChainExtent,
    };

    vk::PipelineViewportStateCreateInfo viewportState = {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer = {
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasConstantFactor = 0.0f, // Optional
        .depthBiasClamp = 0.0f, // Optional
        .depthBiasSlopeFactor = 0.0f, // Optional
        .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::False,
        .minSampleShading = 1.0f, // Optional
        .pSampleMask = nullptr, // Optional
        .alphaToCoverageEnable = vk::False, // Optional
        .alphaToOneEnable = vk::False, // Optional
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = vk::False,
        .srcColorBlendFactor = vk::BlendFactor::eOne, // Optional
        .dstColorBlendFactor = vk::BlendFactor::eZero, // Optional
        .colorBlendOp = vk::BlendOp::eAdd, // Optional
        .srcAlphaBlendFactor = vk::BlendFactor::eOne, // Optional
        .dstAlphaBlendFactor = vk::BlendFactor::eZero, // Optional
        .alphaBlendOp = vk::BlendOp::eAdd, // Optional
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending = {
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy, // Optional
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = std::array<float, 4> {0.0f, 0.0f, 0.0f, 0.0f} // Optional
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil = {
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,

        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False,
        .front = {}, // Optional
        .back = {}, // Optional
        .minDepthBounds = 0.0f, // Optional
        .maxDepthBounds = 1.0f, // Optional
    };

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicState = {
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo = {
        .stageCount = 2,
        .pStages = shaderStages,

        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil, // Optional
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState, // Optional

        .layout = pipelineLayout,

        .renderPass = renderPass,
        .subpass = 0,

        .basePipelineHandle = VK_NULL_HANDLE, // Optional
        .basePipelineIndex = -1, // Optional
    };

    vk::ResultValue<vk::Pipeline> graphicsPipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);
    if (graphicsPipeline.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    device.destroyShaderModule(fragShaderModule);
    device.destroyShaderModule(vertShaderModule);

    return graphicsPipeline.value;
}

vk::Pipeline createComputePipeline(const vk::Device& device, const vk::PipelineLayout& pipelineLayout) {
    auto compShaderCode = readFile("compiledShaders/shader_comp.spv");

    vk::ShaderModule compShaderModule = createShaderModule(device, compShaderCode);

    vk::PipelineShaderStageCreateInfo compShaderStageInfo = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = compShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = { compShaderStageInfo };

    vk::ComputePipelineCreateInfo pipelineInfo = {
        .stage = compShaderStageInfo,
        .layout = pipelineLayout,
    };

    vk::ResultValue<vk::Pipeline> computePipeline = device.createComputePipeline(VK_NULL_HANDLE, pipelineInfo);
    if (computePipeline.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    device.destroyShaderModule(compShaderModule);

    return computePipeline.value;
}

std::vector<vk::Framebuffer> createFramebuffers(const vk::Device& device, const std::vector<vk::ImageView>& swapChainImageViews, const vk::ImageView& colorImageView, const vk::ImageView &depthImageView, const vk::RenderPass& renderPass, const vk::Extent2D& swapChainExtent) {
    std::vector<vk::Framebuffer> swapChainFramebuffers(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        std::array<vk::ImageView, 3> attachments = {
            colorImageView,
            depthImageView,
            swapChainImageViews[i],
        };

        vk::FramebufferCreateInfo framebufferInfo = {
            .renderPass = renderPass,
            .attachmentCount = static_cast<uint32_t>(attachments.size()),
            .pAttachments = attachments.data(),
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1,
        };

        swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
    }
    return swapChainFramebuffers;
}

vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device) {
    vk::DescriptorSetLayoutBinding uboLayoutBinding = {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,

        .stageFlags = vk::ShaderStageFlagBits::eVertex,

        .pImmutableSamplers = nullptr // Optional
    };

    vk::DescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };

    return device.createDescriptorSetLayout(layoutInfo);
}

vk::DescriptorSetLayout createComputeDescriptorSetLayout(const vk::Device& device) {
    std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings = {
        vk::DescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .pImmutableSamplers = nullptr,
        }
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {
        .bindingCount = layoutBindings.size(),
        .pBindings = layoutBindings.data()
    };

    return device.createDescriptorSetLayout(layoutInfo);
}

bool hasStencilComponent(const vk::Format& format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void transitionImageLayout(SingleTimeCommands singleTimeCommands, const vk::Image& image, const vk::ImageLayout& oldLayout, const vk::ImageLayout& newLayout, const uint32_t& mipLevel, const vk::Format& format) {
    vk::ImageMemoryBarrier barrier = {
        .oldLayout = oldLayout,
        .newLayout = newLayout,

        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,

        .image = image,
        .subresourceRange = {
            .baseMipLevel = 0,
            .levelCount = mipLevel,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    // barrier.subresourceRange.aspectMask
    if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    }

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    singleTimeCommands.commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    singleTimeCommands.submit();
}

std::unique_ptr<AllocatedImage> createDepthResources(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::Format& depthFormat, const vk::Extent2D& swapChainExtent, const vk::SampleCountFlagBits& msaaSamples, const vk::CommandPool& commandPool, const vk::Queue& graphicsQueue) {
    vk::ImageCreateInfo imageInfo = {
        .imageType = vk::ImageType::e2D,
        .format = depthFormat,
        .extent = {
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .depth = 1,
            },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaaSamples,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    vk::ImageViewCreateInfo viewInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = depthFormat,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    std::unique_ptr<AllocatedImage> depthImage = std::make_unique<AllocatedImage>(physicalDevice, device, imageInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, viewInfo);

    transitionImageLayout(SingleTimeCommands(device, commandPool, graphicsQueue), depthImage->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1, depthFormat);
    return depthImage;
}

std::unique_ptr<AllocatedImage> createColorResources(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::Format& swapChainImageFormat, const vk::Extent2D& swapChainExtent, const vk::SampleCountFlagBits &msaaSamples, const vk::CommandPool& commandPool, const vk::Queue& graphicsQueue) {
    vk::ImageCreateInfo imageInfo = {
        .imageType = vk::ImageType::e2D,
        .format = swapChainImageFormat,
        .extent = {
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .depth = 1,
            },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaaSamples,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    vk::ImageViewCreateInfo viewInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = swapChainImageFormat,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    std::unique_ptr<AllocatedImage> colorImage = std::make_unique<AllocatedImage>(physicalDevice, device, imageInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, viewInfo);

    // transitionImageLayout(SingleTimeCommands(device, commandPool, graphicsQueue), colorImage->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1, swapChainImageFormat);
    return colorImage;
}

void copyBufferToImage(SingleTimeCommands singleTimeCommands, const vk::Buffer& buffer, const VkImage& image, const uint32_t& width, const uint32_t& height) {
    vk::BufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = {
            width,
            height,
            1
        },
    };
    singleTimeCommands.commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    singleTimeCommands.submit();
}

void generateMipmaps(const vk::PhysicalDevice& physicalDevice, SingleTimeCommands singleTimeCommands, const vk::Image& image, const vk::Format& imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
    // Check if image format supports linear blitting
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);
    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    vk::ImageMemoryBarrier barrier = {
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        singleTimeCommands.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

        vk::ImageBlit blit = {
            .srcSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i - 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{ 0, 0, 0 },
                { mipWidth, mipHeight, 1 }
            },
            .dstSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = i,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffsets = std::array<vk::Offset3D, 2>{
                vk::Offset3D{ 0, 0, 0 },
                { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 }
            },
        };

        singleTimeCommands.commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        singleTimeCommands.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    singleTimeCommands.commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

    singleTimeCommands.submit();
}

std::unique_ptr<AllocatedImage> createTextureImage(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::Queue& graphicsQueue) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    vk::BufferCreateInfo bufferInfo = {
        .size = imageSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };
    AllocatedBuffer stagingBuffer = AllocatedBuffer(physicalDevice, device, bufferInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBuffer.memory);

    stbi_image_free(pixels);

    vk::ImageCreateInfo imageInfo = {
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Srgb,
        .extent = {
            .width = static_cast<uint32_t>(texWidth),
            .height = static_cast<uint32_t>(texHeight),
            .depth = 1,
            },
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    vk::ImageViewCreateInfo viewInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8G8B8A8Srgb,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    std::unique_ptr<AllocatedImage> allocatedImage = std::make_unique<AllocatedImage>(physicalDevice, device, imageInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, viewInfo);

    transitionImageLayout(SingleTimeCommands(device, commandPool, graphicsQueue), allocatedImage->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels, vk::Format::eR8G8B8A8Srgb);
    copyBufferToImage(SingleTimeCommands(device, commandPool, graphicsQueue), stagingBuffer.buffer, allocatedImage->image, texWidth, texHeight);
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps
    generateMipmaps(physicalDevice, SingleTimeCommands(device, commandPool, graphicsQueue), allocatedImage->image, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    return allocatedImage;
}

vk::Sampler createTextureSampler(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, float mipLevels) {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

    vk::SamplerCreateInfo samplerInfo = {
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,

        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,

        .mipLodBias = 0.0f,

        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,

        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0,
        .maxLod = mipLevels,

        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False,
    };

    return device.createSampler(samplerInfo);
}

void copyBuffer(SingleTimeCommands singleTimeCommands, const vk::Buffer& srcBuffer, const vk::Buffer& dstBuffer, const vk::DeviceSize& size) {
    vk::BufferCopy copyRegion{
        .srcOffset = 0, // Optional
        .dstOffset = 0, // Optional
        .size = size
    };
    singleTimeCommands.commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
    singleTimeCommands.submit();
}

std::unique_ptr<AllocatedBuffer> createVertexBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::Queue& graphicsQueue, const std::vector<Vertex>& vertices) {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    vk::BufferCreateInfo stagingBufferInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };
    AllocatedBuffer stagingBuffer(physicalDevice, device, stagingBufferInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = device.mapMemory(stagingBuffer.memory, 0, bufferSize, {});
    memcpy(data, vertices.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBuffer.memory);

    vk::BufferCreateInfo vertexBufferInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        .sharingMode = vk::SharingMode::eExclusive
    };
    std::unique_ptr<AllocatedBuffer> vertexBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, vertexBufferInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(SingleTimeCommands(device, commandPool, graphicsQueue), stagingBuffer.buffer, vertexBuffer->buffer, bufferSize);
    return vertexBuffer;
}

std::unique_ptr<AllocatedBuffer> createIndexBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const vk::CommandPool& commandPool, const vk::Queue& graphicsQueue, const std::vector<uint32_t>& indices) {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
    vk::BufferCreateInfo stagingBufferInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };
    AllocatedBuffer stagingBuffer(physicalDevice, device, stagingBufferInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = device.mapMemory(stagingBuffer.memory, 0, bufferSize, {});
    memcpy(data, indices.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBuffer.memory);

    vk::BufferCreateInfo vertexBufferInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        .sharingMode = vk::SharingMode::eExclusive
    };
    std::unique_ptr<AllocatedBuffer> indexBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, vertexBufferInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(SingleTimeCommands(device, commandPool, graphicsQueue), stagingBuffer.buffer, indexBuffer->buffer, bufferSize);
    return indexBuffer;
}

vk::DescriptorPool createDescriptorPool(const vk::Device& device) {
    std::array<vk::DescriptorPoolSize, 3> poolSizes = {
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2
        },
        {
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT
        },
        /*{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT
        },*/
        {
            .type = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2
        }
    };

    vk::DescriptorPoolCreateInfo poolInfo = {
        .maxSets = MAX_FRAMES_IN_FLIGHT * 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    return device.createDescriptorPool(poolInfo);
}

void recordCommandBuffer(
    const vk::CommandBuffer& commandBuffer, const vk::RenderPass& renderPass,
    const vk::Framebuffer& swapChainFramebuffer, const vk::Extent2D& swapChainExtent, const vk::Pipeline& graphicsPipeline, const vk::Pipeline& particleGraphicsPipeline,
    const vk::Buffer& vertexBuffer, const vk::Buffer& indexBuffer, const std::vector<uint32_t>& indices, const vk::PipelineLayout& pipelineLayout, const vk::PipelineLayout &computePipelineLayout, const vk::DescriptorSet& descriptorSet, const vk::Buffer &particleBuffer, float scale) {
    vk::CommandBufferBeginInfo beginInfo = {};
    commandBuffer.begin(beginInfo);

    std::array<vk::ClearValue, 2> clearValues = {
        vk::ClearValue{.color = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f} },
        vk::ClearValue{.depthStencil = {.depth = 1.0f, .stencil = 0 }}
    };
    vk::RenderPassBeginInfo renderPassInfo = {
        .renderPass = renderPass,
        .framebuffer = swapChainFramebuffer,

        .renderArea = {
            .offset = {.x = 0, .y = 0 },
            .extent = swapChainExtent,
        },
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data(),
    };

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    commandBuffer.setViewport(0, vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)swapChainExtent.width,
        .height = (float)swapChainExtent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
        });
    commandBuffer.setScissor(0, vk::Rect2D{
        .offset = { 0, 0 },
        .extent = swapChainExtent
        });

    vk::Buffer vertexBuffers[] = { vertexBuffer };
    vk::DeviceSize offsets[] = { 0 };
    commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, static_cast<uint32_t>(0), descriptorSet, {});
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    /*commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, particleGraphicsPipeline);
    commandBuffer.pushConstants(computePipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(float), &scale);

    commandBuffer.bindVertexBuffers(0, 1, &particleBuffer, offsets);
    commandBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
    */
    commandBuffer.endRenderPass();

    commandBuffer.end();
}

void recordComputeCommandBuffer(
    const vk::CommandBuffer& commandBuffer, const vk::Pipeline& computePipeline, const vk::PipelineLayout& computePipelineLayout, const vk::DescriptorSet& computeDescriptorSet) {
    vk::CommandBufferBeginInfo beginInfo = {};
    commandBuffer.begin(beginInfo);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, computeDescriptorSet, {});
    commandBuffer.dispatch(PARTICLE_COUNT / 256 + 1, 1, 1);

    commandBuffer.end();
}

void updateUniformBuffer(const vk::Device device, const vk::Extent2D& swapChainExtent, void* uniformBuffersMapped, void* timeBuffersMapped) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - prevTime).count();

    UniformBufferObject ubo{};
    // ubo.model = glm::rotate(glm::rotate(glm::mat4(1.0f), time * glm::radians(15.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(270.0f), glm::vec3(1, 0, 0));
    ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(1, 0, 0));
    ubo.view = fpcon.getViewMatrix();
    // .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;
    /*ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;*/

    //std::cout << deltaTime << std::endl;
    TimeBufferObject timeObj = { .time = time, .deltaTime = deltaTime };
    memcpy(uniformBuffersMapped, &ubo, sizeof(ubo));
    memcpy(timeBuffersMapped, &timeObj, sizeof(TimeBufferObject));
    prevTime = currentTime;
}

bool drawFrame(const vk::Device& device, const vk::SwapchainKHR& swapChain, const vk::Queue& graphicsComputeQueue, const vk::Queue& presentQueue, const vk::Extent2D& swapChainExtent, const std::unique_ptr<FrameRenderingInfo>& frameRenderingInfo,
    const vk::RenderPass& renderPass, const std::vector<vk::Framebuffer>& swapChainFrameBuffers, const vk::Pipeline& graphicsPipeline, const vk::Pipeline &particleGraphicsPipeline, const vk::Pipeline& computePipeline,
    const vk::Buffer& vertexBuffer, const vk::Buffer& indexBuffer, const std::vector<uint32_t>& indices, const vk::PipelineLayout& pipelineLayout, const vk::PipelineLayout& computePipelineLayout, const vk::Buffer &particleBuffer, float scale) {
    assert(device.waitForFences({frameRenderingInfo->computeInFlightFence}, vk::True, UINT64_MAX) == vk::Result::eSuccess);
    device.resetFences({ frameRenderingInfo->computeInFlightFence });

    updateUniformBuffer(device, swapChainExtent, frameRenderingInfo->uniformBufferMapped, frameRenderingInfo->timeBufferMapped);
    vkResetCommandBuffer(frameRenderingInfo->computeCommandBuffer, 0);
    recordComputeCommandBuffer(frameRenderingInfo->computeCommandBuffer, computePipeline, computePipelineLayout, frameRenderingInfo->computeDescriptorSet);

    vk::SubmitInfo computeSubmitInfo = {
        .commandBufferCount = 1,
        .pCommandBuffers = &frameRenderingInfo->computeCommandBuffer,

        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &frameRenderingInfo->computeFinishedSemaphore,
    };
    graphicsComputeQueue.submit(computeSubmitInfo, frameRenderingInfo->computeInFlightFence);

    assert(device.waitForFences({ frameRenderingInfo->inFlightFence }, vk::True, UINT64_MAX) == vk::Result::eSuccess);

    uint32_t imageIndex;
    try {
        vk::ResultValue<uint32_t> resultValue = device.acquireNextImageKHR(swapChain, UINT64_MAX, frameRenderingInfo->imageAvailableSemaphore, VK_NULL_HANDLE);
        imageIndex = resultValue.value;
    } catch (vk::OutOfDateKHRError error) {
        // recreateSwapChain
        return true;
    }

    device.resetFences({ frameRenderingInfo->inFlightFence });
    vkResetCommandBuffer(frameRenderingInfo->commandBuffer, 0);
    recordCommandBuffer(frameRenderingInfo->commandBuffer, renderPass, swapChainFrameBuffers[imageIndex], swapChainExtent, graphicsPipeline, particleGraphicsPipeline, vertexBuffer, indexBuffer, indices, pipelineLayout, computePipelineLayout, frameRenderingInfo->descriptorSet, particleBuffer, scale);
    // updateUniformBuffer(device, swapChainExtent, frameRenderingInfo->uniformBufferMapped);

    vk::Semaphore waitSemaphores[] = { frameRenderingInfo->computeFinishedSemaphore, frameRenderingInfo->imageAvailableSemaphore};
    vk::Semaphore signalSemaphores[] = { frameRenderingInfo->renderFinishedSemaphore };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eColorAttachmentOutput };
    vk::SubmitInfo submitInfo = {
        .waitSemaphoreCount = 2,
        .pWaitSemaphores = waitSemaphores,
        .pWaitDstStageMask = waitStages,

        .commandBufferCount = 1,
        .pCommandBuffers = &frameRenderingInfo->commandBuffer,

        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signalSemaphores,
    };

    graphicsComputeQueue.submit(submitInfo, frameRenderingInfo->inFlightFence);

    vk::PresentInfoKHR presentInfo = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signalSemaphores,

        .swapchainCount = 1,
        .pSwapchains = &swapChain,
        .pImageIndices = &imageIndex,

        .pResults = nullptr, // Optional
    };

    vk::Result result;
    try {
        result = presentQueue.presentKHR(presentInfo);
    } catch (vk::OutOfDateKHRError error) {
        // recreateSwapChain
        return true;
    }
    if (result == vk::Result::eSuboptimalKHR) {
        // recreateSwapChain
        return true;
    }
    return false;
}

std::unique_ptr<AllocatedBuffer> initializeParticles(const vk::PhysicalDevice& physicalDevice, const vk::Device& device) {
    // Initialize particles
    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

    // Initial particle positions on a circle
    std::vector<Particle> particles(PARTICLE_COUNT);
    int i = 0;
    int width = sqrt(PARTICLE_COUNT);
    for (Particle& particle : particles) {
        float r = 1.0f * sqrt(rndDist(rndEngine));
        float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846;
        float x = r * cos(theta);
        float y = r * sin(theta);
        particle.position = glm::vec3(x, y, 0);
        particle.velocity = glm::vec3(0, 0, 0); // glm::normalize(glm::vec3(x, y, 0)) * 0.00025f;
        particle.acceleration = glm::vec3(0, 0, 0);
        particle.color = glm::vec3((i % width) / static_cast<float>(width), (i / width)  / static_cast<float>(width), rndDist(rndEngine)); // rndDist(rndEngine)
        ++i;
    }

    VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
    vk::BufferCreateInfo bufferCreateInfo = {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };
    std::unique_ptr<AllocatedBuffer> stagingBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, bufferCreateInfo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = device.mapMemory(stagingBuffer->memory, 0, bufferSize, {});
    memcpy(data, particles.data(), (size_t)bufferSize);
    device.unmapMemory(stagingBuffer->memory);
    return stagingBuffer;
}

std::unique_ptr<AllocatedBuffer> createParticleBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device, const std::unique_ptr<AllocatedBuffer> &stagingParticleBuffer, SingleTimeCommands singleTimeCommands) {
    vk::BufferCreateInfo particleBufferCreateInfo = {
            .size = stagingParticleBuffer->bufferInfo().size,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive
    };
    std::unique_ptr<AllocatedBuffer> particleBuffer = std::make_unique<AllocatedBuffer>(physicalDevice, device, particleBufferCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::BufferCopy copyRegion{
        .srcOffset = 0, // Optional
        .dstOffset = 0, // Optional
        .size = stagingParticleBuffer->bufferInfo().size
    };
    singleTimeCommands.commandBuffer.copyBuffer(stagingParticleBuffer->buffer, particleBuffer->buffer, copyRegion);
    singleTimeCommands.submit();
    return particleBuffer;
}
}  // namespace

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

/*const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
};

const std::vector<Vertex> vertices = {
    {{-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
    {{-1.05f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};*/

void HelloTriangleApplication::run() {
    window_ = initWindow();
    initVulkan();
    mainLoop();
}

// begin static
void HelloTriangleApplication::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized_ = true;
    // app->drawFrame(); // resize buffers
    // app->drawFrame(); // draw frame
    // std::cout << app->count << " window refresh" << std::endl;
    // app->count++;
}

void HelloTriangleApplication::windowPosCallback(GLFWwindow* window, int xpos, int ypos) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    std::cout << app->count << " window pos" << std::endl;
    app->count++;
}

void HelloTriangleApplication::windowSizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    std::cout << app->count << " window size" << std::endl;
    app->count++;
}

void HelloTriangleApplication::windowRefreshCallback(GLFWwindow* window) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    std::cout << app->count << " window refresh" << std::endl;
    app->count++;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset){
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0) {
        app->scale *= 1.01 * yoffset;
    }
    if (yoffset < 0) {
        app->scale *= .99 * abs(yoffset);
    }
}

void cursorCallback(GLFWwindow* window, double xpos, double ypos) {
    fpcon.onMouseDrag(window, xpos, ypos);
}
// end static
    
GLFWwindow* HelloTriangleApplication::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    // glfwSetWindowRefreshCallback(window_, windowRefreshCallback);
    // glfwSetWindowPosCallback(window_, windowPosCallback);
    // glfwSetWindowSizeCallback(window_, windowSizeCallback);
    return window;
}

void HelloTriangleApplication::initVulkan() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    displayExtensions();
    instance_ = createInstance();
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
    if (ENABLE_VALIDATION_LAYERS) {
        debugMessenger_ = std::make_unique<DebugMessenger>(instance_);
    }
    surface_ = createSurface(instance_, window_);
    physicalDevice_ = pickPhysicalDevice(instance_, surface_);
    msaaSamples_ = getMaxUsableSampleCount(physicalDevice_);
    std::cout << static_cast<int>(msaaSamples_) << std::endl;

    queueFamilyIndices_ = findQueueFamilies(physicalDevice_, surface_);
    device_ = createLogicalDevice(physicalDevice_, queueFamilyIndices_);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

    graphicsComputeQueue_ = device_.getQueue(queueFamilyIndices_.graphicsComputeFamily.value(), 0);
    presentQueue_ = device_.getQueue(queueFamilyIndices_.presentFamily.value(), 0);

    swapChainSetupInfo_ = SwapChainSetupInfo(physicalDevice_, surface_);
    swapChainImageFormat_ = swapChainSetupInfo_.surfaceFormat.format;
    swapChainExtent_ = chooseSwapExtent(window_, swapChainSetupInfo_.capabilities);
    swapChain_ = createSwapChain(device_, surface_, swapChainSetupInfo_, swapChainExtent_, queueFamilyIndices_);
    swapChainImages_ = device_.getSwapchainImagesKHR(swapChain_);
    swapChainImageViews_ = createImageViews(device_, swapChainImages_, swapChainImageFormat_);

    depthFormat_ = findDepthFormat(physicalDevice_);
    renderPass_ = createRenderPass(device_, swapChainImageFormat_, depthFormat_, msaaSamples_);
    descriptorSetLayout_ = createDescriptorSetLayout(device_);
    computeDescriptorSetLayout_ = createComputeDescriptorSetLayout(device_);
    pipelineLayout_ = createPipelineLayout(device_, descriptorSetLayout_);
    graphicsPipeline_ = createGraphicsPipeline(device_, swapChainExtent_, pipelineLayout_, renderPass_, msaaSamples_);
    particleGraphicsPipeline_ = createParticleGraphicsPipeline(device_, swapChainExtent_, pipelineLayout_, renderPass_, msaaSamples_);
    computePipelineLayout_ = createPipelineLayout(device_, computeDescriptorSetLayout_);
    computePipeline_ = createComputePipeline(device_, computePipelineLayout_);
    commandPool_ = device_.createCommandPool({
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices_.graphicsComputeFamily.value(),
        });
    
    depthImage_ = createDepthResources(physicalDevice_, device_, depthFormat_, swapChainExtent_, msaaSamples_, commandPool_, graphicsComputeQueue_);
    colorImage_ = createColorResources(physicalDevice_, device_, swapChainImageFormat_, swapChainExtent_, msaaSamples_, commandPool_, graphicsComputeQueue_);
    swapChainFramebuffers_ = createFramebuffers(device_, swapChainImageViews_, colorImage_->imageView, depthImage_->imageView, renderPass_, swapChainExtent_);
    textureImage_ = createTextureImage(physicalDevice_, device_, commandPool_, graphicsComputeQueue_);
    textureSampler_ = createTextureSampler(physicalDevice_, device_, static_cast<float>(textureImage_->imageInfo().mipLevels));
    loadModel();
    vertexBuffer_ = createVertexBuffer(physicalDevice_, device_, commandPool_, graphicsComputeQueue_, vertices);
    indexBuffer_ = createIndexBuffer(physicalDevice_, device_, commandPool_, graphicsComputeQueue_, indices);
    descriptorPool_ = createDescriptorPool(device_);

    std::unique_ptr<AllocatedBuffer> stagingParticleBuffer = initializeParticles(physicalDevice_, device_);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        particleBuffers_.push_back(createParticleBuffer(physicalDevice_, device_, stagingParticleBuffer, SingleTimeCommands(device_, commandPool_, graphicsComputeQueue_)));
    }
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frameRenderingInfos_.push_back(std::make_unique<FrameRenderingInfo>(physicalDevice_, device_, commandPool_, descriptorPool_, descriptorSetLayout_, computeDescriptorSetLayout_, textureImage_->imageView, textureSampler_, particleBuffers_[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT], particleBuffers_[i]));
    }
}

void HelloTriangleApplication::loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.texCoord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
            };

            vertex.normal = {
                attrib.normals[3 * index.normal_index + 0],
                attrib.normals[3 * index.normal_index + 1],
                attrib.normals[3 * index.normal_index + 2]
            };

            vertex.color = { 1.0f, 1.0f, 1.0f };

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }
            indices.push_back(uniqueVertices[vertex]);
        }
    }

    std::cout << "NumVertices: " << vertices.size() << std::endl;
}

vk::ImageView createImageView(const vk::Device &device, const vk::Image &image, const vk::Format &format, const vk::ImageAspectFlags &aspectFlags, uint32_t mipLevels) {
    vk::ImageViewCreateInfo viewInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = {
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = aspectFlags,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    
    return device.createImageView(viewInfo);
}



void HelloTriangleApplication::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    device_.waitIdle();

    cleanupSwapChain();

    swapChainSetupInfo_ = SwapChainSetupInfo(physicalDevice_, surface_);
    swapChainExtent_ = chooseSwapExtent(window_, swapChainSetupInfo_.capabilities);
    swapChain_ = createSwapChain(device_, surface_, swapChainSetupInfo_, swapChainExtent_, queueFamilyIndices_);
    swapChainImages_ = device_.getSwapchainImagesKHR(swapChain_);
    swapChainImageViews_ = createImageViews(device_, swapChainImages_, swapChainImageFormat_);
    depthImage_ = createDepthResources(physicalDevice_, device_, depthFormat_, swapChainExtent_, msaaSamples_, commandPool_, graphicsComputeQueue_);
    colorImage_ = createColorResources(physicalDevice_, device_, swapChainImageFormat_, swapChainExtent_, msaaSamples_, commandPool_, graphicsComputeQueue_);
    swapChainFramebuffers_ = createFramebuffers(device_, swapChainImageViews_, colorImage_->imageView, depthImage_->imageView, renderPass_, swapChainExtent_);
}

void HelloTriangleApplication::mainLoop() {
    while (!glfwWindowShouldClose(window_)) {
        // std::cout << "\r" << count;
        glfwPollEvents();

        fpcon.update(window_);

        // Also can be set in resize callback
        framebufferResized_ |= drawFrame(device_, swapChain_, graphicsComputeQueue_, presentQueue_, swapChainExtent_, frameRenderingInfos_[frameNum_ % MAX_FRAMES_IN_FLIGHT],
            renderPass_, swapChainFramebuffers_, graphicsPipeline_, particleGraphicsPipeline_, computePipeline_, vertexBuffer_->buffer, indexBuffer_->buffer, indices, pipelineLayout_, computePipelineLayout_, particleBuffers_[frameNum_ % MAX_FRAMES_IN_FLIGHT]->buffer, scale);
        if (framebufferResized_) {
            recreateSwapChain();
            framebufferResized_ = false;
        }
        ++frameNum_;
    }

    device_.waitIdle();
}

void HelloTriangleApplication::cleanupSwapChain() {
    for (auto &framebuffer : swapChainFramebuffers_) {
        device_.destroyFramebuffer(framebuffer);
    }

    depthImage_.reset();
    colorImage_.reset();
    for (auto &imageView : swapChainImageViews_) {
        device_.destroyImageView(imageView);
    }

    device_.destroySwapchainKHR(swapChain_);
}

HelloTriangleApplication::~HelloTriangleApplication() {
    cleanupSwapChain();

    device_.destroySampler(textureSampler_);

    textureImage_.reset();

    device_.destroyDescriptorPool(descriptorPool_);
    device_.destroyDescriptorSetLayout(descriptorSetLayout_);
    device_.destroyDescriptorSetLayout(computeDescriptorSetLayout_);

    indexBuffer_.reset();
    vertexBuffer_.reset();
    for (std::unique_ptr<AllocatedBuffer>& particleBuffer : particleBuffers_) {
        particleBuffer.reset();
    }

    for (std::unique_ptr<FrameRenderingInfo> &frameRenderingInfo : frameRenderingInfos_) {
        frameRenderingInfo.reset();
    }

    device_.destroyCommandPool(commandPool_);

    device_.destroyPipeline(graphicsPipeline_);
    device_.destroyPipeline(particleGraphicsPipeline_);
    device_.destroyPipelineLayout(pipelineLayout_);
    device_.destroyPipeline(computePipeline_);
    device_.destroyPipelineLayout(computePipelineLayout_);
    device_.destroyRenderPass(renderPass_);
    device_.destroy();

    if (ENABLE_VALIDATION_LAYERS) {
        debugMessenger_.reset();
    }

    instance_.destroySurfaceKHR(surface_);
    instance_.destroy();

    glfwDestroyWindow(window_);

    glfwTerminate();
}