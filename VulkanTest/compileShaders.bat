for /r D:\Projects\C++\VulkanTest\VulkanTest\shaders\ %%i in (*.vert) do C:/VulkanSDK/1.2.135.0/Bin/glslc %%i -o D:\Projects\C++\VulkanTest\VulkanTest\compiledShaders\%%~ni_vert.spv
for /r D:\Projects\C++\VulkanTest\VulkanTest\shaders\ %%i in (*.frag) do C:/VulkanSDK/1.2.135.0/Bin/glslc %%i -o D:\Projects\C++\VulkanTest\VulkanTest\compiledShaders\%%~ni_frag.spv
pause