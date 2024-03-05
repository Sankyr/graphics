for /r D:\GitKraken\graphics\VulkanTest\shaders\ %%i in (*.vert) do C:/VulkanSDK/1.3.268.0/Bin/glslc %%i -o D:\GitKraken\graphics\VulkanTest\compiledShaders\%%~ni_vert.spv
for /r D:\GitKraken\graphics\VulkanTest\shaders\ %%i in (*.frag) do C:/VulkanSDK/1.3.268.0/Bin/glslc %%i -o D:\GitKraken\graphics\VulkanTest\compiledShaders\%%~ni_frag.spv
for /r D:\GitKraken\graphics\VulkanTest\shaders\ %%i in (*.comp) do C:/VulkanSDK/1.3.268.0/Bin/glslc %%i -o D:\GitKraken\graphics\VulkanTest\compiledShaders\%%~ni_comp.spv
pause