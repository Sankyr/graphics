for /r %%i in (*.vert) do C:/VulkanSDK/1.2.135.0/Bin/glslc %%i -o compiledShaders/%%~ni_vert.spv
for /r %%i in (*.frag) do C:/VulkanSDK/1.2.135.0/Bin/glslc %%i -o compiledShaders/%%~ni_frag.spv
pause