#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

//push constants block
layout( push_constant ) uniform constants {
	float scale;
} PushConstants;

void main() {
	gl_PointSize = 1;
    gl_Position = vec4(PushConstants.scale * inPosition.xy, 0, 1.0); // ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}