#pragma once
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
class FirstPersonController {
public:
	void update(GLFWwindow* window);
	void onMouseDrag(GLFWwindow* window, double xpos, double ypos);
	glm::mat4 getViewMatrix();
private:
	glm::vec3 position_ = glm::vec3(0, 0, 0);
	float pan_ = 0;
	float tilt_ = 0;
	float speed_ = 0.1;
	float rotSpeed_ = .05;
	float prev_xpos_ = -1;
	float prev_ypos_ = -1;
};

