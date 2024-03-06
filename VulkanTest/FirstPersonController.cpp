#include "FirstPersonController.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_access.hpp>

glm::mat4 FirstPersonController::getViewMatrix() {
	glm::mat4 translate = glm::translate(glm::mat4(1.0f), position_);
	glm::mat4 rotation = glm::rotate(glm::rotate(translate,
			glm::radians(pan_), glm::vec3(0.0f, 1.0f, 0.0f)), 
		glm::radians(tilt_), glm::vec3(1.0f, 0.0f, 0.0f));
	return glm::inverse(rotation);
}

void FirstPersonController::update(GLFWwindow* window) {
	glm::mat4 rotation = glm::rotate(glm::rotate(glm::mat4(1.0f),
		glm::radians(pan_), glm::vec3(0.0f, 1.0f, 0.0f)),
		glm::radians(tilt_), glm::vec3(1.0f, 0.0f, 0.0f));
	glm::vec3 tangentDir = glm::vec3(glm::column(rotation, 0).x, glm::column(rotation, 0).y, glm::column(rotation, 0).z);
	glm::vec3 viewDir = glm::vec3(glm::column(rotation, 2).x, glm::column(rotation, 2).y, glm::column(rotation, 2).z);

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		position_ -= speed_ * tangentDir;
    }
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		position_ += speed_* tangentDir;
	}
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		position_ -= speed_ * viewDir;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		position_ += speed_ * viewDir;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		position_.y -= speed_;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		position_.y += speed_;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		speed_ *= 1.01;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		speed_ /= 1.01;
	}
}

void FirstPersonController::onMouseDrag(GLFWwindow* window, double xpos, double ypos) {
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
		prev_xpos_ = -1;
		prev_ypos_ = -1;
		return;
	}

	if (prev_xpos_ == -1) {
		prev_xpos_ = xpos;
		prev_ypos_ = ypos;
		return;
	}

	pan_ += rotSpeed_ * (xpos - prev_xpos_);
	tilt_ -= static_cast<float>(width) / height * rotSpeed_ * (ypos - prev_ypos_);
	prev_xpos_ = xpos;
	prev_ypos_ = ypos;
}