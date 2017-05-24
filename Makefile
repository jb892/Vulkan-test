# Define Vulkan LunarG SDK path:
VULKAN_SDK_PATH = /home/jack/vulkan/VulkanSDK/1.0.49.0/x86_64

# Define a CFLAGS compiler flags:
CFLAGS = -std=c++11 -I$(VULKAN_SDK_PATH)/include

# Define Linker flags:
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest


