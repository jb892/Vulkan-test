# Define Vulkan LunarG SDK path:
VULKAN_SDK_PATH = /home/jack/vulkan/VulkanSDK/1.0.49.0/x86_64
STB_INCLUDE_PATH = /home/jack/Documents/Vulkan-Test/include
TINYOBJ_INCLUDE_PATH = /home/jack/Documents/Vulkan-Test/include

# Define a CFLAGS compiler flags:
CFLAGS = -std=c++11 -I$(VULKAN_SDK_PATH)/include -I$(STB_INCLUDE_PATH) -I$(TINYOBJ_INCLUDE_PATH)

# Define Linker flags:
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: source/main.cpp
	g++ $(CFLAGS) -o VulkanTest source/main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest


