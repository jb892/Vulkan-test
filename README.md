# Vulkan-test

## Getting Started

### Prerequisites

* [Vulkan LunarG SDK](https://www.lunarg.com/vulkan-sdk/) installed 

* [GLFW](http://www.glfw.org/) 3.2.1 installed

* [CMake](https://cmake.org/) VERSION >= 3.7.1 installed

* [stb Image Loader](https://github.com/nothings/stb) (Already included in the 'include' directory.)

* [TINYOBJ Model Loader](https://github.com/syoyo/tinyobjloader) (Already included in the 'include' directory.)

* Any C++ compiler which support C++11


## Running

* To run the program:
```
make
./VulkanTest
```

* To make a clean:
```
make clean
```

* To compile the vertex and fragment shader:
```
./compile.sh
```
**N.B.: Don't forget to put your compiled shader file with extension name .spv into shader folder.**

## Result

![](https://github.com/jb892/Vulkan-test/blob/master/Vulkan-first-triangle.png "Rendered Triangle")
![](https://github.com/jb892/Vulkan-test/blob/master/ModelWithTexture.png "Loaded OBJ Model with texture")
