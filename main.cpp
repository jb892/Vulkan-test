#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <iostream>

//#include <vulkan/vulkan.h> // Vulkan will be autometically loaded by GLFW.

#include <stdexcept>
#include <functional>
#include <cstring>

//template <typename T>
//class VDeleter {
//public:
//    VDeleter() : VDeleter([](T, VkAllocationCallbacks*) {}) {}

//    VDeleter(std::function<void(T, VkAllocationCallbacks*)> deletef) {
//        this->deleter = [=](T obj) { deletef(obj, nullptr); };
//    }

//    VDeleter(const VDeleter<VkInstance>& instance, std::function<void(VkInstance, T, VkAllocationCallbacks*)> deletef) {
//        this->deleter = [&instance, deletef](T obj) { deletef(instance, obj, nullptr); };
//    }

//    VDeleter(const VDeleter<VkDevice>& device, std::function<void(VkDevice, T, VkAllocationCallbacks*)> deletef) {
//        this->deleter = [&device, deletef](T obj) { deletef(device, obj, nullptr); };
//    }

//    ~VDeleter() {
//        cleanup();
//    }

//    const T* operator &() const {
//        return &object;
//    }

//    T* replace() {
//        cleanup();
//        return &object;
//    }

//    operator T() const {
//        return object;
//    }

//    void operator=(T rhs) {
//        if (rhs != object) {
//            cleanup();
//            object = rhs;
//        }
//    }

//    template<typename V>
//    bool operator==(V rhs) {
//        return object == T(rhs);
//    }

//private:
//    T object{VK_NULL_HANDLE};
//    std::function<void(T)> deleter;

//    void cleanup() {
//        if (object != VK_NULL_HANDLE) {
//            deleter(object);
//        }
//        object = VK_NULL_HANDLE;
//    }
//};

const int WIDTH = 800;
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

class HelloTriangleApplication {
public:
    void run() {
            initWindow();
            initVulkan();
            mainLoop();
    }
	
private:
    GLFWwindow* window;

//    VDeleter<VkInstance> instance {vkDestoryInstance};
    VkInstance instance;

    void initWindow() {
        // init GLFW library.
        glfwInit();

        // Tell GLFW not to create OpenGL context by default.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // Disable resizing window operation here.
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // glfwCreateWindow(width, height, title, monitorIndex(optional))
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    }

    void initVulkan() {
        // init Vulkan library by creating an instance.
        createInstance();
    }

    void mainLoop() {
        // Keep the app running until either an error occurs
        //    or the window is closed.
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        // Optional info
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // not optional
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // Non-optional info. (Define Global extensions and validation layers)
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Specify our desired global extensions.
        unsigned int glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        // Determine the global validation layers to enable.
        createInfo.enabledLayerCount = 0;
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance!");
        }
    }

    /**
     * @brief checkValidationLayerSupport
     * Check if all of the requested layers are available.
     * @return
     */
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> avaliableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, avaliableLayers.data());

        return false;
    }
};

int main()
{
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

