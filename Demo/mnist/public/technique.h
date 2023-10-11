///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../private/technique.h"
#include <string>
#include <vector>

namespace mnist
{
    // Compile time technique settings. Feel free to modify these.
    static const int c_numSRVDescriptors = 256;  // If 0, no heap will be created. One heap shared by all contexts of this technique.
    static const bool c_debugShaders = true; // If true, will compile shaders with debug info enabled.
    static const bool c_debugNames = true; // If true, will set debug names on objects. If false, debug names should be deadstripped from the executable.

    // Information about the technique
    static const bool c_requiresRaytracing = false; // If true, this technique will not work without raytracing support

    enum class LogLevel : int
    {
        Info,
        Warn,
        Error
    };
    using TLogFn = void (*)(int level, const char* msg, ...);
    using TPerfEventBeginFn = void (*)(const char* name, ID3D12CommandList* commandList, int index);
    using TPerfEventEndFn = void (*)(ID3D12CommandList* commandList);

    struct LoadTextureData
    {
        // Information about texture to load
        std::string fileName;
        int numChannels = 4;

        // Loaded texture data
        std::vector<unsigned char> pixelsU8;
        std::vector<float> pixelsF32;
        int width = 1;
        int height = 1;
    };
    using TLoadTextureFn = bool (*)(LoadTextureData& data);

    struct ProfileEntry
    {
        const char* m_label = nullptr;
        float m_gpu = 0.0f;
        float m_cpu = 0.0f;
    };

    struct Context
    {
        // This is the input to the technique that you are expected to fill out
        struct ContextInput
        {

            // Variables
            bool variable_Clear = false;
            float4 variable_MouseState = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 variable_MouseStateLastFrame = {0.0f, 0.0f, 0.0f, 0.0f};
            float3 variable_iResolution = {0.0f, 0.0f, 0.0f};
            float variable_iTime = 0.0f;
            float variable_iTimeDelta = 0.0f;
            float variable_iFrameRate = 0.0f;
            int variable_iFrame = 0;
            float4 variable_iMouse = {0.0f, 0.0f, 0.0f};
            float variable_PenSize = 10.0f;
            bool variable_UseImportedImage = false;
            bool variable_NormalizeDrawing = true;  // MNIST normalization: shrink image to 20x20 and put center of mass in the middle of a 28x28 image

            ID3D12Resource* buffer_NN_Weights = nullptr;
            DXGI_FORMAT buffer_NN_Weights_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
            unsigned int buffer_NN_Weights_stride = 0; // For structured buffers, the size of the structure
            unsigned int buffer_NN_Weights_count = 0; // How many items there are
            D3D12_RESOURCE_STATES buffer_NN_Weights_state = D3D12_RESOURCE_STATE_COMMON;

            static const D3D12_RESOURCE_FLAGS c_buffer_NN_Weights_flags =  D3D12_RESOURCE_FLAG_NONE; // Flags the buffer needs to have been created with

            ID3D12Resource* texture_Presentation_Canvas = nullptr;
            unsigned int texture_Presentation_Canvas_size[3] = { 0, 0, 0 };
            DXGI_FORMAT texture_Presentation_Canvas_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_Presentation_Canvas_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_RESOURCE_STATES texture_Presentation_Canvas_state = D3D12_RESOURCE_STATE_COMMON;

            ID3D12Resource* texture_Imported_Image = nullptr;
            unsigned int texture_Imported_Image_size[3] = { 0, 0, 0 };
            DXGI_FORMAT texture_Imported_Image_format = DXGI_FORMAT_UNKNOWN;
            static const D3D12_RESOURCE_FLAGS texture_Imported_Image_flags =  D3D12_RESOURCE_FLAG_NONE;
            D3D12_RESOURCE_STATES texture_Imported_Image_state = D3D12_RESOURCE_STATE_COMMON;
        };
        ContextInput m_input;

        // This is the output of the technique that you can consume
        struct ContextOutput
        {
        };
        ContextOutput m_output;

        // Internal storage for the technique
        ContextInternal m_internal;

        // If true, will do both cpu and gpu profiling. Call ReadbackProfileData() on the context to get the profiling data.
        bool m_profile = false;
        const ProfileEntry* ReadbackProfileData(ID3D12CommandQueue* commandQueue, int& numItems);

        // Set this static function pointer to your own log function if you want to recieve callbacks on info, warnings and errors.
        static TLogFn LogFn;

        // These callbacks are for perf instrumentation, such as with Pix.
        static TPerfEventBeginFn PerfEventBeginFn;
        static TPerfEventEndFn PerfEventEndFn;

        // This callback is for when the technique needs to load a texture
        static TLoadTextureFn LoadTextureFn;

        // The path to where the shader files for this technique are. Defaults to L"mnist/"
        static std::wstring s_techniqueLocation;

        static int GetContextCount();
        static Context* GetContext(int index);

        // These are released by the technique on shutdown
        ID3D12Resource* CreateManagedBuffer(ID3D12Device* device, unsigned int size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, D3D12_HEAP_TYPE heapType, ID3D12GraphicsCommandList* commandList, void* initialData, LPCWSTR debugName);
        ID3D12Resource* CreateManagedTexture2D(ID3D12Device* device, const unsigned int size[2], DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* commandList, void* initialData, unsigned int initialDataRowPitch, LPCWSTR debugName);

        // Helpers for the host app
        void UploadTextureData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* texture, D3D12_RESOURCE_STATES textureState, void* data, unsigned int dataRowPitch);
        void UploadBufferData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* buffer, D3D12_RESOURCE_STATES bufferState, void* data, unsigned int dataSize);

    private:
        friend void DestroyContext(Context* context);
        ~Context();

        friend void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList);
        void EnsureResourcesCreated(ID3D12Device* device, ID3D12GraphicsCommandList* commandList);

        ProfileEntry m_profileData[6+1]; // One for each action node, and another for the total
    };

    // Create 0 to N contexts at any point
    Context* CreateContext(ID3D12Device* device);

    // Call at the beginning of your frame
    void OnNewFrame(int framesInFlight);

    // Call this 0 to M times a frame on each context to execute the technique
    void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList);

    // Destroy a context
    void DestroyContext(Context* context);
};
