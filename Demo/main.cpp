#define _CRT_SECURE_NO_WARNINGS

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxgidebug.h>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <array>
#include <comdef.h>
#include <string>
#include <chrono>
#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx12.h"
#include <cmath>

#include "mnist/public/technique.h"
#include "mnist/public/imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

// Note: this being true can cause crashes in nsight (nsight says so on startup)
#define BREAK_ON_DX12_ERROR() _DEBUG

static unsigned int c_width = 1280;
static unsigned int c_height = 1000;
static const wchar_t* c_windowTitle = L"MNIST Neural Network Demo";
static const bool g_useWarpDevice = false;
static const UINT FrameCount = 2;
static const bool c_enableGPUBasedValidation = false;

static const UINT c_rtvDescriptors = FrameCount + 1; // one for each frame, plus one for the color target
static const UINT c_srvDescriptors = 1; // one for imgui
static const UINT c_imguiDescriptorIndex = c_srvDescriptors - 1;

#define Assert(X, MSG, ...) if ((X) == false) ShowErrorMessage( __FUNCTION__ "():\n\nExpression:\n" #X "\n\n" MSG, __VA_ARGS__);
#define ThrowIfFailed(hr) Assert(!FAILED(hr), #hr)

bool g_userWantsExit = false;

static HINSTANCE s_hInstance;
static int s_nCmdShow;

std::vector<char> LoadBinaryFileIntoMemory(const char* fileName)
{
    std::vector<char> ret;

    FILE* file = nullptr;
    fopen_s(&file, fileName, "rb");
    if (!file)
        return ret;

    fseek(file, 0, SEEK_END);
    ret.resize(ftell(file));
    fseek(file, 0, SEEK_SET);
    fread(ret.data(), 1, ret.size(), file);

    fclose(file);
    return ret;
}

void ShowErrorMessage(const char* msg, ...)
{
    static std::vector<char> buffer(40960);
    va_list args;
    va_start(args, msg);
    vsprintf_s(buffer.data(), buffer.size(), msg, args);
    va_end(args);

    printf(buffer.data());
    OutputDebugStringA(buffer.data());
    OutputDebugStringA("\n");

    MessageBoxA(nullptr, buffer.data(), "Gigi DX12 Host App", MB_OK);

    DebugBreak();
    exit(100);
}

ID3D12Resource* CreateTexture(ID3D12Device* device, const unsigned int size[3], DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_DIMENSION textureType, LPCWSTR debugName, D3D12_CLEAR_VALUE* clearValue = nullptr, D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT)
{
    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.MipLevels = 1;
    textureDesc.Format = format;
    textureDesc.Width = size[0];
    textureDesc.Height = size[1];
    textureDesc.DepthOrArraySize = size[2];
    textureDesc.Flags = flags;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Dimension = textureType;

    D3D12_HEAP_PROPERTIES heapProperties;
    heapProperties.Type = heapType;
    heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProperties.CreationNodeMask = 1;
    heapProperties.VisibleNodeMask = 1;

    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        state,
        clearValue,
        IID_PPV_ARGS(&resource));

    if (FAILED(hr))
        return nullptr;

    if (debugName)
        resource->SetName(debugName);

    return resource;
}

struct DX12Data
{
    bool m_inited = false;
    bool m_imguiInitialized = false;

    float m_frameTime = 0.0f;
    float m_elapsedTime = 0.0f;
    std::chrono::high_resolution_clock::time_point m_lastFrameStart;
    UINT m_frameCount = 0;  // an integer that increments every frame. Used by shaders to do animation etc.

    bool m_mouseButtons[2] = { false, false };

    HWND m_hwnd = nullptr;

    ID3D12Device* m_device = nullptr;

    IDXGISwapChain3* m_swapChain = nullptr;

    ID3D12DescriptorHeap* m_rtvHeap = nullptr;
    UINT m_rtvDescriptorSize = 0;

    ID3D12DescriptorHeap* m_srvHeap = nullptr;
    UINT m_srvDescriptorSize = 0;

    ID3D12Resource* m_backBuffers[FrameCount];
    ID3D12Resource* m_colorTarget = nullptr;

    ID3D12CommandAllocator* m_commandAllocators[FrameCount];
    ID3D12CommandQueue* m_commandQueue = nullptr;
    ID3D12GraphicsCommandList* m_commandList = nullptr;

    UINT m_frameIndex = 0;  // The index of the frame, for dx12 frame resource / back buffer uses.
    HANDLE m_fenceEvent;
    ID3D12Fence* m_fence = nullptr;
    UINT64 m_fenceValues[FrameCount];

    mnist::Context* m_mnist = nullptr;
    char m_mnistFileName[1024] = "mnist/assets/0.png";
    bool m_mnistFileNameChanged = true;

    // Wait for pending GPU work to complete.
    void WaitForGpu()
    {
        // Schedule a Signal command in the queue.
        ThrowIfFailed(m_commandQueue->Signal(m_fence, m_fenceValues[m_frameIndex]));

        // Wait until the fence has been processed.
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
        WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

        // Increment the fence value for the current frame.
        m_fenceValues[m_frameIndex]++;
    }

    // Prepare to render the next frame.
    void MoveToNextFrame()
    {
        // Schedule a Signal command in the queue.
        const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
        ThrowIfFailed(m_commandQueue->Signal(m_fence, currentFenceValue));

        // Update the frame index.
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

        // If the next frame is not ready to be rendered yet, wait until it is ready.
        if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
        {
            ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
            WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
        }

        // Set the fence value for the next frame.
        m_fenceValues[m_frameIndex] = currentFenceValue + 1;
    }

    void PopulateCommandList()
    {
        // Command list allocators can only be reset when the associated 
        // command lists have finished execution on the GPU; apps should use 
        // fences to determine GPU execution progress.
        ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

        // However, when ExecuteCommandList() is called on a particular command 
        // list, that command list can then be reset at any time and must be before 
        // re-recording.
        ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex], nullptr));

        ID3D12DescriptorHeap* ppHeaps[] = { m_srvHeap };
        m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

        // clear viewport and scissor rect
        D3D12_VIEWPORT viewport = { 0.0f, 0.0f, float(c_width), float(c_height), 0.0f, 1.0f };
        D3D12_RECT scissorRect = { 0, 0, (LONG)c_width, (LONG)c_height };
        m_commandList->RSSetViewports(1, &viewport);
        m_commandList->RSSetScissorRects(1, &scissorRect);

        // make color target ready for rasterization
        {
            D3D12_RESOURCE_BARRIER barrier;
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = m_colorTarget;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            m_commandList->ResourceBarrier(1, &barrier);
        }

        // set color target
        D3D12_CPU_DESCRIPTOR_HANDLE colorTargetHandle = { m_rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + FrameCount * m_rtvDescriptorSize };
        m_commandList->OMSetRenderTargets(1, &colorTargetHandle, TRUE, nullptr);

        if (m_mnist)
        {
            // If the weights buffer isn't yet created, create it and fill it in with the weight data.
            if (!m_mnist->m_input.buffer_NN_Weights)
            {
                std::vector<char> weights = LoadBinaryFileIntoMemory("mnist/assets/Backprop_Weights.bin");
                m_mnist->m_input.buffer_NN_Weights = m_mnist->CreateManagedBuffer(m_device, (unsigned int)weights.size(), m_mnist->m_input.c_buffer_NN_Weights_flags, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, m_commandList, weights.data(), L"MNIST NNWeights");
                m_mnist->m_input.buffer_NN_Weights_format = DXGI_FORMAT_R32_FLOAT;
                m_mnist->m_input.buffer_NN_Weights_stride = 0;
                m_mnist->m_input.buffer_NN_Weights_count = (unsigned int)(weights.size() / sizeof(float));
                m_mnist->m_input.buffer_NN_Weights_state = D3D12_RESOURCE_STATE_COMMON;
            }

            // If the imported image isn't yet created, create it
            if (!m_mnist->m_input.texture_Imported_Image)
            {
                static const unsigned int c_size[2] = { 28, 28 };
                m_mnist->m_input.texture_Imported_Image = m_mnist->CreateManagedTexture2D(m_device, c_size, DXGI_FORMAT_R8_UNORM, m_mnist->m_input.texture_Imported_Image_flags, D3D12_RESOURCE_STATE_COMMON, m_commandList, nullptr, 0, L"MNIST Imported Image");
                m_mnist->m_input.texture_Imported_Image_size[0] = 28;
                m_mnist->m_input.texture_Imported_Image_size[1] = 28;
                m_mnist->m_input.texture_Imported_Image_size[2] = 1;
                m_mnist->m_input.texture_Imported_Image_format = DXGI_FORMAT_R8_UNORM;
                m_mnist->m_input.texture_Imported_Image_state = D3D12_RESOURCE_STATE_COMMON;
            }

            // If the mnist file name changed, try to load it and copy it into texture_Imported_Image
            if (m_mnistFileNameChanged)
            {
                int w, h, c;
                unsigned char* pixels = stbi_load(m_mnistFileName, &w, &h, &c, 1);
                if (pixels)
                {
                    if (w == 28 && h == 28)
                    {
                        unsigned int size[2] = { 28, 28 };
                        m_mnist->UploadTextureData(m_device, m_commandList, m_mnist->m_input.texture_Imported_Image, D3D12_RESOURCE_STATE_COMMON, pixels, 28);
                    }
                    stbi_image_free(pixels);
                }
                m_mnistFileNameChanged = false;
            }

            // mouse state
            {
                float mousePos[2] = { 0.0f, 0.0f };
                POINT p;
                if (GetCursorPos(&p))
                {
                    if (ScreenToClient(m_hwnd, &p))
                    {
                        mousePos[0] = (float)p.x;
                        mousePos[1] = (float)p.y;
                    }
                }

                memcpy(&m_mnist->m_input.variable_MouseStateLastFrame, &m_mnist->m_input.variable_MouseState, sizeof(m_mnist->m_input.variable_MouseState));
                m_mnist->m_input.variable_MouseState[0] = mousePos[0];
                m_mnist->m_input.variable_MouseState[1] = mousePos[1];
                m_mnist->m_input.variable_MouseState[2] = m_mouseButtons[0] ? 1.0f : 0.0f;
                m_mnist->m_input.variable_MouseState[3] = m_mouseButtons[1] ? 1.0f : 0.0f;
            }

            m_mnist->m_input.variable_iFrame = m_frameCount;

            m_mnist->m_input.texture_Presentation_Canvas = m_colorTarget;
            m_mnist->m_input.texture_Presentation_Canvas_size[0] = c_width;
            m_mnist->m_input.texture_Presentation_Canvas_size[1] = c_height;
            m_mnist->m_input.texture_Presentation_Canvas_size[2] = 1;
            m_mnist->m_input.texture_Presentation_Canvas_format = DXGI_FORMAT_R8G8B8A8_UNORM;
            m_mnist->m_input.texture_Presentation_Canvas_state = D3D12_RESOURCE_STATE_RENDER_TARGET;

            mnist::Execute(m_mnist, m_device, m_commandList);
        }

        // restore the SRV descriptor heap, because the techniques set and use their own
        m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

        // render imgui
        {
            ImGui::Begin("MNIST");

            if (ImGui::InputText("Import File Name (28x28)", m_mnistFileName, _countof(m_mnistFileName)))
                m_mnistFileNameChanged = true;

            mnist::MakeUI(m_mnist, m_commandQueue);

            ImGui::End();

            ImGui::Render();
            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), m_commandList);
        }

        // Transition:
        // 1) the color target to copy source
        // 2) the back buffer to copy dest 
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = m_colorTarget;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].Transition.pResource = m_backBuffers[m_frameIndex];
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            m_commandList->ResourceBarrier(2, barriers);
        }

        // copy the resource
        m_commandList->CopyResource(m_backBuffers[m_frameIndex], m_colorTarget);

        // Transition the back buffer to present
        {
            D3D12_RESOURCE_BARRIER barriers[1];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = m_backBuffers[m_frameIndex];
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            m_commandList->ResourceBarrier(1, barriers);
        }

        ThrowIfFailed(m_commandList->Close());
    }

    void Update()
    {
        // handle advancement of time
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        m_frameTime = std::chrono::duration_cast<std::chrono::duration<float>>(now - m_lastFrameStart).count();
        m_elapsedTime += m_frameTime;
        m_lastFrameStart = now;
        m_frameCount = m_frameCount + 1;
    }

    void OnRender()
    {
        Update();

        mnist::OnNewFrame(FrameCount);

        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Record all the commands we need to render the scene into the command list.
        PopulateCommandList();

        // Execute the command list.
        ID3D12CommandList* ppCommandLists[] = { m_commandList };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

        // Present the frame.
        ThrowIfFailed(m_swapChain->Present(1, 0));

        MoveToNextFrame();
    }

    void GetHardwareAdapter(
        IDXGIFactory1* pFactory,
        IDXGIAdapter1** ppAdapter,
        bool requestHighPerformanceAdapter = false)
    {
        *ppAdapter = nullptr;

        IDXGIAdapter1* adapter = nullptr;

        IDXGIFactory6* factory6 = nullptr;
        if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6))))
        {
            for (
                UINT adapterIndex = 0;
                SUCCEEDED(factory6->EnumAdapterByGpuPreference(
                    adapterIndex,
                    requestHighPerformanceAdapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
                    IID_PPV_ARGS(&adapter)));
                ++adapterIndex)
            {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);

                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;

                // Check to see whether the adapter supports Direct3D 12, but don't create the
                // actual device yet.
                if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
                    break;
            }
        }

        if (adapter == nullptr)
        {
            for (UINT adapterIndex = 0; SUCCEEDED(pFactory->EnumAdapters1(adapterIndex, &adapter)); ++adapterIndex)
            {
                DXGI_ADAPTER_DESC1 desc;
                adapter->GetDesc1(&desc);

                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;

                // Check to see whether the adapter supports Direct3D 12, but don't create the
                // actual device yet.
                if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr)))
                    break;
            }
        }

        *ppAdapter = adapter;
        factory6->Release();
    }

    void LoadPipeline()
    {
        UINT dxgiFactoryFlags = 0;

    #if defined(_DEBUG)
        // Enable the debug layer (requires the Graphics Tools "optional feature").
        // NOTE: Enabling the debug layer after device creation will invalidate the active device.
        {
            ID3D12Debug* debugController = nullptr;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
            {
                debugController->EnableDebugLayer();

                // Enable additional debug layers.
                dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }
            debugController->Release();
        }

    #endif

        IDXGIFactory4* factory = nullptr;
        ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

        if (g_useWarpDevice)
        {
            IDXGIAdapter* warpAdapter = nullptr;
            ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

            ThrowIfFailed(D3D12CreateDevice(
                warpAdapter,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_device)
            ));

            warpAdapter->Release();
        }
        else
        {
            IDXGIAdapter1* hardwareAdapter = nullptr;
            GetHardwareAdapter(factory, &hardwareAdapter);

            ThrowIfFailed(D3D12CreateDevice(
                hardwareAdapter,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&m_device)
            ));

            hardwareAdapter->Release();
        }

        // set it up to break on dx errors in debug
        #if BREAK_ON_DX12_ERROR()
        {
            ID3D12InfoQueue* infoQueue = nullptr;
            if (SUCCEEDED(m_device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
            {
                    infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
                    infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
                D3D12_MESSAGE_ID hide[] =
                {
                    D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,
                    D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE
                };
                D3D12_INFO_QUEUE_FILTER filter = {};
                filter.DenyList.NumIDs = _countof(hide);
                filter.DenyList.pIDList = hide;
                infoQueue->AddStorageFilterEntries(&filter);
                infoQueue->Release();
            }
        }
        #endif

        // Describe and create the command queue.
        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

        ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

        // Describe and create the swap chain.
        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
        swapChainDesc.BufferCount = FrameCount;
        swapChainDesc.Width = c_width;
        swapChainDesc.Height = c_height;
        swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;

        IDXGISwapChain1* swapChain = nullptr;
        ThrowIfFailed(factory->CreateSwapChainForHwnd(
            m_commandQueue,        // Swap chain needs the queue so that it can force a flush on it.
            m_hwnd,
            &swapChainDesc,
            nullptr,
            nullptr,
            &swapChain
        ));

        // no fullscreen transitions.
        ThrowIfFailed(factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER));

        ThrowIfFailed(swapChain->QueryInterface(IID_PPV_ARGS(&m_swapChain)));
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

        // Create descriptor heaps.
        {
            // RTV - Render Target View
            {
                D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
                heapDesc.NumDescriptors = c_rtvDescriptors;
                heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
                heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_rtvHeap)));

                m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            }
            // SRV - Shader Resource View
            {
                D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
                heapDesc.NumDescriptors = c_srvDescriptors;
                heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
                ThrowIfFailed(m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_srvHeap)));

                m_srvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
        }

        // Create resources.
        {
            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());
            D3D12_CPU_DESCRIPTOR_HANDLE srvHandle(m_srvHeap->GetCPUDescriptorHandleForHeapStart());

            // Create a RTV and a command allocator for each frame.
            for (UINT n = 0; n < FrameCount; n++)
            {
                ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_backBuffers[n])));
                m_device->CreateRenderTargetView(m_backBuffers[n], nullptr, rtvHandle);
                rtvHandle.ptr += m_rtvDescriptorSize;

                ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocators[n])));
            }

            // create color target
            {
                const unsigned int dims[3] = { c_width, c_height, 1 };
                m_colorTarget = CreateTexture(m_device, dims, swapChainDesc.Format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D, L"Color Target", nullptr);
                Assert(m_colorTarget != nullptr , "Could not create render target");

                // Create an RTV
                m_device->CreateRenderTargetView(m_colorTarget, nullptr, rtvHandle);
                rtvHandle.ptr += m_rtvDescriptorSize;
            }
        }

        swapChain->Release();
        factory->Release();
    }

    void LoadAssets()
    {
        // Create the command list.
        ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocators[m_frameIndex], nullptr, IID_PPV_ARGS(&m_commandList)));

        // Close the command list
        ThrowIfFailed(m_commandList->Close());

        // Create synchronization objects
        {
            ThrowIfFailed(m_device->CreateFence(m_fenceValues[m_frameIndex], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
            m_fenceValues[m_frameIndex]++;

            // Create an event handle to use for frame synchronization.
            m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (m_fenceEvent == nullptr)
            {
                ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
            }
        }
    }

    template <typename T>
    static std::string GetAssetPath();

    template <>
    static std::string GetAssetPath<mnist::LoadTextureData>()
    {
        return "mnist/assets/";
    }

    template <typename T>
    static bool GigiLoadTexture(T& data)
    {
        std::string fullFileName = GetAssetPath<T>() + data.fileName;

        std::string extension;
        size_t extensionStart = fullFileName.find_last_of(".");
        if (extensionStart != std::string::npos)
            extension = fullFileName.substr(extensionStart);
        if (extension == ".hdr")
        {
            int c;
            float* pixels = stbi_loadf(fullFileName.c_str(), &data.width, &data.height, &c, data.numChannels);
            if (!pixels)
                return false;

            data.pixelsF32.resize(data.width * data.height * data.numChannels);
            memcpy(data.pixelsF32.data(), pixels, data.pixelsF32.size() * sizeof(float));
            stbi_image_free(pixels);
        }
        else
        {
            int c;
            unsigned char* pixels = stbi_load(fullFileName.c_str(), &data.width, &data.height, &c, data.numChannels);
            if (!pixels)
                return false;

            data.pixelsU8.resize(data.width * data.height * data.numChannels);
            memcpy(data.pixelsU8.data(), pixels, data.pixelsU8.size());
            stbi_image_free(pixels);
        }

        return true;
    }

    static void GigiLogFn(int level, const char* msg, ...)
    {
        static std::vector<char> buffer(40960);
        va_list args;
        va_start(args, msg);
        vsprintf_s(buffer.data(), buffer.size(), msg, args);
        va_end(args);
        if (level >= 2)
            Assert(false, "Gigi: %s", buffer.data());
    }

    void OnInit(HWND hWnd)
    {
        m_inited = true;

        if (c_enableGPUBasedValidation)
        {
            ID3D12Debug* spDebugController0 = nullptr;
            ID3D12Debug1* spDebugController1 = nullptr;
            D3D12GetDebugInterface(IID_PPV_ARGS(&spDebugController0));
            spDebugController0->QueryInterface(IID_PPV_ARGS(&spDebugController1));
            spDebugController1->SetEnableGPUBasedValidation(true);
            spDebugController0->Release();
            spDebugController1->Release();
        }

        m_hwnd = hWnd;
        LoadPipeline();
        LoadAssets();

        // Set the logging function, perf marker functions, and shader locations, and create the technique contexts
        mnist::Context::LogFn = &GigiLogFn;
        mnist::Context::LoadTextureFn = &GigiLoadTexture<mnist::LoadTextureData>;
        mnist::Context::s_techniqueLocation = L"mnist/";
        m_mnist = mnist::CreateContext(m_device);
        Assert(m_mnist != nullptr, "Could not create mnist context");

        m_lastFrameStart = std::chrono::high_resolution_clock::now();

        // clear out the input structures
        memset(m_mouseButtons, 0, sizeof(m_mouseButtons));
    }

    void OnDestroy()
    {
        if (!m_inited)
            return;

        // Ensure that the GPU is no longer referencing resources that are about to be
        // cleaned up.
        WaitForGpu();

        ImGui_ImplDX12_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();

        // Destroy the gigi technique contexts
        if (m_mnist)
        {
            mnist::DestroyContext(m_mnist);
            m_mnist = nullptr;
        }

        CloseHandle(m_fenceEvent);
        m_fence->Release();

        for (int i = 0; i < FrameCount; ++i)
        {
            m_backBuffers[i]->Release();
            m_commandAllocators[i]->Release();
        }
        m_colorTarget->Release();
        m_commandList->Release();
        m_commandQueue->Release();

        m_swapChain->Release();
        m_srvHeap->Release();
        m_rtvHeap->Release();
        m_device->Release();

    #if defined(_DEBUG)
        {
            IDXGIDebug1* dxgiDebug = nullptr;
            if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug))))
            {
                dxgiDebug->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
            }
            dxgiDebug->Release();
        }
    #endif
    }
};

DX12Data g_dx12;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
        return true;

    switch (message)
    {
        case WM_DESTROY: g_userWantsExit = true; PostQuitMessage(0); return 0;
        case WM_KILLFOCUS:
        {
            g_dx12.m_mouseButtons[0] = false;
            g_dx12.m_mouseButtons[1] = false;
            break;
        }
        case WM_INPUT:
        {
            if (g_dx12.m_imguiInitialized && ImGui::GetIO().WantCaptureMouse)
                break;

            if (GetActiveWindow() != g_dx12.m_hwnd)
                break;

            RAWINPUT raw;
            UINT rawSize = sizeof(raw);
            UINT resultData = GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, &raw, &rawSize, sizeof(RAWINPUTHEADER));
            if (resultData == UINT(-1))
                break;

            if (raw.header.dwType != RIM_TYPEMOUSE)
                break;

            if (raw.data.mouse.usButtonFlags)
            {
                if (raw.data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
                    g_dx12.m_mouseButtons[0] = true;
                else if (raw.data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
                    g_dx12.m_mouseButtons[0] = false;
                if (raw.data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
                    g_dx12.m_mouseButtons[1] = true;
                else if (raw.data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
                    g_dx12.m_mouseButtons[1] = false;
            }

            break;
        }
    }
    return DefWindowProc(hWnd, message, wParam, lParam);
}

void InitializeGraphics()
{
    // Initialize the window class.
    WNDCLASSEX windowClass = { 0 };
    windowClass.cbSize = sizeof(WNDCLASSEX);
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    windowClass.lpfnWndProc = WindowProc;
    windowClass.hInstance = s_hInstance;
    windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    windowClass.lpszClassName = L"MNISTNN";
    RegisterClassEx(&windowClass);

    RECT windowRect = { 0, 0, (LONG)c_width, (LONG)c_height };
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    // Create the window and store a handle to it.
    HWND hWnd = CreateWindow(
        windowClass.lpszClassName,
        c_windowTitle,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        windowRect.right - windowRect.left,
        windowRect.bottom - windowRect.top,
        nullptr,        // We have no parent window.
        nullptr,        // We aren't using menus.
        s_hInstance,
        nullptr
    );

    // init mouse
    RAWINPUTDEVICE Rid;
    Rid.usUsagePage = 0x1 /* HID_USAGE_PAGE_GENERIC */;
    Rid.usUsage = 0x2 /* HID_USAGE_GENERIC_MOUSE */;
    Rid.dwFlags = RIDEV_INPUTSINK;
    Rid.hwndTarget = hWnd;
    if (!RegisterRawInputDevices(&Rid, 1, sizeof(RAWINPUTDEVICE)))
    {
        Assert(false, "Could not init mouse");
    }

    // init directx
    g_dx12.OnInit(hWnd);

    ShowWindow(hWnd, s_nCmdShow);

    // Setup Dear ImGui context
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer backends
        D3D12_CPU_DESCRIPTOR_HANDLE cpuSrvHandle;
        cpuSrvHandle.ptr = g_dx12.m_srvHeap->GetCPUDescriptorHandleForHeapStart().ptr + c_imguiDescriptorIndex * g_dx12.m_srvDescriptorSize;
        D3D12_GPU_DESCRIPTOR_HANDLE gpuSrvHandle;
        gpuSrvHandle.ptr = g_dx12.m_srvHeap->GetGPUDescriptorHandleForHeapStart().ptr + c_imguiDescriptorIndex * g_dx12.m_srvDescriptorSize;
        ImGui_ImplWin32_Init(hWnd);
        ImGui_ImplDX12_Init(g_dx12.m_device, FrameCount,
            DXGI_FORMAT_R8G8B8A8_UNORM, g_dx12.m_srvHeap,
            cpuSrvHandle,
            gpuSrvHandle);

        g_dx12.m_imguiInitialized = true;
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow)
{
    s_hInstance = hInstance;
    s_nCmdShow = nCmdShow;

    // initialize rendering
    InitializeGraphics();

    // Main loop.
    MSG msg = {};
    while (msg.message != WM_QUIT && !g_userWantsExit)
    {
        // Process any messages in the queue.
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        g_dx12.OnRender();
    }

    // destroy gfx
    g_dx12.OnDestroy();

    // Return this part of the WM_QUIT message to Windows.
    return static_cast<char>(msg.wParam);
}
