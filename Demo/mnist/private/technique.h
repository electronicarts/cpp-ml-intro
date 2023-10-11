///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <d3d12.h>
#include <array>
#include <vector>

namespace mnist
{
    using uint = unsigned int;
    using uint2 = std::array<uint, 2>;
    using uint3 = std::array<uint, 3>;
    using uint4 = std::array<uint, 4>;

    using int2 = std::array<int, 2>;
    using int3 = std::array<int, 3>;
    using int4 = std::array<int, 4>;
    using float2 = std::array<float, 2>;
    using float3 = std::array<float, 3>;
    using float4 = std::array<float, 4>;
    using float4x4 = std::array<std::array<float, 4>, 4>;

    struct ContextInternal
    {
        ID3D12QueryHeap* m_TimestampQueryHeap = nullptr;
        ID3D12Resource* m_TimestampReadbackBuffer = nullptr;

        static ID3D12CommandSignature* s_commandSignatureDispatch;

        struct Struct_DrawExtents
        {
            uint MinX = 0;
            uint MaxX = 0;
            uint MinY = 0;
            uint MaxY = 0;
            uint PixelCount = 0;
            uint2 PixelLocationSum = {0, 0};
        };

        struct Struct__DrawCB
        {
            unsigned int Clear = false;
            float PenSize = 10.0f;
            float2 _padding0 = {};  // Padding
            float4 MouseState = {0.0f, 0.0f, 0.0f, 0.0f};
            int iFrame = 0;
            unsigned int UseImportedImage = false;
            float2 _padding1 = {};  // Padding
            float4 MouseStateLastFrame = {0.0f, 0.0f, 0.0f, 0.0f};
        };

        struct Struct__ShrinkCB
        {
            unsigned int UseImportedImage = false;
            unsigned int NormalizeDrawing = true;  // MNIST normalization: shrink image to 20x20 and put center of mass in the middle of a 28x28 image
        };

        struct Struct__PresentationCB
        {
            float PenSize = 10.0f;
            float3 _padding0 = {};  // Padding
            float4 MouseState = {0.0f, 0.0f, 0.0f, 0.0f};
            unsigned int UseImportedImage = false;
        };

        // Variables
        static const int variable_c_numInputNeurons;
        static const int variable_c_numHiddenNeurons;
        static const int variable_c_numOutputNeurons;
        const int variable_c_numHiddenWeights = 23550;  // (c_numInputNeurons + 1) * c_numHiddenNeurons
        static const int variable_c_numOutputWeights;  // (c_numHiddenNeurons + 1) * c_numOutputNeurons
        static const uint2 variable_c_NNInputImageSize;
        static const uint2 variable_c_drawingCanvasSize;

        ID3D12Resource* texture_Drawing_Canvas = nullptr;
        unsigned int texture_Drawing_Canvas_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture_Drawing_Canvas_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_Drawing_Canvas_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_Drawing_Canvas_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture_NN_Input = nullptr;
        unsigned int texture_NN_Input_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture_NN_Input_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture_NN_Input_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_RESOURCE_STATES c_texture_NN_Input_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* buffer_Hidden_Layer_Activations = nullptr;
        DXGI_FORMAT buffer_Hidden_Layer_Activations_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
        unsigned int buffer_Hidden_Layer_Activations_stride = 0; // For structured buffers, the size of the structure
        unsigned int buffer_Hidden_Layer_Activations_count = 0; // How many items there are
        const D3D12_RESOURCE_STATES c_buffer_Hidden_Layer_Activations_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        static const D3D12_RESOURCE_FLAGS c_buffer_Hidden_Layer_Activations_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // Flags the buffer needs to have been created with

        ID3D12Resource* buffer_Output_Layer_Activations = nullptr;
        DXGI_FORMAT buffer_Output_Layer_Activations_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
        unsigned int buffer_Output_Layer_Activations_stride = 0; // For structured buffers, the size of the structure
        unsigned int buffer_Output_Layer_Activations_count = 0; // How many items there are
        const D3D12_RESOURCE_STATES c_buffer_Output_Layer_Activations_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        static const D3D12_RESOURCE_FLAGS c_buffer_Output_Layer_Activations_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // Flags the buffer needs to have been created with

        ID3D12Resource* buffer_Draw_Extents = nullptr;
        DXGI_FORMAT buffer_Draw_Extents_format = DXGI_FORMAT_UNKNOWN; // For typed buffers, the type of the buffer
        unsigned int buffer_Draw_Extents_stride = 0; // For structured buffers, the size of the structure
        unsigned int buffer_Draw_Extents_count = 0; // How many items there are
        const D3D12_RESOURCE_STATES c_buffer_Draw_Extents_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        static const D3D12_RESOURCE_FLAGS c_buffer_Draw_Extents_flags =  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; // Flags the buffer needs to have been created with

        Struct__DrawCB constantBuffer__DrawCB_cpu;
        ID3D12Resource* constantBuffer__DrawCB = nullptr;

        static ID3D12PipelineState* computeShader_Draw_pso;
        static ID3D12RootSignature* computeShader_Draw_rootSig;

        static ID3D12PipelineState* computeShader_CalculateExtents_pso;
        static ID3D12RootSignature* computeShader_CalculateExtents_rootSig;

        Struct__ShrinkCB constantBuffer__ShrinkCB_cpu;
        ID3D12Resource* constantBuffer__ShrinkCB = nullptr;

        static ID3D12PipelineState* computeShader_Shrink_pso;
        static ID3D12RootSignature* computeShader_Shrink_rootSig;

        static ID3D12PipelineState* computeShader_Hidden_Layer_pso;
        static ID3D12RootSignature* computeShader_Hidden_Layer_rootSig;

        static ID3D12PipelineState* computeShader_Output_Layer_pso;
        static ID3D12RootSignature* computeShader_Output_Layer_rootSig;

        ID3D12Resource* texture__loadedTexture_0 = nullptr;
        unsigned int texture__loadedTexture_0_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_0_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_0_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_0_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_1 = nullptr;
        unsigned int texture__loadedTexture_1_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_1_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_1_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_1_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_2 = nullptr;
        unsigned int texture__loadedTexture_2_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_2_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_2_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_2_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_3 = nullptr;
        unsigned int texture__loadedTexture_3_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_3_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_3_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_3_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_4 = nullptr;
        unsigned int texture__loadedTexture_4_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_4_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_4_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_4_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_5 = nullptr;
        unsigned int texture__loadedTexture_5_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_5_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_5_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_5_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_6 = nullptr;
        unsigned int texture__loadedTexture_6_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_6_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_6_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_6_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_7 = nullptr;
        unsigned int texture__loadedTexture_7_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_7_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_7_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_7_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_8 = nullptr;
        unsigned int texture__loadedTexture_8_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_8_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_8_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_8_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_9 = nullptr;
        unsigned int texture__loadedTexture_9_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_9_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_9_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_9_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        ID3D12Resource* texture__loadedTexture_10 = nullptr;
        unsigned int texture__loadedTexture_10_size[3] = { 0, 0, 0 };
        DXGI_FORMAT texture__loadedTexture_10_format = DXGI_FORMAT_UNKNOWN;
        static const D3D12_RESOURCE_FLAGS texture__loadedTexture_10_flags =  D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES c_texture__loadedTexture_10_endingState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        Struct__PresentationCB constantBuffer__PresentationCB_cpu;
        ID3D12Resource* constantBuffer__PresentationCB = nullptr;

        static ID3D12PipelineState* computeShader_Presentation_pso;
        static ID3D12RootSignature* computeShader_Presentation_rootSig;

        // Created for the host when asked, freed on shutdown.
        std::vector<ID3D12Resource*> m_managedResources;
    };
};
