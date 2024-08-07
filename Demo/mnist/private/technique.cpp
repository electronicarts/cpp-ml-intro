///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "../public/technique.h"
#include "DX12Utils/dxutils.h"
#include "DX12Utils/DelayedReleaseTracker.h"
#include "DX12Utils/HeapAllocationTracker.h"
#include "DX12Utils/TextureCache.h"

#include <vector>
#include <chrono>

namespace mnist
{
    static std::vector<Context*> s_allContexts;

    static DX12Utils::Heap                  s_srvHeap;
    static DX12Utils::Heap                  s_rtvHeap;
    static DX12Utils::Heap                  s_dsvHeap;
    static DX12Utils::UploadBufferTracker   s_ubTracker;
    static DX12Utils::DelayedReleaseTracker s_delayedRelease;
    static DX12Utils::HeapAllocationTracker s_heapAllocationTrackerRTV;
    static DX12Utils::HeapAllocationTracker s_heapAllocationTrackerDSV;

    TLogFn Context::LogFn = [] (LogLevel level, const char* msg, ...) {};
    TPerfEventBeginFn Context::PerfEventBeginFn = [] (const char* name, ID3D12GraphicsCommandList* commandList, int index) {};
    TPerfEventEndFn Context::PerfEventEndFn = [] (ID3D12GraphicsCommandList* commandList) {};

    std::wstring Context::s_techniqueLocation = L"./";
    static unsigned int s_timerIndex = 0;

    ID3D12CommandSignature* ContextInternal::s_commandSignatureDispatch = nullptr;
    const int ContextInternal::variable_c_numInputNeurons = 784;
    const int ContextInternal::variable_c_numHiddenNeurons = 30;
    const int ContextInternal::variable_c_numOutputNeurons = 10;
    const int ContextInternal::variable_c_numOutputWeights = 310;
    const uint2 ContextInternal::variable_c_NNInputImageSize = {28, 28};
    const uint2 ContextInternal::variable_c_drawingCanvasSize = {256, 256};

    ID3D12PipelineState* ContextInternal::computeShader_Draw_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_Draw_rootSig = nullptr;

    ID3D12PipelineState* ContextInternal::computeShader_CalculateExtents_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_CalculateExtents_rootSig = nullptr;

    ID3D12PipelineState* ContextInternal::computeShader_Shrink_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_Shrink_rootSig = nullptr;

    ID3D12PipelineState* ContextInternal::computeShader_Hidden_Layer_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_Hidden_Layer_rootSig = nullptr;

    ID3D12PipelineState* ContextInternal::computeShader_Output_Layer_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_Output_Layer_rootSig = nullptr;

    ID3D12PipelineState* ContextInternal::computeShader_Presentation_pso = nullptr;
    ID3D12RootSignature* ContextInternal::computeShader_Presentation_rootSig = nullptr;

    template <typename T>
    T Pow2GE(const T& A)
    {
        float f = std::log2(float(A));
        f = std::ceilf(f);
        return (T)std::pow(2.0f, f);
    }

    bool CreateShared(ID3D12Device* device)
    {

        // Compute Shader: Draw
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[3];

            // Canvas
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // DrawExtents
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            // _DrawCB
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 0;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            if(!DX12Utils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Draw_rootSig, (c_debugNames ? L"Draw" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/draw.hlsl", "Draw", "cs_5_1", defines,
               ContextInternal::computeShader_Draw_rootSig, &ContextInternal::computeShader_Draw_pso, c_debugShaders, (c_debugNames ? L"Draw" : nullptr), Context::LogFn))
                return false;
        }

        // Compute Shader: CalculateExtents
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[2];

            // Canvas
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // DrawExtents
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 0;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            if(!DX12Utils::MakeRootSig(device, ranges, 2, samplers, 0, &ContextInternal::computeShader_CalculateExtents_rootSig, (c_debugNames ? L"CalculateExtents" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/CalculateExtents.hlsl", "CalculateExtents", "cs_5_1", defines,
               ContextInternal::computeShader_CalculateExtents_rootSig, &ContextInternal::computeShader_CalculateExtents_pso, c_debugShaders, (c_debugNames ? L"CalculateExtents" : nullptr), Context::LogFn))
                return false;
        }

        // Compute Shader: Shrink
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[5];

            // Canvas
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // DrawExtents
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            // NNInput
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 0;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            // ImportedImage
            ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[3].NumDescriptors = 1;
            ranges[3].BaseShaderRegister = 2;
            ranges[3].RegisterSpace = 0;
            ranges[3].OffsetInDescriptorsFromTableStart = 3;

            // _ShrinkCB
            ranges[4].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[4].NumDescriptors = 1;
            ranges[4].BaseShaderRegister = 0;
            ranges[4].RegisterSpace = 0;
            ranges[4].OffsetInDescriptorsFromTableStart = 4;

            if(!DX12Utils::MakeRootSig(device, ranges, 5, samplers, 0, &ContextInternal::computeShader_Shrink_rootSig, (c_debugNames ? L"Shrink" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/shrink.hlsl", "Shrink", "cs_5_1", defines,
               ContextInternal::computeShader_Shrink_rootSig, &ContextInternal::computeShader_Shrink_pso, c_debugShaders, (c_debugNames ? L"Shrink" : nullptr), Context::LogFn))
                return false;
        }

        // Compute Shader: Hidden_Layer
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[3];

            // NNInput
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // NNWeights
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            // HiddenLayerActivations
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 0;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            if(!DX12Utils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Hidden_Layer_rootSig, (c_debugNames ? L"Hidden_Layer" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/HiddenLayer.hlsl", "HiddenLayer", "cs_5_1", defines,
               ContextInternal::computeShader_Hidden_Layer_rootSig, &ContextInternal::computeShader_Hidden_Layer_pso, c_debugShaders, (c_debugNames ? L"Hidden_Layer" : nullptr), Context::LogFn))
                return false;
        }

        // Compute Shader: Output_Layer
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[3];

            // NNWeights
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // HiddenLayerActivations
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            // OutputLayerActivations
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 0;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            if(!DX12Utils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Output_Layer_rootSig, (c_debugNames ? L"Output_Layer" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/OutputLayer.hlsl", "OutputLayer", "cs_5_1", defines,
               ContextInternal::computeShader_Output_Layer_rootSig, &ContextInternal::computeShader_Output_Layer_pso, c_debugShaders, (c_debugNames ? L"Output_Layer" : nullptr), Context::LogFn))
                return false;
        }

        // Compute Shader: Presentation
        {
            D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;

            D3D12_DESCRIPTOR_RANGE ranges[17];

            // DrawCanvas
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            // NNInput
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[1].NumDescriptors = 1;
            ranges[1].BaseShaderRegister = 1;
            ranges[1].RegisterSpace = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 1;

            // HiddenLayerActivations
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 2;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            // OutputLayerActivations
            ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[3].NumDescriptors = 1;
            ranges[3].BaseShaderRegister = 3;
            ranges[3].RegisterSpace = 0;
            ranges[3].OffsetInDescriptorsFromTableStart = 3;

            // PresentationCanvas
            ranges[4].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[4].NumDescriptors = 1;
            ranges[4].BaseShaderRegister = 0;
            ranges[4].RegisterSpace = 0;
            ranges[4].OffsetInDescriptorsFromTableStart = 4;

            // _loadedTexture_0
            ranges[5].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[5].NumDescriptors = 1;
            ranges[5].BaseShaderRegister = 4;
            ranges[5].RegisterSpace = 0;
            ranges[5].OffsetInDescriptorsFromTableStart = 5;

            // _loadedTexture_1
            ranges[6].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[6].NumDescriptors = 1;
            ranges[6].BaseShaderRegister = 5;
            ranges[6].RegisterSpace = 0;
            ranges[6].OffsetInDescriptorsFromTableStart = 6;

            // _loadedTexture_2
            ranges[7].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[7].NumDescriptors = 1;
            ranges[7].BaseShaderRegister = 6;
            ranges[7].RegisterSpace = 0;
            ranges[7].OffsetInDescriptorsFromTableStart = 7;

            // _loadedTexture_3
            ranges[8].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[8].NumDescriptors = 1;
            ranges[8].BaseShaderRegister = 7;
            ranges[8].RegisterSpace = 0;
            ranges[8].OffsetInDescriptorsFromTableStart = 8;

            // _loadedTexture_4
            ranges[9].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[9].NumDescriptors = 1;
            ranges[9].BaseShaderRegister = 8;
            ranges[9].RegisterSpace = 0;
            ranges[9].OffsetInDescriptorsFromTableStart = 9;

            // _loadedTexture_5
            ranges[10].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[10].NumDescriptors = 1;
            ranges[10].BaseShaderRegister = 9;
            ranges[10].RegisterSpace = 0;
            ranges[10].OffsetInDescriptorsFromTableStart = 10;

            // _loadedTexture_6
            ranges[11].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[11].NumDescriptors = 1;
            ranges[11].BaseShaderRegister = 10;
            ranges[11].RegisterSpace = 0;
            ranges[11].OffsetInDescriptorsFromTableStart = 11;

            // _loadedTexture_7
            ranges[12].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[12].NumDescriptors = 1;
            ranges[12].BaseShaderRegister = 11;
            ranges[12].RegisterSpace = 0;
            ranges[12].OffsetInDescriptorsFromTableStart = 12;

            // _loadedTexture_8
            ranges[13].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[13].NumDescriptors = 1;
            ranges[13].BaseShaderRegister = 12;
            ranges[13].RegisterSpace = 0;
            ranges[13].OffsetInDescriptorsFromTableStart = 13;

            // _loadedTexture_9
            ranges[14].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[14].NumDescriptors = 1;
            ranges[14].BaseShaderRegister = 13;
            ranges[14].RegisterSpace = 0;
            ranges[14].OffsetInDescriptorsFromTableStart = 14;

            // _loadedTexture_10
            ranges[15].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[15].NumDescriptors = 1;
            ranges[15].BaseShaderRegister = 14;
            ranges[15].RegisterSpace = 0;
            ranges[15].OffsetInDescriptorsFromTableStart = 15;

            // _PresentationCB
            ranges[16].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[16].NumDescriptors = 1;
            ranges[16].BaseShaderRegister = 0;
            ranges[16].RegisterSpace = 0;
            ranges[16].OffsetInDescriptorsFromTableStart = 16;

            if(!DX12Utils::MakeRootSig(device, ranges, 17, samplers, 0, &ContextInternal::computeShader_Presentation_rootSig, (c_debugNames ? L"Presentation" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO defines[] = {
                { "__GigiDispatchMultiply", "uint3(1,1,1)" },
                { "__GigiDispatchDivide", "uint3(1,1,1)" },
                { "__GigiDispatchPreAdd", "uint3(0,0,0)" },
                { "__GigiDispatchPostAdd", "uint3(0,0,0)" },
                { nullptr, nullptr }
            };

            if(!DX12Utils::MakeComputePSO_FXC(device, Context::s_techniqueLocation.c_str(), L"shaders/Presentation.hlsl", "Presentation", "cs_5_1", defines,
               ContextInternal::computeShader_Presentation_rootSig, &ContextInternal::computeShader_Presentation_pso, c_debugShaders, (c_debugNames ? L"Presentation" : nullptr), Context::LogFn))
                return false;
        }

        // Create heaps
        if (c_numSRVDescriptors > 0 && !CreateHeap(s_srvHeap, device, c_numSRVDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, Context::LogFn))
            return false;

        if (c_numRTVDescriptors > 0 && !CreateHeap(s_rtvHeap, device, c_numRTVDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, Context::LogFn))
            return false;

        if (c_numDSVDescriptors > 0 && !CreateHeap(s_dsvHeap, device, c_numDSVDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, Context::LogFn))
            return false;

        s_heapAllocationTrackerRTV.Init(s_rtvHeap.m_heap, c_numRTVDescriptors, (int)device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV));
        s_heapAllocationTrackerDSV.Init(s_dsvHeap.m_heap, c_numDSVDescriptors, (int)device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV));

        // create indirect dispatch command
        {
            D3D12_INDIRECT_ARGUMENT_DESC dispatchArg = {};
            dispatchArg.Type						 = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

            D3D12_COMMAND_SIGNATURE_DESC dispatchDesc = {};
            dispatchDesc.ByteStride					  = sizeof(uint32_t) * 3;
            dispatchDesc.NumArgumentDescs			  = 1;
            dispatchDesc.pArgumentDescs				  = &dispatchArg;
            dispatchDesc.NodeMask					  = 0x0;

            device->CreateCommandSignature(
                &dispatchDesc,
                nullptr,
                IID_PPV_ARGS(&ContextInternal::s_commandSignatureDispatch));
        }

        return true;
    }

    void DestroyShared()
    {

        if(ContextInternal::computeShader_Draw_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Draw_pso);
            ContextInternal::computeShader_Draw_pso = nullptr;
        }

        if(ContextInternal::computeShader_Draw_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Draw_rootSig);
            ContextInternal::computeShader_Draw_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_CalculateExtents_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_CalculateExtents_pso);
            ContextInternal::computeShader_CalculateExtents_pso = nullptr;
        }

        if(ContextInternal::computeShader_CalculateExtents_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_CalculateExtents_rootSig);
            ContextInternal::computeShader_CalculateExtents_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Shrink_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Shrink_pso);
            ContextInternal::computeShader_Shrink_pso = nullptr;
        }

        if(ContextInternal::computeShader_Shrink_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Shrink_rootSig);
            ContextInternal::computeShader_Shrink_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Hidden_Layer_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Hidden_Layer_pso);
            ContextInternal::computeShader_Hidden_Layer_pso = nullptr;
        }

        if(ContextInternal::computeShader_Hidden_Layer_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Hidden_Layer_rootSig);
            ContextInternal::computeShader_Hidden_Layer_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Output_Layer_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Output_Layer_pso);
            ContextInternal::computeShader_Output_Layer_pso = nullptr;
        }

        if(ContextInternal::computeShader_Output_Layer_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Output_Layer_rootSig);
            ContextInternal::computeShader_Output_Layer_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Presentation_pso)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Presentation_pso);
            ContextInternal::computeShader_Presentation_pso = nullptr;
        }

        if(ContextInternal::computeShader_Presentation_rootSig)
        {
            s_delayedRelease.Add(ContextInternal::computeShader_Presentation_rootSig);
            ContextInternal::computeShader_Presentation_rootSig = nullptr;
        }

        // Clear out heap trackers
        s_heapAllocationTrackerRTV.Release();
        s_heapAllocationTrackerDSV.Release();

        // Destroy Heaps
        DestroyHeap(s_srvHeap);
        DestroyHeap(s_rtvHeap);
        DestroyHeap(s_dsvHeap);

        // Destroy any upload buffers
        s_ubTracker.Release();

        // Finish any delayed release
        s_delayedRelease.Release();

        // Destroy indirect dispatch command
        if (ContextInternal::s_commandSignatureDispatch)
        {
            ContextInternal::s_commandSignatureDispatch->Release();
            ContextInternal::s_commandSignatureDispatch = nullptr;
        }
    }

    Context* CreateContext(ID3D12Device* device)
    {
        if (s_allContexts.size() == 0)
        {
            if (!CreateShared(device))
                return nullptr;
        }

        Context* ret = new Context;
        s_allContexts.push_back(ret);
        return ret;
    }

    void DestroyContext(Context* context)
    {
        s_allContexts.erase(std::remove(s_allContexts.begin(), s_allContexts.end(), context), s_allContexts.end());
        delete context;
        if (s_allContexts.size() == 0)
            DestroyShared();
    }

    void OnNewFrame(int framesInFlight)
    {
        s_delayedRelease.OnNewFrame(framesInFlight);
        s_ubTracker.OnNewFrame(framesInFlight);
        s_heapAllocationTrackerRTV.OnNewFrame(framesInFlight);
        s_heapAllocationTrackerDSV.OnNewFrame(framesInFlight);
    }

    int Context::GetContextCount()
    {
        return (int)s_allContexts.size();
    }

    Context* Context::GetContext(int index)
    {
        if (index >= 0 && index < GetContextCount())
            return s_allContexts[index];
        else
            return nullptr;
    }

    ID3D12Resource* Context::CreateManagedBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, const void* data, size_t size, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState)
    {
        // Make a buffer and have the context manage it
        ID3D12Resource* ret = DX12Utils::CreateBuffer(
            device,
            (unsigned int)size,
            flags,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_HEAP_TYPE_DEFAULT,
            debugName,
            Context::LogFn
        );
        AddManagedResource(ret);

        // Copy the data to the buffer if we should
        if (data != nullptr && size > 0)
            UploadBufferData(device, commandList, ret, D3D12_RESOURCE_STATE_COPY_DEST, data, (unsigned int)size);

        // Do a resource transition if we should
        if (desiredState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = ret;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = desiredState;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }

        // return the resource
        return ret;
    }

    ID3D12Resource* Context::CreateManagedTexture(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, const unsigned int size[3], unsigned int numMips, DX12Utils::ResourceType resourceType, const void* initialData, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState)
    {
        // Create a texture
        ID3D12Resource* ret = DX12Utils::CreateTexture(device, size, numMips, format, flags, D3D12_RESOURCE_STATE_COPY_DEST, resourceType, debugName, Context::LogFn);
        AddManagedResource(ret);

        // copy initial data in, if we should
        if (initialData != nullptr)
        {
            DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(format, Context::LogFn);
            UploadTextureData(device, commandList, ret, D3D12_RESOURCE_STATE_COPY_DEST, initialData, size[0] * formatInfo.bytesPerPixel);
        }

        // Put the resource into the desired state
        if (desiredState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = ret;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = desiredState;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }

        return ret;
    }

    ID3D12Resource* Context::CreateManagedTextureAndClear(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, const unsigned int size[3], unsigned int numMips, DX12Utils::ResourceType resourceType, void* clearValue, size_t clearValueSize, const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState)
    {
        // Make sure the clear value is the correct size
        DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(format, Context::LogFn);
        if (clearValue != nullptr && clearValueSize > 0 && clearValueSize != formatInfo.bytesPerPixel)
            return nullptr;

        // Copy data into the resource
        std::vector<unsigned char> expandedClearValue;
        void* initialData = nullptr;
        if (clearValue != nullptr && clearValueSize > 0)
        {
            expandedClearValue.resize(size[0] * size[1] * size[2] * formatInfo.bytesPerPixel);
            unsigned char* dest = expandedClearValue.data();
            for (size_t i = 0; i < size[0] * size[1] * size[2]; ++i)
            {
                memcpy(dest, clearValue, formatInfo.bytesPerPixel);
                dest += formatInfo.bytesPerPixel;
            }
            initialData = expandedClearValue.data();
        }

        // make and return the texture
        return CreateManagedTexture(device, commandList, flags, format, size, numMips, resourceType, initialData, debugName, desiredState);
    }

    ID3D12Resource* Context::CreateManagedTextureFromFile(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format, DX12Utils::ResourceType resourceType, const char* fileName, bool sourceIsSRGB, unsigned int size[3], const wchar_t* debugName, D3D12_RESOURCE_STATES desiredState)
    {
        // Get the desired channel type
        DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(format, Context::LogFn);
        DX12Utils::TextureCache::Type desiredChannelType = DX12Utils::TextureCache::Type::U8;
        if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
            desiredChannelType = DX12Utils::TextureCache::Type::U8;
        else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
            desiredChannelType = DX12Utils::TextureCache::Type::F32;
        else
            return nullptr;

        if (resourceType == DX12Utils::ResourceType::Texture2D)
        {
            // Load the texture and convert as necessary
            DX12Utils::TextureCache::Texture texture = DX12Utils::TextureCache::GetAs(fileName, sourceIsSRGB, desiredChannelType, formatInfo.sRGB, formatInfo.channelCount);
            if (!texture.Valid())
                return nullptr;

            // store off image properties
            size[0] = texture.width;
            size[1] = texture.height;
            size[2] = 1;

            // make and return the texture
            return CreateManagedTexture(device, commandList, flags, format, size, 1, resourceType, texture.pixels.data(), debugName, desiredState);
        }
        else if (resourceType == DX12Utils::ResourceType::Texture2DArray ||
                 resourceType == DX12Utils::ResourceType::Texture3D ||
                 resourceType == DX12Utils::ResourceType::TextureCube)
        {
            static const char* c_cubeMapNames[] =
            {
                "Right",
                "Left",
                "Up",
                "Down",
                "Front",
                "Back"
            };

            bool useCubeMapNames = (resourceType == DX12Utils::ResourceType::TextureCube && strstr(fileName, "%s") != nullptr);
            bool hasPercentI = strstr(fileName, "%i") != nullptr;
            if (!useCubeMapNames && !hasPercentI)
                return nullptr;

            std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;

            // Load multiple textures
            int textureIndex = -1;
            while (1)
            {
                textureIndex++;
                char indexedFileName[1024];

                if (useCubeMapNames)
                    sprintf_s(indexedFileName, fileName, c_cubeMapNames[textureIndex]);
                else
                    sprintf_s(indexedFileName, fileName, textureIndex);

                // Load the texture and convert as necessary
                DX12Utils::TextureCache::Texture loadedTextureSlice = DX12Utils::TextureCache::GetAs(indexedFileName, sourceIsSRGB, desiredChannelType, formatInfo.sRGB, formatInfo.channelCount);
                if (!loadedTextureSlice.Valid())
                {
                    if (textureIndex == 0)
                        return nullptr;
                    break;
                }

                // make sure the textures are the same size
                if (textureIndex > 0 && (loadedTextureSlice.width != loadedTextureSlices[0].width || loadedTextureSlice.height != loadedTextureSlices[0].height))
                    return nullptr;

                loadedTextureSlices.push_back(loadedTextureSlice);
            }

            // store the texture size
            size[0] = loadedTextureSlices[0].width;
            size[1] = loadedTextureSlices[0].height;
            size[2] = (unsigned int)loadedTextureSlices.size();

            // gather up all pixels into a contiguous chunk of memory
            std::vector<unsigned char> allPixels;
            for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                allPixels.insert(allPixels.end(), texture.pixels.begin(), texture.pixels.end());

            // make and return the texture
            return CreateManagedTexture(device, commandList, flags, format, size, 1, resourceType, allPixels.data(), debugName, desiredState);
        }
        else
            return nullptr;
    }

    void Context::UploadTextureData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* texture, D3D12_RESOURCE_STATES textureState, const void* data, unsigned int unalignedPitch)
    {
        // Get information about the texture
        int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
        D3D12_RESOURCE_DESC textureDesc = texture->GetDesc();

        // transition the resource to copy dest if it isn't already
        if (textureState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = texture;
            barrier.Transition.StateBefore = textureState;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }

        // 3d textures do a single copy because it's a single sub resource.
        if (textureDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        {
            // Get the upload buffer
            DX12Utils::UploadBufferTracker::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * textureDesc.Height * textureDesc.DepthOrArraySize, Context::LogFn, false);

            // Map, copy, unmap
            {
                unsigned char* dest = nullptr;
                D3D12_RANGE readRange = { 0, 0 };
                HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                if (FAILED(hr))
                {
                    Context::LogFn(LogLevel::Error, "Could not map upload buffer.");
                }
                else
                {
                    const unsigned char* src = (const unsigned char*)data;
                    for (int iz = 0; iz < textureDesc.DepthOrArraySize; ++iz)
                    {
                        for (int iy = 0; iy < (int)textureDesc.Height; ++iy)
                        {
                            memcpy(dest, src, unalignedPitch);
                            src += unalignedPitch;
                            dest += alignedPitch;
                        }
                    }

                    uploadBuffer->buffer->Unmap(0, nullptr);
                }
            }

            // copy the upload buffer into the texture
            {
                unsigned char layoutMem[sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)];
                D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem;
                device->GetCopyableFootprints(&textureDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                D3D12_TEXTURE_COPY_LOCATION src = {};
                src.pResource = uploadBuffer->buffer;
                src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint = *layout;

                D3D12_TEXTURE_COPY_LOCATION dest = {};
                dest.pResource = texture;
                dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                dest.SubresourceIndex = 0;

                commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
            }
        }
        // 2d array textures do a copy for each slice
        else if (textureDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        {
            for (int iz = 0; iz < textureDesc.DepthOrArraySize; ++iz)
            {
                // Get the upload buffer
                DX12Utils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * textureDesc.Height, Context::LogFn, false);

                // Map, copy, unmap
                {
                    unsigned char* dest = nullptr;
                    D3D12_RANGE readRange = { 0, 0 };
                    HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                    if (FAILED(hr))
                    {
                        Context::LogFn(LogLevel::Error, "Could not map upload buffer.");
                    }
                    else
                    {
                        const unsigned char* src = &((const unsigned char*)data)[unalignedPitch * textureDesc.Height * iz];
                        for (int iy = 0; iy < (int)textureDesc.Height; ++iy)
                        {
                            memcpy(dest, src, unalignedPitch);
                            src += unalignedPitch;
                            dest += alignedPitch;
                        }

                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }
                }

                 // copy the upload buffer into the texture
                 {
                     unsigned char layoutMem[sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)];
                     D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem;
                     device->GetCopyableFootprints(&textureDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                     D3D12_TEXTURE_COPY_LOCATION src = {};
                     src.pResource = uploadBuffer->buffer;
                     src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                     src.PlacedFootprint = *layout;

                     D3D12_TEXTURE_COPY_LOCATION dest = {};
                     dest.pResource = texture;
                     dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                     dest.SubresourceIndex = iz;

                     commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                 }
            }
        }
        else
        {
            Context::LogFn(LogLevel::Error, "Unhandled texture dimension.");
        }

        // transition the resource back to what it was
        if (textureState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = texture;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = textureState;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }
    }

    void Context::UploadBufferData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* buffer, D3D12_RESOURCE_STATES bufferState, const void* data, unsigned int dataSize)
    {
        // Get the upload buffer
        DX12Utils::UploadBufferTracker::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, dataSize, Context::LogFn, false);

        // copy cpu data to the upload buffer
        {
            void* start = nullptr;
            HRESULT hr = uploadBuffer->buffer->Map(0, nullptr, reinterpret_cast<void**>(&start));
            if(hr)
            {
                Context::LogFn(LogLevel::Error, "Could not map upload buffer");
                return;
            }

            memcpy(start, data, dataSize);

            uploadBuffer->buffer->Unmap(0, nullptr);
        }

        // transition the resource to copy dest if it isn't already
        if (bufferState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = buffer;
            barrier.Transition.StateBefore = bufferState;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }

        // copy the resource
        commandList->CopyResource(buffer, uploadBuffer->buffer);

        // transition the resource back to what it was
        if (bufferState != D3D12_RESOURCE_STATE_COPY_DEST)
        {
            D3D12_RESOURCE_BARRIER barrier;

            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = buffer;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = bufferState;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, &barrier);
        }
    }

    int Context::GetRTV(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format, D3D12_RTV_DIMENSION dimension, int arrayIndex, int mipIndex, const char* debugName)
    {
        // Make the key
        DX12Utils::SubResourceHeapAllocationInfo key;
        key.resource = resource;
        key.arrayIndex = arrayIndex;
        key.mipIndex = mipIndex;

        // If it already exists, use it
        auto it = m_internal.m_RTVCache.find(key);
        if (it != m_internal.m_RTVCache.end())
            return it->second;

        // Allocate an RTV index
        int rtvIndex = -1;
        if (!s_heapAllocationTrackerRTV.Allocate(rtvIndex, debugName))
            return -1;

        // Create the RTV
        if (!DX12Utils::CreateRTV(device, resource, s_heapAllocationTrackerRTV.GetCPUHandle(rtvIndex), format, dimension, arrayIndex, mipIndex))
        {
            s_heapAllocationTrackerRTV.Free(rtvIndex);
            return -1;
        }

        // store the result
        m_internal.m_RTVCache[key] = rtvIndex;
        return rtvIndex;
    }

    int Context::GetDSV(ID3D12Device* device, ID3D12Resource* resource, DXGI_FORMAT format, D3D12_DSV_DIMENSION dimension, int arrayIndex, int mipIndex, const char* debugName)
    {
	    // Make the key
        DX12Utils::SubResourceHeapAllocationInfo key;
        key.resource = resource;
        key.arrayIndex = arrayIndex;
        key.mipIndex = mipIndex;

	    // If it already exists, use it
	    auto it = m_internal.m_DSVCache.find(key);
	    if (it != m_internal.m_DSVCache.end())
            return it->second;

        // Allocate a DSV index
        int dsvIndex = -1;
        if (!s_heapAllocationTrackerDSV.Allocate(dsvIndex, debugName))
            return -1;

        // Create the DSV
        if (!DX12Utils::CreateDSV(device, resource, s_heapAllocationTrackerDSV.GetCPUHandle(dsvIndex), format, dimension, arrayIndex, mipIndex))
        {
            s_heapAllocationTrackerDSV.Free(dsvIndex);
            return -1;
        }

        // store the result
        m_internal.m_DSVCache[key] = dsvIndex;
        return dsvIndex;
    }

    const ProfileEntry* Context::ReadbackProfileData(ID3D12CommandQueue* commandQueue, int& numItems)
    {
        numItems = 0;

        if (!m_profile || !m_internal.m_TimestampReadbackBuffer)
            return nullptr;

        uint64_t GPUFrequency;
        commandQueue->GetTimestampFrequency(&GPUFrequency);
        double GPUTickDelta = 1.0 / static_cast<double>(GPUFrequency);

        D3D12_RANGE range;
        range.Begin = 0;
        range.End = ((6 + 1) * 2) * sizeof(uint64_t);

        uint64_t* timeStampBuffer = nullptr;
        m_internal.m_TimestampReadbackBuffer->Map(0, &range, (void**)&timeStampBuffer);

        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: Draw
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: CalculateExtents
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: Shrink
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: Hidden_Layer
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: Output_Layer
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+2] - timeStampBuffer[numItems*2+1])); numItems++; // compute shader: Presentation
        m_profileData[numItems].m_gpu = float(GPUTickDelta * double(timeStampBuffer[numItems*2+1] - timeStampBuffer[0])); numItems++; // GPU total

        D3D12_RANGE emptyRange = {};
        m_internal.m_TimestampReadbackBuffer->Unmap(0, &emptyRange);

        return m_profileData;
    }

    Context::~Context()
    {
        for (const auto& pair : m_internal.m_RTVCache)
            s_heapAllocationTrackerRTV.Free(pair.second);
        m_internal.m_RTVCache.clear();

        for (const auto& pair : m_internal.m_DSVCache)
            s_heapAllocationTrackerDSV.Free(pair.second);
        m_internal.m_DSVCache.clear();

        for (ID3D12Resource* resource : m_internal.m_managedResources)
            resource->Release();
        m_internal.m_managedResources.clear();

        if(m_internal.m_TimestampQueryHeap)
        {
            m_internal.m_TimestampQueryHeap->Release();
            m_internal.m_TimestampQueryHeap = nullptr;
        }

        if(m_internal.m_TimestampReadbackBuffer)
        {
            m_internal.m_TimestampReadbackBuffer->Release();
            m_internal.m_TimestampReadbackBuffer = nullptr;
        }

        if(m_internal.texture_Drawing_Canvas)
        {
            s_delayedRelease.Add(m_internal.texture_Drawing_Canvas);
            m_internal.texture_Drawing_Canvas = nullptr;
        }

        if(m_internal.texture_NN_Input)
        {
            s_delayedRelease.Add(m_internal.texture_NN_Input);
            m_internal.texture_NN_Input = nullptr;
        }

        if(m_internal.buffer_Hidden_Layer_Activations)
        {
            s_delayedRelease.Add(m_internal.buffer_Hidden_Layer_Activations);
            m_internal.buffer_Hidden_Layer_Activations = nullptr;
        }

        if(m_internal.buffer_Output_Layer_Activations)
        {
            s_delayedRelease.Add(m_internal.buffer_Output_Layer_Activations);
            m_internal.buffer_Output_Layer_Activations = nullptr;
        }

        if(m_internal.buffer_Draw_Extents)
        {
            s_delayedRelease.Add(m_internal.buffer_Draw_Extents);
            m_internal.buffer_Draw_Extents = nullptr;
        }

        // _DrawCB
        if (m_internal.constantBuffer__DrawCB)
        {
            s_delayedRelease.Add(m_internal.constantBuffer__DrawCB);
            m_internal.constantBuffer__DrawCB = nullptr;
        }

        // _ShrinkCB
        if (m_internal.constantBuffer__ShrinkCB)
        {
            s_delayedRelease.Add(m_internal.constantBuffer__ShrinkCB);
            m_internal.constantBuffer__ShrinkCB = nullptr;
        }

        if(m_internal.texture__loadedTexture_0)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_0);
            m_internal.texture__loadedTexture_0 = nullptr;
        }

        if(m_internal.texture__loadedTexture_1)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_1);
            m_internal.texture__loadedTexture_1 = nullptr;
        }

        if(m_internal.texture__loadedTexture_2)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_2);
            m_internal.texture__loadedTexture_2 = nullptr;
        }

        if(m_internal.texture__loadedTexture_3)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_3);
            m_internal.texture__loadedTexture_3 = nullptr;
        }

        if(m_internal.texture__loadedTexture_4)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_4);
            m_internal.texture__loadedTexture_4 = nullptr;
        }

        if(m_internal.texture__loadedTexture_5)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_5);
            m_internal.texture__loadedTexture_5 = nullptr;
        }

        if(m_internal.texture__loadedTexture_6)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_6);
            m_internal.texture__loadedTexture_6 = nullptr;
        }

        if(m_internal.texture__loadedTexture_7)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_7);
            m_internal.texture__loadedTexture_7 = nullptr;
        }

        if(m_internal.texture__loadedTexture_8)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_8);
            m_internal.texture__loadedTexture_8 = nullptr;
        }

        if(m_internal.texture__loadedTexture_9)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_9);
            m_internal.texture__loadedTexture_9 = nullptr;
        }

        if(m_internal.texture__loadedTexture_10)
        {
            s_delayedRelease.Add(m_internal.texture__loadedTexture_10);
            m_internal.texture__loadedTexture_10 = nullptr;
        }

        // _PresentationCB
        if (m_internal.constantBuffer__PresentationCB)
        {
            s_delayedRelease.Add(m_internal.constantBuffer__PresentationCB);
            m_internal.constantBuffer__PresentationCB = nullptr;
        }
    }

    void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList)
    {
        // reset the timer index
        s_timerIndex = 0;

        ScopedPerfEvent scopedPerf("mnist", commandList, 28);

        std::chrono::high_resolution_clock::time_point startPointCPUTechnique;
        if(context->m_profile)
        {
            startPointCPUTechnique = std::chrono::high_resolution_clock::now();
            if(context->m_internal.m_TimestampQueryHeap == nullptr)
            {
                D3D12_QUERY_HEAP_DESC QueryHeapDesc;
                QueryHeapDesc.Count = (6+1) * 2;
                QueryHeapDesc.NodeMask = 1;
                QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
                device->CreateQueryHeap(&QueryHeapDesc, IID_PPV_ARGS(&context->m_internal.m_TimestampQueryHeap));
                if (c_debugNames)
                    context->m_internal.m_TimestampQueryHeap->SetName(L"mnist Time Stamp Query Heap");

                context->m_internal.m_TimestampReadbackBuffer = DX12Utils::CreateBuffer(device, sizeof(uint64_t) * (6+1) * 2, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK, (c_debugNames ? L"mnist Time Stamp Query Heap" : nullptr), nullptr);
            }
            commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
        }

        // Set variables
        if(context->m_internal.variable_initialized)
            context->m_input.variable_iFrame = context->m_input.variable_iFrame + 1;
        if(!context->m_internal.variable_initialized)
            context->m_input.variable_iFrame = 0 + 0;

        if (!context->m_input.buffer_NN_Weights)
        {
            Context::LogFn(LogLevel::Error, "mnist: Imported buffer \"NN_Weights\" is null.\n");
            return;
        }

        if (!context->m_input.texture_Presentation_Canvas)
        {
            Context::LogFn(LogLevel::Error, "mnist: Imported texture \"Presentation_Canvas\" is null.\n");
            return;
        }

        if (!context->m_input.texture_Imported_Image)
        {
            Context::LogFn(LogLevel::Error, "mnist: Imported texture \"Imported_Image\" is null.\n");
            return;
        }

        // Make sure internally owned resources are created and are the right size and format
        context->EnsureResourcesCreated(device, commandList);

        // set the heaps
        ID3D12DescriptorHeap* heaps[] =
        {
            s_srvHeap.m_heap,
        };
        commandList->SetDescriptorHeaps(_countof(heaps), heaps);

        // Make sure imported resources are in the correct state
        {
            int barrierCount = 0;
            D3D12_RESOURCE_BARRIER barriers[3];

            if(context->m_input.buffer_NN_Weights_state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.buffer_NN_Weights;
                barriers[barrierCount].Transition.StateBefore = context->m_input.buffer_NN_Weights_state;
                barriers[barrierCount].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }

            if(context->m_input.texture_Presentation_Canvas_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.texture_Presentation_Canvas;
                barriers[barrierCount].Transition.StateBefore = context->m_input.texture_Presentation_Canvas_state;
                barriers[barrierCount].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }
            else
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].UAV.pResource = context->m_input.texture_Presentation_Canvas;
                barrierCount++;
            }

            if(context->m_input.texture_Imported_Image_state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.texture_Imported_Image;
                barriers[barrierCount].Transition.StateBefore = context->m_input.texture_Imported_Image_state;
                barriers[barrierCount].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }

            if(barrierCount > 0)
                commandList->ResourceBarrier(barrierCount, barriers);
        }

        // Shader Constants: _DrawCB
        {
            context->m_internal.constantBuffer__DrawCB_cpu.Clear = context->m_input.variable_Clear;
            context->m_internal.constantBuffer__DrawCB_cpu.MouseState = context->m_input.variable_MouseState;
            context->m_internal.constantBuffer__DrawCB_cpu.MouseStateLastFrame = context->m_input.variable_MouseStateLastFrame;
            context->m_internal.constantBuffer__DrawCB_cpu.PenSize = context->m_input.variable_PenSize;
            context->m_internal.constantBuffer__DrawCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            context->m_internal.constantBuffer__DrawCB_cpu.iFrame = context->m_input.variable_iFrame;
            DX12Utils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__DrawCB, context->m_internal.constantBuffer__DrawCB_cpu, Context::LogFn);
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.texture_Drawing_Canvas;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].Transition.pResource = context->m_internal.buffer_Draw_Extents;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(2, barriers);
        }

        // Compute Shader: Draw
        {
            ScopedPerfEvent scopedPerf("Compute Shader: Draw", commandList, 1);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Draw_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Draw_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count, 0 },
                { context->m_internal.constantBuffer__DrawCB, DXGI_FORMAT_UNKNOWN, DX12Utils::AccessType::CBV, DX12Utils::ResourceType::Buffer, false, 256, 1, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_Drawing_Canvas_size[0],
                context->m_internal.texture_Drawing_Canvas_size[1],
                context->m_internal.texture_Drawing_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Draw";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.texture_Drawing_Canvas;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].UAV.pResource = context->m_internal.buffer_Draw_Extents;

            commandList->ResourceBarrier(2, barriers);
        }

        // Compute Shader: CalculateExtents
        {
            ScopedPerfEvent scopedPerf("Compute Shader: CalculateExtents", commandList, 13);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_CalculateExtents_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_CalculateExtents_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 2, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_Drawing_Canvas_size[0],
                context->m_internal.texture_Drawing_Canvas_size[1],
                context->m_internal.texture_Drawing_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "CalculateExtents";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Shader Constants: _ShrinkCB
        {
            context->m_internal.constantBuffer__ShrinkCB_cpu.NormalizeDrawing = context->m_input.variable_NormalizeDrawing;
            context->m_internal.constantBuffer__ShrinkCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            DX12Utils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__ShrinkCB, context->m_internal.constantBuffer__ShrinkCB_cpu, Context::LogFn);
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.texture_NN_Input;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].Transition.pResource = context->m_internal.buffer_Draw_Extents;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(2, barriers);
        }

        // Compute Shader: Shrink
        {
            ScopedPerfEvent scopedPerf("Compute Shader: Shrink", commandList, 3);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Shrink_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Shrink_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count, 0 },
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_input.texture_Imported_Image, context->m_input.texture_Imported_Image_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.constantBuffer__ShrinkCB, DXGI_FORMAT_UNKNOWN, DX12Utils::AccessType::CBV, DX12Utils::ResourceType::Buffer, false, 256, 1, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 5, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_NN_Input_size[0],
                context->m_internal.texture_NN_Input_size[1],
                context->m_internal.texture_NN_Input_size[2]
            };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Shrink";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.texture_NN_Input;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].Transition.pResource = context->m_internal.buffer_Hidden_Layer_Activations;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(2, barriers);
        }

        // Compute Shader: Hidden_Layer
        {
            ScopedPerfEvent scopedPerf("Compute Shader: Hidden_Layer", commandList, 5);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Hidden_Layer_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Hidden_Layer_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_input.buffer_NN_Weights, context->m_input.buffer_NN_Weights_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_input.buffer_NN_Weights_stride, context->m_input.buffer_NN_Weights_count, 0 },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = { (unsigned int)context->m_internal.variable_c_numHiddenNeurons, 1, 1 };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 64 - 1) / 64,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 1 - 1) / 1,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Hidden_Layer";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[2];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.buffer_Hidden_Layer_Activations;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[1].Transition.pResource = context->m_internal.buffer_Output_Layer_Activations;
            barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(2, barriers);
        }

        // Compute Shader: Output_Layer
        {
            ScopedPerfEvent scopedPerf("Compute Shader: Output_Layer", commandList, 7);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Output_Layer_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Output_Layer_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_input.buffer_NN_Weights, context->m_input.buffer_NN_Weights_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_input.buffer_NN_Weights_stride, context->m_input.buffer_NN_Weights_count, 0 },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count, 0 },
                { context->m_internal.buffer_Output_Layer_Activations, context->m_internal.buffer_Output_Layer_Activations_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Output_Layer_Activations_stride, context->m_internal.buffer_Output_Layer_Activations_count, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = { (unsigned int)context->m_internal.variable_c_numOutputNeurons, 1, 1 };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 64 - 1) / 64,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 1 - 1) / 1,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Output_Layer";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Shader Constants: _PresentationCB
        {
            context->m_internal.constantBuffer__PresentationCB_cpu.MouseState = context->m_input.variable_MouseState;
            context->m_internal.constantBuffer__PresentationCB_cpu.PenSize = context->m_input.variable_PenSize;
            context->m_internal.constantBuffer__PresentationCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            DX12Utils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__PresentationCB, context->m_internal.constantBuffer__PresentationCB_cpu, Context::LogFn);
        }

        // Transition resources for the next action
        {
            D3D12_RESOURCE_BARRIER barriers[1];

            barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barriers[0].Transition.pResource = context->m_internal.buffer_Output_Layer_Activations;
            barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

            commandList->ResourceBarrier(1, barriers);
        }

        // Compute Shader: Presentation
        {
            ScopedPerfEvent scopedPerf("Compute Shader: Presentation", commandList, 9);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Presentation_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Presentation_pso);

            DX12Utils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count, 0 },
                { context->m_internal.buffer_Output_Layer_Activations, context->m_internal.buffer_Output_Layer_Activations_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Buffer, false, context->m_internal.buffer_Output_Layer_Activations_stride, context->m_internal.buffer_Output_Layer_Activations_count, 0 },
                { context->m_input.texture_Presentation_Canvas, context->m_input.texture_Presentation_Canvas_format, DX12Utils::AccessType::UAV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_0, context->m_internal.texture__loadedTexture_0_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_1, context->m_internal.texture__loadedTexture_1_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_2, context->m_internal.texture__loadedTexture_2_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_3, context->m_internal.texture__loadedTexture_3_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_4, context->m_internal.texture__loadedTexture_4_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_5, context->m_internal.texture__loadedTexture_5_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_6, context->m_internal.texture__loadedTexture_6_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_7, context->m_internal.texture__loadedTexture_7_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_8, context->m_internal.texture__loadedTexture_8_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_9, context->m_internal.texture__loadedTexture_9_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.texture__loadedTexture_10, context->m_internal.texture__loadedTexture_10_format, DX12Utils::AccessType::SRV, DX12Utils::ResourceType::Texture2D, false, 0, 0, 0 },
                { context->m_internal.constantBuffer__PresentationCB, DXGI_FORMAT_UNKNOWN, DX12Utils::AccessType::CBV, DX12Utils::ResourceType::Buffer, false, 256, 1, 0 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 17, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_input.texture_Presentation_Canvas_size[0],
                context->m_input.texture_Presentation_Canvas_size[1],
                context->m_input.texture_Presentation_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                (((baseDispatchSize[0] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[1] + 0) * 1) / 1 + 0 + 8 - 1) / 8,
                (((baseDispatchSize[2] + 0) * 1) / 1 + 0 + 1 - 1) / 1
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Presentation";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
        }

        // Make sure imported resources are put back in the state they were given to us in
        {
            int barrierCount = 0;
            D3D12_RESOURCE_BARRIER barriers[3];

            if(context->m_input.buffer_NN_Weights_state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.buffer_NN_Weights;
                barriers[barrierCount].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[barrierCount].Transition.StateAfter = context->m_input.buffer_NN_Weights_state;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }

            if(context->m_input.texture_Presentation_Canvas_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.texture_Presentation_Canvas;
                barriers[barrierCount].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[barrierCount].Transition.StateAfter = context->m_input.texture_Presentation_Canvas_state;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }
            else
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].UAV.pResource = context->m_input.texture_Presentation_Canvas;
                barrierCount++;
            }

            if(context->m_input.texture_Imported_Image_state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            {
                barriers[barrierCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[barrierCount].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barriers[barrierCount].Transition.pResource = context->m_input.texture_Imported_Image;
                barriers[barrierCount].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barriers[barrierCount].Transition.StateAfter = context->m_input.texture_Imported_Image_state;
                barriers[barrierCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barrierCount++;
            }

            if(barrierCount > 0)
                commandList->ResourceBarrier(barrierCount, barriers);
        }

        if(context->m_profile)
        {
            context->m_profileData[(s_timerIndex-1)/2].m_label = "Total";
            context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPUTechnique).count();
            commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            commandList->ResolveQueryData(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, s_timerIndex, context->m_internal.m_TimestampReadbackBuffer, 0);
        }

        // Set variables
        context->m_internal.variable_initialized = true || true;
    }

    void Context::EnsureResourcesCreated(ID3D12Device* device, ID3D12GraphicsCommandList* commandList)
    {
        bool dirty = false;

        // Drawing_Canvas
        {

            unsigned int baseSize[3] = { (unsigned int)m_internal.variable_c_drawingCanvasSize[0], (unsigned int)m_internal.variable_c_drawingCanvasSize[1], 1 };

            unsigned int desiredSize[3] = {
                ((baseSize[0] + 0) * 1) / 1 + 0,
                ((baseSize[1] + 0) * 1) / 1 + 0,
                ((baseSize[2] + 0) * 1) / 1 + 0
            };

            static const unsigned int desiredNumMips = 1;

            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R8_UNORM;

            if(!m_internal.texture_Drawing_Canvas ||
               m_internal.texture_Drawing_Canvas_size[0] != desiredSize[0] ||
               m_internal.texture_Drawing_Canvas_size[1] != desiredSize[1] ||
               m_internal.texture_Drawing_Canvas_size[2] != desiredSize[2] ||
               m_internal.texture_Drawing_Canvas_numMips != desiredNumMips ||
               m_internal.texture_Drawing_Canvas_format != desiredFormat)
            {
                dirty = true;
                if(m_internal.texture_Drawing_Canvas)
                    s_delayedRelease.Add(m_internal.texture_Drawing_Canvas);

                m_internal.texture_Drawing_Canvas = DX12Utils::CreateTexture(device, desiredSize, desiredNumMips, desiredFormat, m_internal.texture_Drawing_Canvas_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"Drawing_Canvas" : nullptr), Context::LogFn);
                m_internal.texture_Drawing_Canvas_size[0] = desiredSize[0];
                m_internal.texture_Drawing_Canvas_size[1] = desiredSize[1];
                m_internal.texture_Drawing_Canvas_size[2] = desiredSize[2];
                m_internal.texture_Drawing_Canvas_numMips = desiredNumMips;
                m_internal.texture_Drawing_Canvas_format = desiredFormat;
            }
        }

        // NN_Input
        {
            unsigned int baseSize[3] = { 1, 1, 1 };

            unsigned int desiredSize[3] = {
                ((baseSize[0] + 0) * 28) / 1 + 0,
                ((baseSize[1] + 0) * 28) / 1 + 0,
                ((baseSize[2] + 0) * 1) / 1 + 0
            };

            static const unsigned int desiredNumMips = 1;

            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R8_UNORM;

            if(!m_internal.texture_NN_Input ||
               m_internal.texture_NN_Input_size[0] != desiredSize[0] ||
               m_internal.texture_NN_Input_size[1] != desiredSize[1] ||
               m_internal.texture_NN_Input_size[2] != desiredSize[2] ||
               m_internal.texture_NN_Input_numMips != desiredNumMips ||
               m_internal.texture_NN_Input_format != desiredFormat)
            {
                dirty = true;
                if(m_internal.texture_NN_Input)
                    s_delayedRelease.Add(m_internal.texture_NN_Input);

                m_internal.texture_NN_Input = DX12Utils::CreateTexture(device, desiredSize, desiredNumMips, desiredFormat, m_internal.texture_NN_Input_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"NN_Input" : nullptr), Context::LogFn);
                m_internal.texture_NN_Input_size[0] = desiredSize[0];
                m_internal.texture_NN_Input_size[1] = desiredSize[1];
                m_internal.texture_NN_Input_size[2] = desiredSize[2];
                m_internal.texture_NN_Input_numMips = desiredNumMips;
                m_internal.texture_NN_Input_format = desiredFormat;
            }
        }

        // Hidden_Layer_Activations
        {

            unsigned int baseCount = (unsigned int)m_internal.variable_c_numHiddenNeurons;
            unsigned int desiredCount = ((baseCount + 0 ) * 1) / 1 + 0;
            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R32_FLOAT;
            unsigned int desiredStride = 0;

            if(!m_internal.buffer_Hidden_Layer_Activations ||
               m_internal.buffer_Hidden_Layer_Activations_count != desiredCount ||
               m_internal.buffer_Hidden_Layer_Activations_format != desiredFormat ||
               m_internal.buffer_Hidden_Layer_Activations_stride != desiredStride)
            {
                dirty = true;
                if(m_internal.buffer_Hidden_Layer_Activations)
                    s_delayedRelease.Add(m_internal.buffer_Hidden_Layer_Activations);

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DX12Utils::Get_DXGI_FORMAT_Info(desiredFormat, Context::LogFn).bytesPerPixel);

                m_internal.buffer_Hidden_Layer_Activations = DX12Utils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Hidden_Layer_Activations_flags, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Hidden_Layer_Activations" : nullptr), Context::LogFn);
                m_internal.buffer_Hidden_Layer_Activations_count = desiredCount;
                m_internal.buffer_Hidden_Layer_Activations_format = desiredFormat;
                m_internal.buffer_Hidden_Layer_Activations_stride = desiredStride;
            }
        }

        // Output_Layer_Activations
        {

            unsigned int baseCount = (unsigned int)m_internal.variable_c_numOutputNeurons;
            unsigned int desiredCount = ((baseCount + 0 ) * 1) / 1 + 0;
            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R32_FLOAT;
            unsigned int desiredStride = 0;

            if(!m_internal.buffer_Output_Layer_Activations ||
               m_internal.buffer_Output_Layer_Activations_count != desiredCount ||
               m_internal.buffer_Output_Layer_Activations_format != desiredFormat ||
               m_internal.buffer_Output_Layer_Activations_stride != desiredStride)
            {
                dirty = true;
                if(m_internal.buffer_Output_Layer_Activations)
                    s_delayedRelease.Add(m_internal.buffer_Output_Layer_Activations);

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DX12Utils::Get_DXGI_FORMAT_Info(desiredFormat, Context::LogFn).bytesPerPixel);

                m_internal.buffer_Output_Layer_Activations = DX12Utils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Output_Layer_Activations_flags, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Output_Layer_Activations" : nullptr), Context::LogFn);
                m_internal.buffer_Output_Layer_Activations_count = desiredCount;
                m_internal.buffer_Output_Layer_Activations_format = desiredFormat;
                m_internal.buffer_Output_Layer_Activations_stride = desiredStride;
            }
        }

        // Draw_Extents
        {
            unsigned int baseCount = 1;
            unsigned int desiredCount = ((baseCount + 0 ) * 1) / 1 + 0;
            DXGI_FORMAT desiredFormat = DXGI_FORMAT_UNKNOWN;
            unsigned int desiredStride = 28;

            if(!m_internal.buffer_Draw_Extents ||
               m_internal.buffer_Draw_Extents_count != desiredCount ||
               m_internal.buffer_Draw_Extents_format != desiredFormat ||
               m_internal.buffer_Draw_Extents_stride != desiredStride)
            {
                dirty = true;
                if(m_internal.buffer_Draw_Extents)
                    s_delayedRelease.Add(m_internal.buffer_Draw_Extents);

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DX12Utils::Get_DXGI_FORMAT_Info(desiredFormat, Context::LogFn).bytesPerPixel);

                m_internal.buffer_Draw_Extents = DX12Utils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Draw_Extents_flags, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Draw_Extents" : nullptr), Context::LogFn);
                m_internal.buffer_Draw_Extents_count = desiredCount;
                m_internal.buffer_Draw_Extents_format = desiredFormat;
                m_internal.buffer_Draw_Extents_stride = desiredStride;
            }
        }

        // _DrawCB
        if (m_internal.constantBuffer__DrawCB == nullptr)
        {
            dirty = true;
            m_internal.constantBuffer__DrawCB = DX12Utils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_DrawCB" : nullptr), Context::LogFn);
        }

        // _ShrinkCB
        if (m_internal.constantBuffer__ShrinkCB == nullptr)
        {
            dirty = true;
            m_internal.constantBuffer__ShrinkCB = DX12Utils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_ShrinkCB" : nullptr), Context::LogFn);
        }

        // _loadedTexture_0
        {
            if (!m_internal.texture__loadedTexture_0)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 0.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/0.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 0.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_0_size[0] = size[0];
                m_internal.texture__loadedTexture_0_size[1] = size[1];
                m_internal.texture__loadedTexture_0_size[2] = size[2];
                m_internal.texture__loadedTexture_0_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_0_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_0 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_0_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_0" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_0, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_1
        {
            if (!m_internal.texture__loadedTexture_1)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 1.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/1.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 1.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_1_size[0] = size[0];
                m_internal.texture__loadedTexture_1_size[1] = size[1];
                m_internal.texture__loadedTexture_1_size[2] = size[2];
                m_internal.texture__loadedTexture_1_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_1_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_1 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_1_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_1" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_1, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_2
        {
            if (!m_internal.texture__loadedTexture_2)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 2.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/2.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 2.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_2_size[0] = size[0];
                m_internal.texture__loadedTexture_2_size[1] = size[1];
                m_internal.texture__loadedTexture_2_size[2] = size[2];
                m_internal.texture__loadedTexture_2_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_2_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_2 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_2_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_2" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_2, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_3
        {
            if (!m_internal.texture__loadedTexture_3)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 3.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/3.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 3.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_3_size[0] = size[0];
                m_internal.texture__loadedTexture_3_size[1] = size[1];
                m_internal.texture__loadedTexture_3_size[2] = size[2];
                m_internal.texture__loadedTexture_3_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_3_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_3 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_3_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_3" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_3, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_4
        {
            if (!m_internal.texture__loadedTexture_4)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 4.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/4.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 4.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_4_size[0] = size[0];
                m_internal.texture__loadedTexture_4_size[1] = size[1];
                m_internal.texture__loadedTexture_4_size[2] = size[2];
                m_internal.texture__loadedTexture_4_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_4_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_4 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_4_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_4" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_4, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_5
        {
            if (!m_internal.texture__loadedTexture_5)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 5.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/5.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 5.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_5_size[0] = size[0];
                m_internal.texture__loadedTexture_5_size[1] = size[1];
                m_internal.texture__loadedTexture_5_size[2] = size[2];
                m_internal.texture__loadedTexture_5_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_5_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_5 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_5_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_5" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_5, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_6
        {
            if (!m_internal.texture__loadedTexture_6)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 6.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/6.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 6.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_6_size[0] = size[0];
                m_internal.texture__loadedTexture_6_size[1] = size[1];
                m_internal.texture__loadedTexture_6_size[2] = size[2];
                m_internal.texture__loadedTexture_6_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_6_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_6 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_6_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_6" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_6, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_7
        {
            if (!m_internal.texture__loadedTexture_7)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 7.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/7.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 7.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_7_size[0] = size[0];
                m_internal.texture__loadedTexture_7_size[1] = size[1];
                m_internal.texture__loadedTexture_7_size[2] = size[2];
                m_internal.texture__loadedTexture_7_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_7_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_7 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_7_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_7" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_7, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_8
        {
            if (!m_internal.texture__loadedTexture_8)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 8.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/8.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 8.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_8_size[0] = size[0];
                m_internal.texture__loadedTexture_8_size[1] = size[1];
                m_internal.texture__loadedTexture_8_size[2] = size[2];
                m_internal.texture__loadedTexture_8_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_8_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_8 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_8_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_8" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_8, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_9
        {
            if (!m_internal.texture__loadedTexture_9)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: 9.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/9.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: 9.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_9_size[0] = size[0];
                m_internal.texture__loadedTexture_9_size[1] = size[1];
                m_internal.texture__loadedTexture_9_size[2] = size[2];
                m_internal.texture__loadedTexture_9_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_9_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_9 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_9_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_9" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_9, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _loadedTexture_10
        {
            if (!m_internal.texture__loadedTexture_10)
            {
                // Load the texture
                std::vector<DX12Utils::TextureCache::Texture> loadedTextureSlices;
                DX12Utils::DXGI_FORMAT_Info formatInfo = DX12Utils::Get_DXGI_FORMAT_Info(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, Context::LogFn);
                DX12Utils::TextureCache::Type desiredType = DX12Utils::TextureCache::Type::U8;
                if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_uint8_t)
                    desiredType = DX12Utils::TextureCache::Type::U8;
                else if (formatInfo.channelType == DX12Utils::DXGI_FORMAT_Info::ChannelType::_float)
                    desiredType = DX12Utils::TextureCache::Type::F32;
                else
                    Context::LogFn(LogLevel::Error, "Unhandled channel type for image: instructions.png");

                char loadedTextureFileName[1024];
                sprintf_s(loadedTextureFileName, "%lsassets/instructions.png", s_techniqueLocation.c_str());

                loadedTextureSlices.push_back(DX12Utils::TextureCache::GetAs(loadedTextureFileName, true, desiredType, formatInfo.sRGB, formatInfo.channelCount));
                DX12Utils::TextureCache::Texture& loadedTexture = loadedTextureSlices[0];
                if(!loadedTexture.Valid())
                    Context::LogFn(LogLevel::Error, "Could not load image: instructions.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                static const unsigned int desiredNumMips = 1;

                // Create the texture
                dirty = true;
                m_internal.texture__loadedTexture_10_size[0] = size[0];
                m_internal.texture__loadedTexture_10_size[1] = size[1];
                m_internal.texture__loadedTexture_10_size[2] = size[2];
                m_internal.texture__loadedTexture_10_numMips = desiredNumMips;
                m_internal.texture__loadedTexture_10_format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                m_internal.texture__loadedTexture_10 = DX12Utils::CreateTexture(device, size, desiredNumMips, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, m_internal.texture__loadedTexture_10_flags, D3D12_RESOURCE_STATE_COPY_DEST, DX12Utils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_10" : nullptr), Context::LogFn);


                std::vector<unsigned char> pixels;
                for (const DX12Utils::TextureCache::Texture& texture : loadedTextureSlices)
                    pixels.insert(pixels.end(), texture.pixels.begin(), texture.pixels.end());

                DX12Utils::UploadTextureToGPUAndMakeMips(device, commandList, s_ubTracker, m_internal.texture__loadedTexture_10, pixels, size, desiredNumMips, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, LogFn);
            }
        }

        // _PresentationCB
        if (m_internal.constantBuffer__PresentationCB == nullptr)
        {
            dirty = true;
            m_internal.constantBuffer__PresentationCB = DX12Utils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_PresentationCB" : nullptr), Context::LogFn);
        }
        EnsureDrawCallPSOsCreated(device, dirty);
    }

    bool Context::EnsureDrawCallPSOsCreated(ID3D12Device* device, bool dirty)
    {
        return true;
    }
};
