///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "../public/technique.h"
#include "dxutils.h"

#include <vector>
#include <chrono>

namespace mnist
{
    static std::vector<Context*> s_allContexts;

    static DXUtils::Heap s_srvHeap;
    static DXUtils::UploadBufferTracker s_ubTracker;

    TLogFn Context::LogFn = [] (int level, const char* msg, ...) {};
    TPerfEventBeginFn Context::PerfEventBeginFn = [] (const char* name, ID3D12CommandList* commandList, int index) {};
    TPerfEventEndFn Context::PerfEventEndFn = [] (ID3D12CommandList* commandList) {};
    TLoadTextureFn Context::LoadTextureFn = [] (LoadTextureData& data) { Context::LogFn((int)LogLevel::Error, "A texture needs to be loaded but no load texture callback has been given!"); return false; };

    std::wstring Context::s_techniqueLocation = L"mnist/";
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

            // _cb
            ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[2].NumDescriptors = 1;
            ranges[2].BaseShaderRegister = 0;
            ranges[2].RegisterSpace = 0;
            ranges[2].OffsetInDescriptorsFromTableStart = 2;

            if(!DXUtils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Draw_rootSig, (c_debugNames ? L"Draw" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/draw.hlsl", "Draw", "cs_5_1", defines,
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

            if(!DXUtils::MakeRootSig(device, ranges, 2, samplers, 0, &ContextInternal::computeShader_CalculateExtents_rootSig, (c_debugNames ? L"CalculateExtents" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/CalculateExtents.hlsl", "CalculateExtents", "cs_5_1", defines,
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

            // _cb
            ranges[4].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[4].NumDescriptors = 1;
            ranges[4].BaseShaderRegister = 0;
            ranges[4].RegisterSpace = 0;
            ranges[4].OffsetInDescriptorsFromTableStart = 4;

            if(!DXUtils::MakeRootSig(device, ranges, 5, samplers, 0, &ContextInternal::computeShader_Shrink_rootSig, (c_debugNames ? L"Shrink" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/shrink.hlsl", "Shrink", "cs_5_1", defines,
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

            if(!DXUtils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Hidden_Layer_rootSig, (c_debugNames ? L"Hidden_Layer" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/HiddenLayer.hlsl", "HiddenLayer", "cs_5_1", defines,
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

            if(!DXUtils::MakeRootSig(device, ranges, 3, samplers, 0, &ContextInternal::computeShader_Output_Layer_rootSig, (c_debugNames ? L"Output_Layer" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/OutputLayer.hlsl", "OutputLayer", "cs_5_1", defines,
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

            // _cb
            ranges[16].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            ranges[16].NumDescriptors = 1;
            ranges[16].BaseShaderRegister = 0;
            ranges[16].RegisterSpace = 0;
            ranges[16].OffsetInDescriptorsFromTableStart = 16;

            if(!DXUtils::MakeRootSig(device, ranges, 17, samplers, 0, &ContextInternal::computeShader_Presentation_rootSig, (c_debugNames ? L"Presentation" : nullptr), Context::LogFn))
                return false;

            D3D_SHADER_MACRO* defines = nullptr;

            if(!DXUtils::MakeComputePSO(device, Context::s_techniqueLocation.c_str(), L"shaders/Presentation.hlsl", "Presentation", "cs_5_1", defines,
               ContextInternal::computeShader_Presentation_rootSig, &ContextInternal::computeShader_Presentation_pso, c_debugShaders, (c_debugNames ? L"Presentation" : nullptr), Context::LogFn))
                return false;
        }

        // Create SRV heap
        if(c_numSRVDescriptors > 0 && !CreateHeap(s_srvHeap, device, c_numSRVDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, Context::LogFn))
            return false;

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
            ContextInternal::computeShader_Draw_pso->Release();
            ContextInternal::computeShader_Draw_pso = nullptr;
        }

        if(ContextInternal::computeShader_Draw_rootSig)
        {
            ContextInternal::computeShader_Draw_rootSig->Release();
            ContextInternal::computeShader_Draw_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_CalculateExtents_pso)
        {
            ContextInternal::computeShader_CalculateExtents_pso->Release();
            ContextInternal::computeShader_CalculateExtents_pso = nullptr;
        }

        if(ContextInternal::computeShader_CalculateExtents_rootSig)
        {
            ContextInternal::computeShader_CalculateExtents_rootSig->Release();
            ContextInternal::computeShader_CalculateExtents_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Shrink_pso)
        {
            ContextInternal::computeShader_Shrink_pso->Release();
            ContextInternal::computeShader_Shrink_pso = nullptr;
        }

        if(ContextInternal::computeShader_Shrink_rootSig)
        {
            ContextInternal::computeShader_Shrink_rootSig->Release();
            ContextInternal::computeShader_Shrink_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Hidden_Layer_pso)
        {
            ContextInternal::computeShader_Hidden_Layer_pso->Release();
            ContextInternal::computeShader_Hidden_Layer_pso = nullptr;
        }

        if(ContextInternal::computeShader_Hidden_Layer_rootSig)
        {
            ContextInternal::computeShader_Hidden_Layer_rootSig->Release();
            ContextInternal::computeShader_Hidden_Layer_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Output_Layer_pso)
        {
            ContextInternal::computeShader_Output_Layer_pso->Release();
            ContextInternal::computeShader_Output_Layer_pso = nullptr;
        }

        if(ContextInternal::computeShader_Output_Layer_rootSig)
        {
            ContextInternal::computeShader_Output_Layer_rootSig->Release();
            ContextInternal::computeShader_Output_Layer_rootSig = nullptr;
        }

        if(ContextInternal::computeShader_Presentation_pso)
        {
            ContextInternal::computeShader_Presentation_pso->Release();
            ContextInternal::computeShader_Presentation_pso = nullptr;
        }

        if(ContextInternal::computeShader_Presentation_rootSig)
        {
            ContextInternal::computeShader_Presentation_rootSig->Release();
            ContextInternal::computeShader_Presentation_rootSig = nullptr;
        }

        // Destroy SRV Heap
        DestroyHeap(s_srvHeap);

        // Destroy any upload buffers
        s_ubTracker.Release();

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
            if(!CreateShared(device))
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
        s_ubTracker.OnNewFrame(framesInFlight);
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

    ID3D12Resource* Context::CreateManagedBuffer(ID3D12Device* device, unsigned int size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, D3D12_HEAP_TYPE heapType, ID3D12GraphicsCommandList* commandList, void* initialData, LPCWSTR debugName)
    {
        ID3D12Resource* ret = DXUtils::CreateBuffer(device, size, flags, initialData ? D3D12_RESOURCE_STATE_COMMON : state, heapType, c_debugNames ? debugName : L"", Context::LogFn);
        m_internal.m_managedResources.push_back(ret);

        // Copy initial data into the resource, if it was given to us
        if (initialData)
            UploadBufferData(device, commandList, ret, state, initialData, size);

        return ret;
    }

    ID3D12Resource* Context::CreateManagedTexture2D(ID3D12Device* device, const unsigned int size[2], DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, ID3D12GraphicsCommandList* commandList, void* initialData, unsigned int initialDataRowPitch, LPCWSTR debugName)
    {
        unsigned int size_[3] =
        {
            size[0],
            size[1],
            1
        };

        ID3D12Resource* ret = DXUtils::CreateTexture(device, size_, format, flags, initialData ? D3D12_RESOURCE_STATE_GENERIC_READ : state, DXUtils::ResourceType::Texture2D, c_debugNames ? debugName : L"", Context::LogFn);
        m_internal.m_managedResources.push_back(ret);

        // Copy initial data into the resource, if it was given to us
        if (initialData)
            UploadTextureData(device, commandList, ret, state, initialData, initialDataRowPitch);

        return ret;
    }

    void Context::UploadTextureData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* texture, D3D12_RESOURCE_STATES textureState, void* data, unsigned int dataRowPitch)
    {
        // Get information about the texture
        D3D12_RESOURCE_DESC textureDesc = texture->GetDesc();
        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
        device->GetCopyableFootprints(&textureDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

        // We can upgrade this function later
        if (layout->Footprint.Depth != 1)
        {
            Context::LogFn((int)LogLevel::Error, "UploadTextureData only works for 2d textures.");
            return;
        }

        // Get the upload buffer
        DXUtils::UploadBufferTracker::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, layout->Footprint.RowPitch * textureDesc.Height, Context::LogFn, false);

        // copy cpu data to the upload buffer
        {
            void* start = nullptr;
            HRESULT hr = uploadBuffer->buffer->Map(0, nullptr, reinterpret_cast<void**>(&start));
            if (hr)
            {
                Context::LogFn((int)LogLevel::Error, "Could not map upload buffer");
                return;
            }

            for (unsigned int iy = 0; iy < textureDesc.Height; ++iy)
            {
                char* dest = &((char*)start)[iy * layout->Footprint.RowPitch];
                const char* src = &((char*)data)[iy * dataRowPitch];
                memcpy(dest, src, dataRowPitch);
            }

            uploadBuffer->buffer->Unmap(0, nullptr);
        }

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

        // Copy
        {
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

    void Context::UploadBufferData(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, ID3D12Resource* buffer, D3D12_RESOURCE_STATES bufferState, void* data, unsigned int dataSize)
    {
        // Get the upload buffer
        DXUtils::UploadBufferTracker::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, dataSize, Context::LogFn, false);

        // copy cpu data to the upload buffer
        {
            void* start = nullptr;
            HRESULT hr = uploadBuffer->buffer->Map(0, nullptr, reinterpret_cast<void**>(&start));
            if(hr)
            {
                Context::LogFn((int)LogLevel::Error, "Could not map upload buffer");
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
            m_internal.texture_Drawing_Canvas->Release();
            m_internal.texture_Drawing_Canvas = nullptr;
        }

        if(m_internal.texture_NN_Input)
        {
            m_internal.texture_NN_Input->Release();
            m_internal.texture_NN_Input = nullptr;
        }

        if(m_internal.buffer_Hidden_Layer_Activations)
        {
            m_internal.buffer_Hidden_Layer_Activations->Release();
            m_internal.buffer_Hidden_Layer_Activations = nullptr;
        }

        if(m_internal.buffer_Output_Layer_Activations)
        {
            m_internal.buffer_Output_Layer_Activations->Release();
            m_internal.buffer_Output_Layer_Activations = nullptr;
        }

        if(m_internal.buffer_Draw_Extents)
        {
            m_internal.buffer_Draw_Extents->Release();
            m_internal.buffer_Draw_Extents = nullptr;
        }

        // _DrawCB
        if (m_internal.constantBuffer__DrawCB)
        {
            m_internal.constantBuffer__DrawCB->Release();
            m_internal.constantBuffer__DrawCB = nullptr;
        }

        // _ShrinkCB
        if (m_internal.constantBuffer__ShrinkCB)
        {
            m_internal.constantBuffer__ShrinkCB->Release();
            m_internal.constantBuffer__ShrinkCB = nullptr;
        }

        if(m_internal.texture__loadedTexture_0)
        {
            m_internal.texture__loadedTexture_0->Release();
            m_internal.texture__loadedTexture_0 = nullptr;
        }

        if(m_internal.texture__loadedTexture_1)
        {
            m_internal.texture__loadedTexture_1->Release();
            m_internal.texture__loadedTexture_1 = nullptr;
        }

        if(m_internal.texture__loadedTexture_2)
        {
            m_internal.texture__loadedTexture_2->Release();
            m_internal.texture__loadedTexture_2 = nullptr;
        }

        if(m_internal.texture__loadedTexture_3)
        {
            m_internal.texture__loadedTexture_3->Release();
            m_internal.texture__loadedTexture_3 = nullptr;
        }

        if(m_internal.texture__loadedTexture_4)
        {
            m_internal.texture__loadedTexture_4->Release();
            m_internal.texture__loadedTexture_4 = nullptr;
        }

        if(m_internal.texture__loadedTexture_5)
        {
            m_internal.texture__loadedTexture_5->Release();
            m_internal.texture__loadedTexture_5 = nullptr;
        }

        if(m_internal.texture__loadedTexture_6)
        {
            m_internal.texture__loadedTexture_6->Release();
            m_internal.texture__loadedTexture_6 = nullptr;
        }

        if(m_internal.texture__loadedTexture_7)
        {
            m_internal.texture__loadedTexture_7->Release();
            m_internal.texture__loadedTexture_7 = nullptr;
        }

        if(m_internal.texture__loadedTexture_8)
        {
            m_internal.texture__loadedTexture_8->Release();
            m_internal.texture__loadedTexture_8 = nullptr;
        }

        if(m_internal.texture__loadedTexture_9)
        {
            m_internal.texture__loadedTexture_9->Release();
            m_internal.texture__loadedTexture_9 = nullptr;
        }

        if(m_internal.texture__loadedTexture_10)
        {
            m_internal.texture__loadedTexture_10->Release();
            m_internal.texture__loadedTexture_10 = nullptr;
        }

        // _PresentationCB
        if (m_internal.constantBuffer__PresentationCB)
        {
            m_internal.constantBuffer__PresentationCB->Release();
            m_internal.constantBuffer__PresentationCB = nullptr;
        }
    }

    void Execute(Context* context, ID3D12Device* device, ID3D12GraphicsCommandList* commandList)
    {
        // reset the timer index
        s_timerIndex = 0;

        Context::PerfEventBeginFn("mnist", commandList, 28);

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

                context->m_internal.m_TimestampReadbackBuffer = DXUtils::CreateBuffer(device, sizeof(uint64_t) * (6+1) * 2, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK, (c_debugNames ? L"mnist Time Stamp Query Heap" : nullptr), nullptr);
            }
            commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
        }

        // Make sure internally owned resources are created and are the right size and format
        context->EnsureResourcesCreated(device, commandList);

        // set the SRV heap
        commandList->SetDescriptorHeaps(1, &s_srvHeap.m_heap);

        // Make sure imported textures are in the correct state
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
            context->m_internal.constantBuffer__DrawCB_cpu.PenSize = context->m_input.variable_PenSize;
            context->m_internal.constantBuffer__DrawCB_cpu.MouseState = context->m_input.variable_MouseState;
            context->m_internal.constantBuffer__DrawCB_cpu.iFrame = context->m_input.variable_iFrame;
            context->m_internal.constantBuffer__DrawCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            context->m_internal.constantBuffer__DrawCB_cpu.MouseStateLastFrame = context->m_input.variable_MouseStateLastFrame;
            DXUtils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__DrawCB, context->m_internal.constantBuffer__DrawCB_cpu, Context::LogFn);
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
            Context::PerfEventBeginFn("Compute Shader: Draw", commandList, 1);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Draw_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Draw_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count },
                { context->m_internal.constantBuffer__DrawCB, DXGI_FORMAT_UNKNOWN, DXUtils::AccessType::CBV, DXUtils::ResourceType::Buffer, false, 256, 1 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_Drawing_Canvas_size[0],
                context->m_internal.texture_Drawing_Canvas_size[1],
                context->m_internal.texture_Drawing_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[1] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Draw";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
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
            Context::PerfEventBeginFn("Compute Shader: CalculateExtents", commandList, 13);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_CalculateExtents_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_CalculateExtents_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 2, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_Drawing_Canvas_size[0],
                context->m_internal.texture_Drawing_Canvas_size[1],
                context->m_internal.texture_Drawing_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[1] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "CalculateExtents";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
        }

        // Shader Constants: _ShrinkCB
        {
            context->m_internal.constantBuffer__ShrinkCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            context->m_internal.constantBuffer__ShrinkCB_cpu.NormalizeDrawing = context->m_input.variable_NormalizeDrawing;
            DXUtils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__ShrinkCB, context->m_internal.constantBuffer__ShrinkCB_cpu, Context::LogFn);
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
            Context::PerfEventBeginFn("Compute Shader: Shrink", commandList, 3);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Shrink_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Shrink_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.buffer_Draw_Extents, context->m_internal.buffer_Draw_Extents_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Draw_Extents_stride, context->m_internal.buffer_Draw_Extents_count },
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_input.texture_Imported_Image, context->m_input.texture_Imported_Image_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.constantBuffer__ShrinkCB, DXGI_FORMAT_UNKNOWN, DXUtils::AccessType::CBV, DXUtils::ResourceType::Buffer, false, 256, 1 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 5, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_internal.texture_NN_Input_size[0],
                context->m_internal.texture_NN_Input_size[1],
                context->m_internal.texture_NN_Input_size[2]
            };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[1] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Shrink";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
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
            Context::PerfEventBeginFn("Compute Shader: Hidden_Layer", commandList, 5);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Hidden_Layer_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Hidden_Layer_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_input.buffer_NN_Weights, context->m_input.buffer_NN_Weights_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_input.buffer_NN_Weights_stride, context->m_input.buffer_NN_Weights_count },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = { (unsigned int)context->m_internal.variable_c_numHiddenNeurons, 1, 1 };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 63) * 1) / 64 + 0,
                ((baseDispatchSize[1] + 0) * 1) / 1 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Hidden_Layer";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
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
            Context::PerfEventBeginFn("Compute Shader: Output_Layer", commandList, 7);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Output_Layer_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Output_Layer_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_input.buffer_NN_Weights, context->m_input.buffer_NN_Weights_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_input.buffer_NN_Weights_stride, context->m_input.buffer_NN_Weights_count },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count },
                { context->m_internal.buffer_Output_Layer_Activations, context->m_internal.buffer_Output_Layer_Activations_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Output_Layer_Activations_stride, context->m_internal.buffer_Output_Layer_Activations_count }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 3, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = { (unsigned int)context->m_internal.variable_c_numOutputNeurons, 1, 1 };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 63) * 1) / 64 + 0,
                ((baseDispatchSize[1] + 0) * 1) / 1 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Output_Layer";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
        }

        // Shader Constants: _PresentationCB
        {
            context->m_internal.constantBuffer__PresentationCB_cpu.PenSize = context->m_input.variable_PenSize;
            context->m_internal.constantBuffer__PresentationCB_cpu.MouseState = context->m_input.variable_MouseState;
            context->m_internal.constantBuffer__PresentationCB_cpu.UseImportedImage = context->m_input.variable_UseImportedImage;
            DXUtils::CopyConstantsCPUToGPU(s_ubTracker, device, commandList, context->m_internal.constantBuffer__PresentationCB, context->m_internal.constantBuffer__PresentationCB_cpu, Context::LogFn);
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
            Context::PerfEventBeginFn("Compute Shader: Presentation", commandList, 9);
            std::chrono::high_resolution_clock::time_point startPointCPU;
            if(context->m_profile)
            {
                startPointCPU = std::chrono::high_resolution_clock::now();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }

            commandList->SetComputeRootSignature(ContextInternal::computeShader_Presentation_rootSig);
            commandList->SetPipelineState(ContextInternal::computeShader_Presentation_pso);

            DXUtils::ResourceDescriptor descriptors[] = {
                { context->m_internal.texture_Drawing_Canvas, context->m_internal.texture_Drawing_Canvas_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture_NN_Input, context->m_internal.texture_NN_Input_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.buffer_Hidden_Layer_Activations, context->m_internal.buffer_Hidden_Layer_Activations_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Hidden_Layer_Activations_stride, context->m_internal.buffer_Hidden_Layer_Activations_count },
                { context->m_internal.buffer_Output_Layer_Activations, context->m_internal.buffer_Output_Layer_Activations_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Buffer, false, context->m_internal.buffer_Output_Layer_Activations_stride, context->m_internal.buffer_Output_Layer_Activations_count },
                { context->m_input.texture_Presentation_Canvas, context->m_input.texture_Presentation_Canvas_format, DXUtils::AccessType::UAV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_0, context->m_internal.texture__loadedTexture_0_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_1, context->m_internal.texture__loadedTexture_1_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_2, context->m_internal.texture__loadedTexture_2_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_3, context->m_internal.texture__loadedTexture_3_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_4, context->m_internal.texture__loadedTexture_4_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_5, context->m_internal.texture__loadedTexture_5_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_6, context->m_internal.texture__loadedTexture_6_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_7, context->m_internal.texture__loadedTexture_7_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_8, context->m_internal.texture__loadedTexture_8_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_9, context->m_internal.texture__loadedTexture_9_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.texture__loadedTexture_10, context->m_internal.texture__loadedTexture_10_format, DXUtils::AccessType::SRV, DXUtils::ResourceType::Texture2D, false, 0, 0 },
                { context->m_internal.constantBuffer__PresentationCB, DXGI_FORMAT_UNKNOWN, DXUtils::AccessType::CBV, DXUtils::ResourceType::Buffer, false, 256, 1 }
            };

            D3D12_GPU_DESCRIPTOR_HANDLE descriptorTable = GetDescriptorTable(device, s_srvHeap, descriptors, 17, Context::LogFn);
            commandList->SetComputeRootDescriptorTable(0, descriptorTable);

            unsigned int baseDispatchSize[3] = {
                context->m_input.texture_Presentation_Canvas_size[0],
                context->m_input.texture_Presentation_Canvas_size[1],
                context->m_input.texture_Presentation_Canvas_size[2]
            };

            unsigned int dispatchSize[3] = {
                ((baseDispatchSize[0] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[1] + 7) * 1) / 8 + 0,
                ((baseDispatchSize[2] + 0) * 1) / 1 + 0
            };

            commandList->Dispatch(dispatchSize[0], dispatchSize[1], dispatchSize[2]);

            if(context->m_profile)
            {
                context->m_profileData[(s_timerIndex-1)/2].m_label = "Presentation";
                context->m_profileData[(s_timerIndex-1)/2].m_cpu = (float)std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startPointCPU).count();
                commandList->EndQuery(context->m_internal.m_TimestampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, s_timerIndex++);
            }
            Context::PerfEventEndFn(commandList);
        }

        // Make sure imported textures are put back in the state they were given to us in
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

        Context::PerfEventEndFn(commandList);
    }

    void Context::EnsureResourcesCreated(ID3D12Device* device, ID3D12GraphicsCommandList* commandList)
    {

        // Drawing_Canvas
        {

            unsigned int baseSize[3] = { (unsigned int)m_internal.variable_c_drawingCanvasSize[0], (unsigned int)m_internal.variable_c_drawingCanvasSize[1], 1 };

            unsigned int desiredSize[3] = {
                ((baseSize[0] + 0) * 1) / 1 + 0,
                ((baseSize[1] + 0) * 1) / 1 + 0,
                ((baseSize[2] + 0) * 1) / 1 + 0
            };

            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R8_UNORM;

            if(!m_internal.texture_Drawing_Canvas ||
               m_internal.texture_Drawing_Canvas_size[0] != desiredSize[0] ||
               m_internal.texture_Drawing_Canvas_size[1] != desiredSize[1] ||
               m_internal.texture_Drawing_Canvas_size[2] != desiredSize[2] ||
               m_internal.texture_Drawing_Canvas_format != desiredFormat)
            {
                if(m_internal.texture_Drawing_Canvas)
                    m_internal.texture_Drawing_Canvas->Release();

                m_internal.texture_Drawing_Canvas = DXUtils::CreateTexture(device, desiredSize, desiredFormat, m_internal.texture_Drawing_Canvas_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"Drawing_Canvas" : nullptr), Context::LogFn);
                m_internal.texture_Drawing_Canvas_size[0] = desiredSize[0];
                m_internal.texture_Drawing_Canvas_size[1] = desiredSize[1];
                m_internal.texture_Drawing_Canvas_size[2] = desiredSize[2];
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

            DXGI_FORMAT desiredFormat = DXGI_FORMAT_R8_UNORM;

            if(!m_internal.texture_NN_Input ||
               m_internal.texture_NN_Input_size[0] != desiredSize[0] ||
               m_internal.texture_NN_Input_size[1] != desiredSize[1] ||
               m_internal.texture_NN_Input_size[2] != desiredSize[2] ||
               m_internal.texture_NN_Input_format != desiredFormat)
            {
                if(m_internal.texture_NN_Input)
                    m_internal.texture_NN_Input->Release();

                m_internal.texture_NN_Input = DXUtils::CreateTexture(device, desiredSize, desiredFormat, m_internal.texture_NN_Input_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"NN_Input" : nullptr), Context::LogFn);
                m_internal.texture_NN_Input_size[0] = desiredSize[0];
                m_internal.texture_NN_Input_size[1] = desiredSize[1];
                m_internal.texture_NN_Input_size[2] = desiredSize[2];
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
                if(m_internal.buffer_Hidden_Layer_Activations)
                    m_internal.buffer_Hidden_Layer_Activations->Release();

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DXUtils::SizeOfFormat(desiredFormat, Context::LogFn));

                m_internal.buffer_Hidden_Layer_Activations = DXUtils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Hidden_Layer_Activations_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Hidden_Layer_Activations" : nullptr), Context::LogFn);
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
                if(m_internal.buffer_Output_Layer_Activations)
                    m_internal.buffer_Output_Layer_Activations->Release();

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DXUtils::SizeOfFormat(desiredFormat, Context::LogFn));

                m_internal.buffer_Output_Layer_Activations = DXUtils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Output_Layer_Activations_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Output_Layer_Activations" : nullptr), Context::LogFn);
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
                if(m_internal.buffer_Draw_Extents)
                    m_internal.buffer_Draw_Extents->Release();

                unsigned int desiredSize = desiredCount * ((desiredStride > 0) ? desiredStride : DXUtils::SizeOfFormat(desiredFormat, Context::LogFn));

                m_internal.buffer_Draw_Extents = DXUtils::CreateBuffer(device, desiredSize, m_internal.c_buffer_Draw_Extents_flags, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"Draw_Extents" : nullptr), Context::LogFn);
                m_internal.buffer_Draw_Extents_count = desiredCount;
                m_internal.buffer_Draw_Extents_format = desiredFormat;
                m_internal.buffer_Draw_Extents_stride = desiredStride;
            }
        }

        // _DrawCB
        if (m_internal.constantBuffer__DrawCB == nullptr)
            m_internal.constantBuffer__DrawCB = DXUtils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_DrawCB" : nullptr), Context::LogFn);

        // _ShrinkCB
        if (m_internal.constantBuffer__ShrinkCB == nullptr)
            m_internal.constantBuffer__ShrinkCB = DXUtils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_ShrinkCB" : nullptr), Context::LogFn);

        // _loadedTexture_0
        {
            if (!m_internal.texture__loadedTexture_0)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "0.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 0.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_0_size[0] = size[0];
                m_internal.texture__loadedTexture_0_size[1] = size[1];
                m_internal.texture__loadedTexture_0_size[2] = size[2];
                m_internal.texture__loadedTexture_0 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_0_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_0" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_0->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_0;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_0;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_1
        {
            if (!m_internal.texture__loadedTexture_1)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "1.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 1.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_1_size[0] = size[0];
                m_internal.texture__loadedTexture_1_size[1] = size[1];
                m_internal.texture__loadedTexture_1_size[2] = size[2];
                m_internal.texture__loadedTexture_1 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_1_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_1" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_1->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_1;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_1;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_2
        {
            if (!m_internal.texture__loadedTexture_2)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "2.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 2.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_2_size[0] = size[0];
                m_internal.texture__loadedTexture_2_size[1] = size[1];
                m_internal.texture__loadedTexture_2_size[2] = size[2];
                m_internal.texture__loadedTexture_2 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_2_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_2" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_2->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_2;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_2;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_3
        {
            if (!m_internal.texture__loadedTexture_3)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "3.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 3.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_3_size[0] = size[0];
                m_internal.texture__loadedTexture_3_size[1] = size[1];
                m_internal.texture__loadedTexture_3_size[2] = size[2];
                m_internal.texture__loadedTexture_3 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_3_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_3" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_3->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_3;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_3;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_4
        {
            if (!m_internal.texture__loadedTexture_4)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "4.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 4.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_4_size[0] = size[0];
                m_internal.texture__loadedTexture_4_size[1] = size[1];
                m_internal.texture__loadedTexture_4_size[2] = size[2];
                m_internal.texture__loadedTexture_4 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_4_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_4" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_4->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_4;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_4;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_5
        {
            if (!m_internal.texture__loadedTexture_5)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "5.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 5.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_5_size[0] = size[0];
                m_internal.texture__loadedTexture_5_size[1] = size[1];
                m_internal.texture__loadedTexture_5_size[2] = size[2];
                m_internal.texture__loadedTexture_5 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_5_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_5" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_5->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_5;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_5;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_6
        {
            if (!m_internal.texture__loadedTexture_6)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "6.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 6.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_6_size[0] = size[0];
                m_internal.texture__loadedTexture_6_size[1] = size[1];
                m_internal.texture__loadedTexture_6_size[2] = size[2];
                m_internal.texture__loadedTexture_6 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_6_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_6" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_6->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_6;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_6;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_7
        {
            if (!m_internal.texture__loadedTexture_7)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "7.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 7.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_7_size[0] = size[0];
                m_internal.texture__loadedTexture_7_size[1] = size[1];
                m_internal.texture__loadedTexture_7_size[2] = size[2];
                m_internal.texture__loadedTexture_7 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_7_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_7" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_7->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_7;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_7;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_8
        {
            if (!m_internal.texture__loadedTexture_8)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "8.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 8.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_8_size[0] = size[0];
                m_internal.texture__loadedTexture_8_size[1] = size[1];
                m_internal.texture__loadedTexture_8_size[2] = size[2];
                m_internal.texture__loadedTexture_8 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_8_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_8" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_8->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_8;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_8;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_9
        {
            if (!m_internal.texture__loadedTexture_9)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "9.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: 9.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_9_size[0] = size[0];
                m_internal.texture__loadedTexture_9_size[1] = size[1];
                m_internal.texture__loadedTexture_9_size[2] = size[2];
                m_internal.texture__loadedTexture_9 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_9_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_9" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_9->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_9;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_9;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _loadedTexture_10
        {
            if (!m_internal.texture__loadedTexture_10)
            {
                // Load the texture
                std::vector<LoadTextureData> loadedTextureSlices;
                loadedTextureSlices.resize(1);
                LoadTextureData& loadedTexture = loadedTextureSlices[0];
                loadedTexture.fileName = "instructions.png";
                loadedTexture.numChannels = DXUtils::FormatChannelCount(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                if(!Context::LoadTextureFn(loadedTexture))
                    Context::LogFn((int)LogLevel::Error, "Could not load image: instructions.png");

                unsigned int size[3] = { (unsigned int)loadedTexture.width, (unsigned int)loadedTexture.height, 1 };

                // Create the texture
                m_internal.texture__loadedTexture_10_size[0] = size[0];
                m_internal.texture__loadedTexture_10_size[1] = size[1];
                m_internal.texture__loadedTexture_10_size[2] = size[2];
                m_internal.texture__loadedTexture_10 = DXUtils::CreateTexture(device, size, DXGI_FORMAT_R8_UNORM, m_internal.texture__loadedTexture_10_flags, D3D12_RESOURCE_STATE_COPY_DEST, DXUtils::ResourceType::Texture2D, (c_debugNames ? L"_loadedTexture_10" : nullptr), Context::LogFn);

                for (int sliceIndex = 0; sliceIndex < (int)loadedTextureSlices.size(); ++sliceIndex)
                {
                    LoadTextureData& loadedTexture = loadedTextureSlices[sliceIndex];

                    // Create an upload buffer
                    int unalignedPitch = loadedTexture.width * DXUtils::SizeOfFormat(DXGI_FORMAT_R8_UNORM, Context::LogFn);
                    int alignedPitch = ALIGN(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, unalignedPitch);
                    DXUtils::UploadBufferTracker::Buffer* uploadBuffer = s_ubTracker.GetBuffer(device, alignedPitch * loadedTexture.height, LogFn);

                    // Copy the pixels to the buffer
                    {
                        unsigned char* dest = nullptr;
                        D3D12_RANGE  readRange = { 0, 0 };
                        HRESULT hr = uploadBuffer->buffer->Map(0, &readRange, (void**)&dest);
                        if(hr)
                            LogFn((int)mnist::LogLevel::Error, "Could not map upload buffer");

                        // Handle type conversion
                        const unsigned char* srcPixels = nullptr;
                        switch(DXUtils::GetFormatChannelType(DXGI_FORMAT_R8_UNORM, LogFn))
                        {
                            case DXUtils::FormatChannelType::U8:
                            {
                                if(loadedTexture.pixelsU8.size() == 0)
                                {
                                    loadedTexture.pixelsU8.resize(loadedTexture.pixelsF32.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsF32.size(); ++pixelIndex)
                                        loadedTexture.pixelsU8[pixelIndex] = (unsigned char)max(min(loadedTexture.pixelsF32[pixelIndex] * 256.0f, 255.0f), 0.0f);
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsU8.data();
                                break;
                            }
                            case DXUtils::FormatChannelType::F32:
                            {
                                if(loadedTexture.pixelsF32.size() == 0)
                                {
                                    loadedTexture.pixelsF32.resize(loadedTexture.pixelsU8.size());
                                    for (size_t pixelIndex = 0; pixelIndex < loadedTexture.pixelsU8.size(); ++pixelIndex)
                                        loadedTexture.pixelsF32[pixelIndex] = float(loadedTexture.pixelsU8[pixelIndex]) / 255.0f;
                                }
                                srcPixels = (const unsigned char*)loadedTexture.pixelsF32.data();
                                break;
                            }
                            default: Context::LogFn((int)mnist::LogLevel::Error, "Unhandled FormatChannelType");
                        }

                        for (int y = 0; y < loadedTexture.height; ++y)
                        {
                            const unsigned char* src = &srcPixels[y * unalignedPitch];
                            memcpy(&dest[y * alignedPitch], src, unalignedPitch);
                        }
                        uploadBuffer->buffer->Unmap(0, nullptr);
                    }

                    // copy the buffer into the texture
                    {
                        D3D12_RESOURCE_DESC resourceDesc = m_internal.texture__loadedTexture_10->GetDesc();
                        std::vector<unsigned char> layoutMem(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64));
                        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)layoutMem.data();
                        device->GetCopyableFootprints(&resourceDesc, 0, 1, 0, layout, nullptr, nullptr, nullptr);

                        D3D12_TEXTURE_COPY_LOCATION src = {};
                        src.pResource = uploadBuffer->buffer;
                        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        src.PlacedFootprint = *layout;

                        D3D12_TEXTURE_COPY_LOCATION dest = {};
                        dest.pResource = m_internal.texture__loadedTexture_10;
                        dest.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        dest.SubresourceIndex = sliceIndex;

                        commandList->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
                    }
                }

                // Transition the texture to the proper state
                D3D12_RESOURCE_BARRIER barrier;
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrier.Transition.pResource = m_internal.texture__loadedTexture_10;
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(1, &barrier);
            }
        }

        // _PresentationCB
        if (m_internal.constantBuffer__PresentationCB == nullptr)
            m_internal.constantBuffer__PresentationCB = DXUtils::CreateBuffer(device, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_DEFAULT, (c_debugNames ? L"_PresentationCB" : nullptr), Context::LogFn);
    }
};
