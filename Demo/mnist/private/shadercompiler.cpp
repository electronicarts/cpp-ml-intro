///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#include "shadercompiler.h"
#include "../public/technique.h"
#include <D3Dcompiler.h>
#include <string>

namespace mnist
{

namespace DXUtils
{
    static ID3DBlob* CompileShaderToByteCode_Private(
        LPCWSTR shaderDir,
        LPCWSTR shaderFile,
        const char* entryPoint,
        const char* shaderModel,
        const D3D_SHADER_MACRO* defines,
        bool debugShaders,
        mnist::TLogFn logFn)
    {
        // Compile Shaders
        ID3DBlob* error = nullptr;
        ID3DBlob* shader = nullptr;
        UINT compileFlags = debugShaders ? D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION : 0;

        // compile the shader from file
        std::wstring fullFileName = std::wstring(shaderDir) + std::wstring(shaderFile);
        HRESULT hr = D3DCompileFromFile(fullFileName.c_str(), defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, shaderModel, compileFlags, 0, &shader, &error);
        if (FAILED(hr))
        {
            if (error)
            {
                const char* errorMsg = (const char*)error->GetBufferPointer();
                logFn((int)mnist::LogLevel::Error, "Could not compile shader %ls:\n%s", shaderFile, errorMsg);
                error->Release();
                error = nullptr;
            }
            else
            {
                logFn((int)mnist::LogLevel::Error, "Could not compile shader %ls", shaderFile);
            }
            return nullptr;
        }
        else if (error)
        {
            const char* errorMsg = (const char*)error->GetBufferPointer();
            logFn((int)mnist::LogLevel::Warn, "Shader compilation warning %ls:\n%s", shaderFile, errorMsg);
            error->Release();
            error = nullptr;
        }

        return shader;
    }

    bool MakeComputePSO(
        ID3D12Device* device,
        LPCWSTR shaderDir,
        LPCWSTR shaderFile,
        const char* entryPoint,
        const char* shaderModel,
        const D3D_SHADER_MACRO* defines,
        ID3D12RootSignature* rootSig,
        ID3D12PipelineState** pso,
        bool debugShaders,
        LPCWSTR debugName,
        mnist::TLogFn logFn)
    {
        ID3DBlob* shader = CompileShaderToByteCode_Private(shaderDir, shaderFile, entryPoint, shaderModel, defines, debugShaders, logFn);
        if (!shader)
            return false;

        // Put shader bytecode into PSO
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = rootSig;
        desc.CS.pShaderBytecode = shader->GetBufferPointer();
        desc.CS.BytecodeLength = shader->GetBufferSize();

        // Make PSO
        HRESULT hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(pso));
        if (FAILED(hr))
        {
            logFn((int)mnist::LogLevel::Error, "Could not create PSO for shader %ls", shaderFile);
            return false;
        }

        if(shader) shader->Release();
        shader = nullptr;

        if (debugName)
            (*pso)->SetName(debugName);

        return true;
    }

    std::vector<unsigned char> CompileShaderToByteCode(
        LPCWSTR shaderDir,
        LPCWSTR shaderFile,
        const char* entryPoint,
        const char* shaderModel,
        const D3D_SHADER_MACRO* defines,
        bool debugShaders,
        mnist::TLogFn logFn)
    {
        std::vector<unsigned char> ret;

        ID3DBlob* shader = CompileShaderToByteCode_Private(shaderDir, shaderFile, entryPoint, shaderModel, defines, debugShaders, logFn);
        if(!shader)
            return ret;

        ret.resize(shader->GetBufferSize());
        memcpy(ret.data(), shader->GetBufferPointer(), ret.size());

        shader->Release();
        return ret;
    }
};

};
