///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <d3d12.h>
#include <vector>

namespace mnist
{
enum class LogLevel : int;
using TLogFn = void (*)(int level, const char* msg, ...);

namespace DXUtils
{
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
        mnist::TLogFn logFn);

    std::vector<unsigned char> CompileShaderToByteCode(
        LPCWSTR shaderDir,
        LPCWSTR shaderFile,
        const char* entryPoint,
        const char* shaderModel,
        const D3D_SHADER_MACRO* defines,
        bool debugShaders,
        mnist::TLogFn logFn);
};

};
