///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

struct Struct__PresentationCB
{
    float4 MouseState;
    float PenSize;
    uint UseImportedImage;
    float2 _padding0;
};

Texture2D<float> DrawCanvas : register(t0);
Texture2D<float> NNInput : register(t1);
Buffer<float> HiddenLayerActivations : register(t2);
Buffer<float> OutputLayerActivations : register(t3);
RWTexture2D<float4> PresentationCanvas : register(u0);
Texture2D<float> _loadedTexture_0 : register(t4);
Texture2D<float> _loadedTexture_1 : register(t5);
Texture2D<float> _loadedTexture_2 : register(t6);
Texture2D<float> _loadedTexture_3 : register(t7);
Texture2D<float> _loadedTexture_4 : register(t8);
Texture2D<float> _loadedTexture_5 : register(t9);
Texture2D<float> _loadedTexture_6 : register(t10);
Texture2D<float> _loadedTexture_7 : register(t11);
Texture2D<float> _loadedTexture_8 : register(t12);
Texture2D<float> _loadedTexture_9 : register(t13);
Texture2D<float> _loadedTexture_10 : register(t14);
ConstantBuffer<Struct__PresentationCB> _PresentationCB : register(b0);

#line 1


[numthreads(8, 8, 1)]
#line 3
void Presentation(uint3 DTid : SV_DispatchThreadID)
{
    const int c_borderSize = 3;

    const int2 c_drawPanelPos = int2(30, 30);
    const int2 c_drawPanelSize = int2(256, 256);

    const int2 c_inputPanelPos = int2(c_drawPanelPos.x + c_drawPanelSize.x + c_borderSize * 2 + 10, 30);
    const int2 c_inputPanelSize = int2(28, 28);

    const int2 c_hiddenPanelPos = int2(c_inputPanelPos.x + c_inputPanelSize.x + c_borderSize * 2 + 10, 30);
    const int2 c_hiddenPanelSize = int2(28, 840 + c_borderSize * 29);

    const int2 c_outputPanelPos = int2(c_hiddenPanelPos.x + c_hiddenPanelSize.x + c_borderSize * 2 + 10, 30);
    const int2 c_outputPanelSize = int2(28, 280 + c_borderSize * 9);

    const int2 c_outputLabelsPos = int2(c_outputPanelPos.x + c_outputPanelSize.x + c_borderSize * 2, 30);
    const int2 c_outputLabelsSize = int2(28, 280 + c_borderSize * 9);

    const int2 c_instructionsPos = int2(30, c_drawPanelPos.y + c_drawPanelSize.y + c_borderSize * 2 + 30);
    const int2 c_instructionsSize = int2(290, 85);

    const float4 c_borderColor = float4(0.8f, 0.8f, 0.0f, 1.0f);
    const float4 c_backgroundColor = float4(0.2f, 0.2f, 0.2f, 1.0f);

    const float4 c_mouseCursorColor = float4(1.0f, 1.0f, 1.0f, 0.15f);

    // Draw the draw panel
    {
        int2 relPos = int2(DTid.xy) - c_drawPanelPos;
        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_drawPanelSize.x && relPos.y < c_drawPanelSize.y)
        {
            float4 mouse = _PresentationCB.MouseState;
            float3 color;

            if (!_PresentationCB.UseImportedImage)
            {
                color = float3(0.0f, DrawCanvas[relPos], 0.0f);
            }
            else
            {
                const uint2 c_drawingCanvasSize = uint2(256, 256);
                const uint2 c_NNInputImageSize = uint2(28, 28);
                int2 srcPos = float2(relPos) * float2(c_NNInputImageSize) / float2(c_drawingCanvasSize);
                float value = NNInput[srcPos];

                color = float3(value, value, 0.0f);
            }

            if (length(mouse.xy - float2(DTid.xy)) < _PresentationCB.PenSize)
                color = lerp(color, c_mouseCursorColor.rgb, c_mouseCursorColor.aaa);

            PresentationCanvas[DTid.xy] = float4(color, 1.0f);
            return;
        }

        if (relPos.x >= -c_borderSize && relPos.y >= -c_borderSize && relPos.x < c_drawPanelSize.x + c_borderSize && relPos.y < c_drawPanelSize.y + c_borderSize)
        {
            PresentationCanvas[DTid.xy] = c_borderColor;
            return;
        }
    }

    // Draw the input layer activations (the NN input)
    {
        int2 relPos = int2(DTid.xy) - c_inputPanelPos;
        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_inputPanelSize.x && relPos.y < c_inputPanelSize.y)
        {
            float value = NNInput[relPos];
            PresentationCanvas[DTid.xy] = float4(value.xxx, 1.0f);
            return;
        }

        if (relPos.x >= -c_borderSize && relPos.y >= -c_borderSize && relPos.x < c_inputPanelSize.x + c_borderSize && relPos.y < c_inputPanelSize.y + c_borderSize)
        {
            PresentationCanvas[DTid.xy] = c_borderColor;
            return;
        }
    }

    // Draw the hidden layer activations
    {
        int2 relPos = int2(DTid.xy) - c_hiddenPanelPos;

        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_hiddenPanelSize.x && relPos.y < c_hiddenPanelSize.y)
        {
            if ((relPos.y % 31) >= 28)
            {
                PresentationCanvas[DTid.xy] = c_borderColor;
                return;
            }

            relPos.y /= 31;
            float value = HiddenLayerActivations[relPos.y];
            PresentationCanvas[DTid.xy] = float4(value.xxx, 1.0f);
            return;
        }

        if (relPos.x >= -c_borderSize && relPos.y >= -c_borderSize && relPos.x < c_hiddenPanelSize.x + c_borderSize && relPos.y < c_hiddenPanelSize.y + c_borderSize)
        {
            PresentationCanvas[DTid.xy] = c_borderColor;
            return;
        }
    }

    // Draw the output layer activations
    {
        int2 relPos = int2(DTid.xy) - c_outputPanelPos;

        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_outputPanelSize.x && relPos.y < c_outputPanelSize.y)
        {
            if ((relPos.y % 31) >= 28)
            {
                PresentationCanvas[DTid.xy] = c_borderColor;
                return;
            }

            relPos.y /= 31;
            float value = OutputLayerActivations[relPos.y];
            PresentationCanvas[DTid.xy] = float4(value.xxx, 1.0f);
            return;
        }

        if (relPos.x >= -c_borderSize && relPos.y >= -c_borderSize && relPos.x < c_outputPanelSize.x + c_borderSize && relPos.y < c_outputPanelSize.y + c_borderSize)
        {
            PresentationCanvas[DTid.xy] = c_borderColor;
            return;
        }
    }

    // Draw the output layer labels
    {
        int2 relPos = int2(DTid.xy) - c_outputLabelsPos;

        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_outputLabelsSize.x && relPos.y < c_outputLabelsSize.y)
        {
            if ((relPos.y % 31) < 28)
            {
                int index = relPos.y / 31;
                relPos.y = relPos.y % 31;

                float alpha = 0.0f;
                switch(index)
                {
                    case 0: alpha = _loadedTexture_0[relPos].r; break;
                    case 1: alpha = _loadedTexture_1[relPos].r; break;
                    case 2: alpha = _loadedTexture_2[relPos].r; break;
                    case 3: alpha = _loadedTexture_3[relPos].r; break;
                    case 4: alpha = _loadedTexture_4[relPos].r; break;
                    case 5: alpha = _loadedTexture_5[relPos].r; break;
                    case 6: alpha = _loadedTexture_6[relPos].r; break;
                    case 7: alpha = _loadedTexture_7[relPos].r; break;
                    case 8: alpha = _loadedTexture_8[relPos].r; break;
                    case 9: alpha = _loadedTexture_9[relPos].r; break;
                }

                if (alpha > 0.0f)
                {
                    float3 pixelColor = lerp(float3(0.4f, 0.0f, 0.0f), float3(1.0f, 1.0f, 0.0f), OutputLayerActivations[index]);
                    pixelColor = lerp(c_backgroundColor.rgb, pixelColor, alpha);
                    PresentationCanvas[DTid.xy] = float4(pixelColor, 1.0f);
                    return;
                }
            }
        }
    }

    // Draw the instructions
    {
        int2 relPos = int2(DTid.xy) - c_instructionsPos;

        if (relPos.x >= 0 && relPos.y >= 0 && relPos.x < c_instructionsSize.x && relPos.y < c_instructionsSize.y)
        {
            PresentationCanvas[DTid.xy] = float4(_loadedTexture_10[relPos.xy].rrr, 1.0f);
            return;
        }

        if (relPos.x >= -c_borderSize && relPos.y >= -c_borderSize && relPos.x < c_instructionsSize.x + c_borderSize && relPos.y < c_instructionsSize.y + c_borderSize)
        {
            PresentationCanvas[DTid.xy] = c_borderColor;
            return;
        }
    }

    // background color
    PresentationCanvas[DTid.xy] = c_backgroundColor;
}
