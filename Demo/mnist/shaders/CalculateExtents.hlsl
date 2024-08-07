///////////////////////////////////////////////////////////////////////////////
//             Machine Learning Introduction For Game Developers             //
//         Copyright (c) 2023 Electronic Arts Inc. All rights reserved.      //
///////////////////////////////////////////////////////////////////////////////

struct Struct_DrawExtents
{
    uint MinX;
    uint MaxX;
    uint MinY;
    uint MaxY;
    uint PixelCount;
    uint2 PixelLocationSum;
};

Texture2D<float> Canvas : register(t0);
RWStructuredBuffer<Struct_DrawExtents> DrawExtents : register(u0);

#line 1


[numthreads(8, 8, 1)]
#line 3
void CalculateExtents(uint3 DTid : SV_DispatchThreadID)
{
	if (Canvas[DTid.xy] == 0.0f)
		return;

	uint dummy = 0;
	InterlockedMin(DrawExtents[0].MinX, DTid.x, dummy);
	InterlockedMax(DrawExtents[0].MaxX, DTid.x, dummy);
	InterlockedMin(DrawExtents[0].MinY, DTid.y, dummy);
	InterlockedMax(DrawExtents[0].MaxY, DTid.y, dummy);

	InterlockedAdd(DrawExtents[0].PixelCount, 1, dummy);
	InterlockedAdd(DrawExtents[0].PixelLocationSum.x, DTid.x, dummy);
	InterlockedAdd(DrawExtents[0].PixelLocationSum.y, DTid.y, dummy);
}