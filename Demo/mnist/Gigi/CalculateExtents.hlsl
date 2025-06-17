/*$(ShaderResources)*/

/*$(_compute:CalculateExtents)*/(uint3 DTid : SV_DispatchThreadID)
{
	#if 0
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
	#else
		if (DTid.x != 0 || DTid.y != 0)
			return;

		uint2 canvasDims;
		Canvas.GetDimensions(canvasDims.x, canvasDims.y);

		uint minX = DrawExtents[0].MinX;
		uint maxX = DrawExtents[0].MaxX;
		uint minY = DrawExtents[0].MinY;
		uint maxY = DrawExtents[0].MaxY;
		uint pixelcount = DrawExtents[0].PixelCount;
		uint2 PixelLocationSum = DrawExtents[0].PixelLocationSum;

		for (uint iy = 0; iy < canvasDims.y; ++iy)
		{
			for (uint ix = 0; ix < canvasDims.x; ++ix)
			{
				if (Canvas[uint2(ix,iy)] == 0.0f)
					continue;

				minX = min(minX, ix);
				maxX = max(maxX, ix);

				minY = min(minY, iy);
				maxY = max(maxY, iy);

				pixelcount++;
				PixelLocationSum.x += ix;
				PixelLocationSum.y += iy;
			}
		}

		DrawExtents[0].MinX = minX;
		DrawExtents[0].MaxX = maxX;
		DrawExtents[0].MinY = minY;
		DrawExtents[0].MaxY = maxY;
		DrawExtents[0].PixelCount = pixelcount;
		DrawExtents[0].PixelLocationSum = PixelLocationSum;
	#endif
}
