// Copyright 2020 Google LLC

Texture2D textureColor : register(t0, space0);
SamplerState samplerColor : register(s0, space0);

// tonemapping
float3 Tonemap_ACES(const float3 c) {
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    // const float a = 2.51;
    // const float b = 0.03;
    // const float c = 2.43;
    // const float d = 0.59;
    // const float e = 0.14;
    // return saturate((x*(a*x+b))/(x*(c*x+d)+e));

    //ACES RRT/ODT curve fit courtesy of Stephen Hill
	float3 a = c * (c + 0.0245786) - 0.000090537;
	float3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
	return a / b;
}

float4 main([[vk::location(0)]] float2 inUV : TEXCOORD0) : SV_TARGET
{
    const float3 color = textureColor.Sample(samplerColor, inUV).xyz;
	float3 result = Tonemap_ACES(color);
    // result = pow(result, float3(0.6, 0.6, 0.6));
	return float4(result, 1.0);
}