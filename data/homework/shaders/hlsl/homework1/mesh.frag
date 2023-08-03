// Copyright 2020 Google LLC

Texture2D textureColorMap : register(t0, space1);
SamplerState samplerColorMap : register(s0, space1);
Texture2D metallicRoughnessTextureMap : register(t1, space1);
SamplerState samplerMetallicRoughnessMap : register(s1, space1);
Texture2D normalTextureMap : register(t2, space1);
SamplerState samplerNormalMap : register(s2, space1);
Texture2D emissiveTextureMap : register(t3, space1);
SamplerState samplerEmissiveMap : register(s3, space1);
Texture2D occlusionTextureMap : register(t4, space1);
SamplerState samplerOcclusionMap : register(s4, space1);


struct UBO
{
	float4x4 projection;
	float4x4 view;
	float4 lightPos;
	float4 viewPos;
};

cbuffer ubo : register(b0) { UBO ubo; }

struct PushConsts {
	float4x4 model;
};
[[vk::push_constant]] PushConsts primitive;

struct VSOutput
{
[[vk::location(0)]] float3 Normal : NORMAL0;
[[vk::location(1)]] float3 Color : COLOR0;
[[vk::location(2)]] float2 UV : TEXCOORD0;
[[vk::location(3)]] float3 ViewVec : TEXCOORD1;
[[vk::location(4)]] float3 LightVec : TEXCOORD2;
[[vk::location(5)]] float4 Tangent : TEXCOORD3;
};

#define PI 3.1415926535897932384626433832795
#define ALBEDO(uv) textureColorMap.Sample(samplerColorMap, uv).rgb


// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
float3 F_Schlick(float cosTheta, float3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
float3 F_SchlickR(float cosTheta, float3 F0, float roughness)
{
	return F0 + (max((1.0 - roughness).xxx, F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 CalculateBRDF(float3 N, float3 L, float3 V, float metallic, float roughness)
{
	return float3(1,1,1);
}

float3 specularContribution(float2 inUV, float3 L, float3 V, float3 N, float3 F0, float metallic, float roughness)
{
	// Precalculate vectors and dot products
	float3 H = normalize (V + L);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);

	// Light color fixed
	float3 lightColor = float3(1.0, 1.0, 1.0);

	float3 color = float3(0.0, 0.0, 0.0);

	if (dotNL > 0.0) {
		// D = Normal distribution (Distribution of the microfacets)
		float D = D_GGX(dotNH, roughness);
		// G = Geometric shadowing term (Microfacets shadowing)
		float G = G_SchlicksmithGGX(dotNL, dotNV, roughness);
		// F = Fresnel factor (Reflectance depending on angle of incidence)
		float3 F = F_Schlick(dotNV, F0);
		float3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);
		float3 kD = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
		color += (kD * ALBEDO(inUV) / PI + spec) * dotNL;
	}

	return color;
}

float3 calculateNormal(VSOutput input)
{
	float3 tangentNormal = normalTextureMap.Sample(samplerNormalMap, input.UV).xyz * 2.0 - 1.0;

	float3 N = normalize(input.Normal);
	float3 T = normalize(input.Tangent.xyz);
	float3 B = normalize(cross(N, T));
	float3x3 TBN = transpose(float3x3(T, B, N));

	return normalize(mul(TBN, tangentNormal));
}

// ----------------------------------------------------------------------------
float4 main(VSOutput input) : SV_TARGET
{
	float3 N =  calculateNormal(input);
	float3 L = normalize(input.LightVec);
	float3 V = normalize(input.ViewVec);
	float3 R = reflect(-V, N);

	float metallic = metallicRoughnessTextureMap.Sample(samplerMetallicRoughnessMap, input.UV).x;
	float roughness = metallicRoughnessTextureMap.Sample(samplerMetallicRoughnessMap, input.UV).y;

	float3 F0 = float3(0.04, 0.04, 0.04);
	F0 = lerp(F0, ALBEDO(input.UV), metallic);

	// Specular contribution
	float3 Lo = float3(0.0, 0.0, 0.0);
	Lo += specularContribution(input.UV, L, V, N, F0, metallic, roughness);

	float3 diffuse = ALBEDO(input.UV);// max(dot(N, L), 0.0) * ALBEDO(input.UV);

	float3 F = F_SchlickR(max(dot(N, V), 0.0), F0, roughness);
	
	// Specular reflectance
	// float3 specular = pow(max(dot(R, V), 0.0), 16.0) * float3(0.75, 0.75, 0.75);
	float3 specular = 0;//reflection * (F * Lo.x + Lo.y);

	// Ambient part
	float3 kD = 1.0 - F;
	kD *= 1.0 - metallic;
	float3 ambient = (1 * diffuse + specular) * occlusionTextureMap.Sample(samplerOcclusionMap, input.UV).rrr;

	// Emission
	float3 emission = emissiveTextureMap.Sample(samplerEmissiveMap, input.UV).rgb;
	float3 color = ambient + Lo + emission;
	
	// Gamma correct
	// color = pow(color, float3(0.4545, 0.4545, 0.4545));

	return float4(color , 1.0);
}