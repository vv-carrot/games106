#version 450

layout (set = 1, binding = 0) uniform sampler2D albedoMap;
layout (set = 1, binding = 1) uniform sampler2D metallicRoughnessMap;
layout (set = 1, binding = 2) uniform sampler2D normalMap;
layout (set = 1, binding = 3) uniform sampler2D emissiveMap;
layout (set = 1, binding = 4) uniform sampler2D occlusionMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 5) in vec4 inTangent;

layout (location = 0) out vec4 outFragColor;

const float PI = 3.14159265359;

vec3 calculateNormal() {
    vec3 tangentNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;

    vec3 N = normalize(inNormal);
    vec3 T = normalize(inTangent.xyz);
    vec3 B = normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);
    return normalize(TBN * tangentNormal);
}

vec3 calculateColor() {
    return texture(albedoMap, inUV).rgb;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float aPow = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotHPow = NdotH * NdotH;

    float numerator = aPow;
    float denominator = NdotHPow * (aPow - 1.0) + 1.0;
    denominator = PI * denominator * denominator;

    return numerator / denominator;
}

float GeometrySchlickGGX(float num, float roughness) {
    float a = roughness + 1.0;
    float k = (a * a) / 8.0;

    float numerator = num;
    float denominator = num * (1.0 - k) + k;

    return numerator / denominator;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float ggx_in = GeometrySchlickGGX(NdotL, roughness);
    float ggx_out = GeometrySchlickGGX(NdotV, roughness);

    return ggx_in * ggx_out;
}

vec3 fresnelSchlick(vec3 F0, float cosTheta) {
    float term = 1.0 - cosTheta;
    term = pow(term, 5.0);

    return F0 + (1.0 - F0) * term;
}

vec3 calculateBRDF(vec3 N, vec3 L, vec3 V, float metallic, float roughness) {
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04), calculateColor(), metallic);

    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    vec3 BRDF = vec3(0.0);

    if (NdotL > 0.0) {
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(F0, NdotV);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * NdotL * NdotV + 0.001;
        BRDF = numerator / denominator;

        vec3 Ks = F;
        vec3 Kd = (vec3(1.0) - Ks) * (1.0 - metallic);

        BRDF += (Kd * pow(calculateColor(), vec3(2.2)) / PI + BRDF) * NdotL;
    }

    return BRDF;
}

void main() {
    vec3 N = calculateNormal();
    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);

    float metallic = texture(metallicRoughnessMap, inUV).r;
    float roughness = texture(metallicRoughnessMap, inUV).g;

    vec3 specular = calculateBRDF(N, L, V, metallic, roughness);
    vec3 ambient = calculateColor() * 0.4 * texture(occlusionMap, inUV).rrr;
    vec3 emission = texture(emissiveMap, inUV).rgb;

    vec3 color = specular + ambient + emission;
    outFragColor = vec4(color, 1.0);
}