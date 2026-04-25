#pragma once
#include <shaderc/shaderc.hpp>
#include <vector>
#include <stdexcept>
#include <string>

inline std::vector<uint32_t> compileGLSL(
    const std::string& src,
    shaderc_shader_kind kind,
    const char* name = "shader")
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan,shaderc_env_version_vulkan_1_2);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(src, kind, name, opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error(std::string("Shader compile error: ")
            + result.GetErrorMessage());

    return { result.cbegin(), result.cend() };
}

// GLSL source strings

static const std::string QUAD_VERT_GLSL = R"glsl(
#version 450

layout(push_constant) uniform PC {
    float x0, y0, x1, y1;
    uint  texSlot;
} pc;

layout(location = 0) out vec2 vUV;

void main() {
    const vec2 corners[4] = vec2[](
        vec2(pc.x0, pc.y1),
        vec2(pc.x1, pc.y1),
        vec2(pc.x1, pc.y0),
        vec2(pc.x0, pc.y0)
    );
    // UV: Vulkan clip space Y+ is down; image was already flipped in CPU.
    // (0,0)=top-left of texture maps to top-left of quad.
    const vec2 uvs[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0)
    );
    const int idx[6] = int[](0, 1, 2, 2, 3, 0);
    int i = idx[gl_VertexIndex];
    gl_Position = vec4(corners[i], 0.0, 1.0);
    vUV = uvs[i];
}
)glsl";

static const std::string QUAD_FRAG_GLSL = R"glsl(
#version 450

layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    // For HDR images the loader already normalized to linear scene referred
    // values so nothing extra is needed here.
    outColor = vec4(texture(uTex, vUV).rgb, 1.0);
}
)glsl";

static const std::string CROSSHAIR_VERT_GLSL = R"glsl(
#version 450

layout(location = 0) in vec2 aPos;

layout(push_constant) uniform PC {
    float aspect;
} pc;

void main() {
    gl_Position = vec4(aPos.x / pc.aspect, aPos.y, 0.0, 1.0);
}
)glsl";

static const std::string CROSSHAIR_FRAG_GLSL = R"glsl(
#version 450

layout(location = 0) out vec4 outColor;

void main() {
    // Pure white – renders correctly on both SDR and HDR surfaces.
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)glsl";