#include "shader.h"
#include <iostream>
#include <Windows.h>

bool Shader::load(const std::string& vertSrc, const std::string& fragSrc) {
    GLuint vert = compile(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compile(GL_FRAGMENT_SHADER, fragSrc);

    if (!vert || !frag) return false;

    id = glCreateProgram();
    glAttachShader(id, vert);
    glAttachShader(id, frag);
    glLinkProgram(id);

    int ok;
    glGetProgramiv(id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(id, 512, nullptr, log);
        // shaders not linked properly
        glDeleteProgram(id);
        id = 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    return id != 0;
}

void Shader::use() const {
    glUseProgram(id);
}

void Shader::setInt(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}
void Shader::setVec4(const std::string& name, float x, float y, float z, float w) const {
    glUniform4f(glGetUniformLocation(id, name.c_str()), x, y, z, w);
}

void Shader::setFloat(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}
void Shader::setVec2(const std::string& name, float x, float y)
{
    glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
}
void Shader::setBool(const std::string& name, bool value) const {
    //glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
    GLint loc = glGetUniformLocation(id, name.c_str());
    if (loc == -1) {
        OutputDebugString(L"shader not found");
        return;
    }
    glUniform1i(loc, (int)value);
}

Shader::~Shader() {
    if (id) glDeleteProgram(id);
}

GLuint Shader::compile(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(shader, 1, &c, nullptr);
    glCompileShader(shader);

    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        // shaders were not compiled correctly
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}