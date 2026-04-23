#pragma once

#include <glad/glad.h>
#include <string>

class Shader {
public:
    GLuint id = 0;

    bool load(const std::string& vertSrc, const std::string& fragSrc);
    void use() const;
    void setInt(const std::string& name, int value) const;
    void setBool(const std::string& name, bool value) const;
    void setVec4(const std::string& name, float x, float y, float z, float w) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, float x, float y);
    ~Shader();

private:
    static GLuint compile(GLenum type, const std::string& src);
};