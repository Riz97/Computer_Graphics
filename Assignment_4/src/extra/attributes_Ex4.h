#pragma once

#include <Eigen/Core>
#include <utility>
#include <string>

class VertexAttributes {
public:
    VertexAttributes(float x = 0, float y = 0, float z = 0, float w = 1,
                     Eigen::Vector4f n = Eigen::Vector4f(0, 0, 0, 0)) {
        position << x, y, z, w;
        normal = std::move(n);
    }

    // Interpolates the vertex attributes
    static VertexAttributes interpolate(
            const VertexAttributes &a,
            const VertexAttributes &b,
            const VertexAttributes &c,
            const float alpha,
            const float beta,
            const float gamma
    ) {
        VertexAttributes r;
        r.position = alpha*(a.position / a.position[3]) + beta*(b.position / b.position[3])
                     + gamma*(c.position / c.position[3]);
        r.normal = (alpha * a.normal + beta * b.normal + gamma * c.normal).normalized();
        r.color = alpha * a.color + beta * b.color + gamma * c.color;
        return r;
    }

    Eigen::Vector4f position;
    Eigen::Vector4f normal;
    Eigen::Vector3f color;
};

class FragmentAttributes {
public:
    FragmentAttributes(float r = 0, float g = 0, float b = 0, float a = 1, float depth = 10) {
        color << r, g, b, a;
        this->depth = depth;
    }

    Eigen::Vector4f color;
    float depth;
};

class FrameBufferAttributes {
public:
    FrameBufferAttributes(uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255, float depth = 10) {
        color << r, g, b, a;
        this->depth = depth;
    }

    Eigen::Matrix<uint8_t, 4, 1> color;
    float depth;
};

class UniformAttributes {
public:
    std::string shading_option;
    bool transformation_option = false;

    Eigen::Matrix4f view;
    Eigen::Matrix4f M_model;
    Eigen::Matrix4f M_cam;
    Eigen::Matrix4f M_projection;
    Eigen::Vector4f light_position;
    Eigen::Vector3f light_intensity;
    Eigen::Vector3f diffuse_color;
    Eigen::Vector3f specular_color;
    float specular_exponent;
    Eigen::Vector4f camera_position;
};