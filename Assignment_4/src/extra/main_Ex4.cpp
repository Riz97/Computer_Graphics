// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Geometry>
#include <gif.h>

// Utilities for the Assignment
#include "raster.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!

#include "stb_image_write.h"

using namespace std;

// Read a triangle mesh from an off file
//Load the same scene used in Assignment 3
void load_off(const std::string &filename, Eigen::MatrixXf &V, Eigen::MatrixXi &F) {
    std::ifstream in(filename);
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    V.resize(nv, 3);
    F.resize(nf, 3);
    for (int i = 0; i < nv; ++i) {
        in >> V(i, 0) >> V(i, 1) >> V(i, 2);
    }
    for (int i = 0; i < nf; ++i) {
        int s;
        in >> s >> F(i, 0) >> F(i, 1) >> F(i, 2);
        assert(s == 3);
    }
}

int main() {

    // The Framebuffer storing the image rendered by the rasterizer
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(500, 500);

    // Global Constants (empty in this example)
    UniformAttributes uniform;

    // Basic rasterization program
    Program program;

    // The vertex shader is the identity
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        VertexAttributes out;
        out.position = uniform.M_cam * uniform.M_model * va.position;
        out.normal = (uniform.M_cam * uniform.M_model * va.normal).normalized();

        Eigen::Vector4f Li = (uniform.light_position - out.position).normalized();
        Eigen::Vector4f N = out.normal;

        // Diffuse 
        Eigen::Vector3f diffuse = uniform.diffuse_color * std::max(Li.dot(N), float(0));

        // Specular 
        Eigen::Vector4f H = ((uniform.camera_position - out.position) +
                             (uniform.light_position - out.position)).normalized();
        Eigen::Vector3f specular =
                uniform.specular_color * pow(std::max(H.dot(N), float(0)), uniform.specular_exponent);
//        Eigen::Vector3f specular = Eigen::Vector3f(0,0,0);

        // Attenuate lights according to the squared distance to the lights
        Eigen::Vector4f D = uniform.light_position - out.position;
        out.color = (diffuse + specular).cwiseProduct(uniform.light_intensity) / D.squaredNorm();

        if (uniform.shading_option == "Wireframe") {
            out.color = Eigen::Vector3f(out.color(0) - float(0.2),
                                        out.color(1) - float(0.2),
                                        out.color(2) - float(0.2));
        }
        out.color = Eigen::Vector3f(max(out.color(0), float(0)),
                                    max(out.color(1), float(0)),
                                    max(out.color(2), float(0)));

        out.color = Eigen::Vector3f(min(out.color(0), float(1)),
                                    min(out.color(1), float(1)),
                                    min(out.color(2), float(1)));

        out.position = uniform.M_projection * out.position;
        return out;
    };

    // The fragment shader uses a fixed color
    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        float depth = (uniform.camera_position - va.position).norm();
        if (uniform.shading_option == "Wireframe")
            depth -= 0.01;
        return FragmentAttributes(va.color(0), va.color(1), va.color(2), 1, depth);
    };

    // The blending shader converts colors between 0 and 1 to uint8
    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        if (fa.depth < previous.depth - 0.0001)
            return FrameBufferAttributes(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255,
                                         fa.depth);
        else return previous;
    };
    

    uniform.shading_option = "Per-Vertex"; // <---- Choose the shading desired (Wireframe , Flat , Per-Vertex)
    uniform.transformation_option = true; // <---- Choose if you wish to produce gifs (true or false) 

    //Same values of assignment 2 json
    uniform.light_position = Eigen::Vector4f(0, 0, 5, 1);
    uniform.light_intensity = Eigen::Vector3f(20, 20, 20);
    uniform.diffuse_color = Eigen::Vector3f(0.5, 0.5, 0.5);
    uniform.specular_color = Eigen::Vector3f(0.2, 0.2, 0.2);
    uniform.specular_exponent = 256.0;

    Eigen::Matrix4f M_orth;
    float l = -1, b = -1, n = -3;
    float r = 1, t = 1, f = -6;

    M_orth << 
            2 / (r - l), 0, 0, -(r + l) / (r - l),
            0, 2 / (t - b), 0, -(t + b) / (t - b),
            0, 0, 2 / (n - f), -(n + f) / (n - f),
            0, 0, 0, 1;

    //Exercise 5 
    //Complete Perspective Transformation
    Eigen::Matrix4f P;
    P << 
        n, 0, 0, 0,
        0, n, 0, 0,
        0, 0, n + f, -(f * n),
        0, 0, 1, 0;  

    uniform.M_projection = M_orth * P;

    Eigen::Vector3f E(0, 0, 5);  //camera position or the eye position
    Eigen::Vector3f G(0, 0, -1);   //gaze direction
    Eigen::Vector3f T(0, 1, 0);   //view up vector

    Eigen::Vector3f w = -G.normalized();
    Eigen::Vector3f u = T.cross(w).normalized();
    Eigen::Vector3f v = w.cross(u);

    uniform.M_cam << 
            u(0), v(0), w(0), E(0),
            u(1), v(1), w(1), E(1),
            u(2), v(2), w(2), E(2),
            0, 0, 0, 1;
    
    //Camera Matrix
    uniform.M_cam = uniform.M_cam.inverse().eval();
    uniform.camera_position << E(0), E(1), E(2), 1;
    uniform.light_position = uniform.M_cam * uniform.light_position;

    // One triangle in the center of the screen
    vector<VertexAttributes> vertices;
    vector<VertexAttributes> vertices2;
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    load_off("../data/bunny.off", V, F);
    Eigen::Vector3f barycenter(V.col(0).sum() / V.rows(), V.col(1).sum() / V.rows(), V.col(2).sum() / V.rows());

    //Translate the image in the origin of the coordinates
    Eigen::Matrix4f M_translation;
    M_translation << 1, 0, 0, -barycenter(0),
            0, 1, 0, -barycenter(1),
            0, 0, 1, -barycenter(2),
            0, 0, 0, 1;
    
    //Enlarge bunny by 5
    Eigen::Matrix4f M_scale;
    M_scale << 
            5, 0, 0, 0,
            0, 5, 0, 0,
            0, 0, 5, 0,
            0, 0, 0, 1;

            //Model Matrix
    uniform.M_model = M_scale * M_translation;

    //Wireframe shading case
    if (uniform.shading_option == "Wireframe") {
        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector3f A = V.row(F(i, 1)) - V.row(F(i, 0));
            Eigen::Vector3f B = V.row(F(i, 2)) - V.row(F(i, 0));
            Eigen::Vector3f normal = A.cross(B).normalized();
            Eigen::Vector4f N(normal(0), normal(1), normal(2), 0);

            vertices.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N);
            vertices.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N);
            vertices.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N);
            vertices.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N);
            vertices.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N);
            vertices.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N);
        }

        //Flat shading case
    } else if (uniform.shading_option == "Flat") {
        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector3f A = V.row(F(i, 1)) - V.row(F(i, 0));
            Eigen::Vector3f B = V.row(F(i, 2)) - V.row(F(i, 0));
            Eigen::Vector3f normal = A.cross(B).normalized();
            Eigen::Vector4f N(normal(0), normal(1), normal(2), 0);

            vertices.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N);
            vertices.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N);
            vertices.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N);
        }
        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector3f A = V.row(F(i, 1)) - V.row(F(i, 0));
            Eigen::Vector3f B = V.row(F(i, 2)) - V.row(F(i, 0));
            Eigen::Vector3f normal = A.cross(B).normalized();
            Eigen::Vector4f N(normal(0), normal(1), normal(2), 0);

            vertices2.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N);
            vertices2.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N);
            vertices2.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N);
            vertices2.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N);
            vertices2.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N);
            vertices2.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N);
        }
        //Per vertex shading case 
    } else if (uniform.shading_option == "Per-Vertex") {
        Eigen::MatrixXf average_normals;
        average_normals.resize(V.rows(), 4);
        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector3f A = V.row(F(i, 1)) - V.row(F(i, 0));
            Eigen::Vector3f B = V.row(F(i, 2)) - V.row(F(i, 0));
            Eigen::Vector3f normal = A.cross(B).normalized();
            Eigen::Vector4f new_row(normal(0), normal(1), normal(2), 1);

            average_normals.row(F(i, 0)) += new_row;
            average_normals.row(F(i, 1)) += new_row;
            average_normals.row(F(i, 2)) += new_row;
        }

        for (int i = 0; i < F.rows(); ++i) {
            Eigen::Vector4f N1(average_normals(F(i, 0), 0) / average_normals(F(i, 0), 3),
                               average_normals(F(i, 0), 1) / average_normals(F(i, 0), 3),
                               average_normals(F(i, 0), 2) / average_normals(F(i, 0), 3), 0);
            vertices.emplace_back(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2), 1, N1.normalized());
            Eigen::Vector4f N2(average_normals(F(i, 1), 0) / average_normals(F(i, 1), 3),
                               average_normals(F(i, 1), 1) / average_normals(F(i, 1), 3),
                               average_normals(F(i, 1), 2) / average_normals(F(i, 1), 3), 0);
            vertices.emplace_back(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2), 1, N2.normalized());
            Eigen::Vector4f N3(average_normals(F(i, 2), 0) / average_normals(F(i, 2), 3),
                               average_normals(F(i, 2), 1) / average_normals(F(i, 2), 3),
                               average_normals(F(i, 2), 2) / average_normals(F(i, 2), 3), 0);
            vertices.emplace_back(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2), 1, N3.normalized());
        }
    }
    //Exercise 3 
    //Gifs Construction
    if (uniform.transformation_option) {
        vector<uint8_t> image;
        GifWriter g{};
        GifBegin(&g, "bunny.gif", frameBuffer.rows(), frameBuffer.cols(), 25);

        for (int i = 0; i < 15; i++) {
            frameBuffer.setConstant(FrameBufferAttributes(255, 255, 255));

            if (uniform.shading_option == "Wireframe")
                rasterize_lines(program, uniform, vertices, 1, frameBuffer);
            else if (uniform.shading_option == "Flat") {
                rasterize_triangles(program, uniform, vertices, frameBuffer);
                uniform.shading_option = "Wireframe";
                rasterize_lines(program, uniform, vertices2, 1, frameBuffer);
                uniform.shading_option = "Flat";
            } else if (uniform.shading_option == "Per-Vertex")
                rasterize_triangles(program, uniform, vertices, frameBuffer);

            framebuffer_to_uint8(frameBuffer, image);
            GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), 25);

            //Rotation Matrix
            Eigen::Matrix4f M_rotateX, M_rotateY, M_rotateZ;
            M_rotateX << 
                    1, 0, 0, 0,
                    0, float(cos(M_PI / 5)), float(-sin(M_PI / 5)), 0,
                    0, float(sin(M_PI / 5)), float(cos(M_PI / 5)), 0,
                    0, 0, 0, 1;
            M_rotateY << 
                    float(cos(M_PI / 5)), 0, float(sin(M_PI / 5)), 0,
                    0, 1, 0, 0,
                    float(-sin(M_PI / 5)), 0, float(cos(M_PI / 5)), 0,
                    0, 0, 0, 1;
            M_rotateZ << 
                    float(cos(M_PI / 5)), float(-sin(M_PI / 5)), 0, 0,
                    float(sin(M_PI / 5)), float(cos(M_PI / 5)), 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

            Eigen::Matrix4f M_translate;
            M_translate << 
                    1, 0, 0, w(0) * float(0.1),
                    0, 1, 0, w(1) * float(0.1),
                    0, 0, 1, w(2) * float(0.1),
                    0, 0, 0, 1;
            Eigen::Matrix4f M_moveToOrigin;
            M_moveToOrigin << 
                    1, 0, 0, -w(0) * float(i * 0.1),
                    0, 1, 0, -w(1) * float(i * 0.1),
                    0, 0, 1, -w(2) * float(i * 0.1),
                    0, 0, 0, 1;
            Eigen::Matrix4f M_moveBack;
            M_moveBack << 
                    1, 0, 0, w(0) * float(i * 0.1),
                    0, 1, 0, w(1) * float(i * 0.1),
                    0, 0, 1, w(2) * float(i * 0.1),
                    0, 0, 0, 1;

            uniform.M_model = M_translate * M_moveBack * M_rotateY * M_moveToOrigin * uniform.M_model;
        }
        GifEnd(&g);
    } else {
        if (uniform.shading_option == "Wireframe")
            rasterize_lines(program, uniform, vertices, 1, frameBuffer);
        else if (uniform.shading_option == "Flat") {
            rasterize_triangles(program, uniform, vertices, frameBuffer);
            uniform.shading_option = "Wireframe";
            rasterize_lines(program, uniform, vertices2, 1, frameBuffer);
            uniform.shading_option = "Flat";
        } else if (uniform.shading_option == "Per-Vertex")
            rasterize_triangles(program, uniform, vertices, frameBuffer);

        vector<uint8_t> image;
        framebuffer_to_uint8(frameBuffer, image);
        stbi_write_png("bunny.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);
    }


    return 0;
}
