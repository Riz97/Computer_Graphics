// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <Eigen/Geometry>


// Utilities for the Assignment
#include "raster.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!

#include "stb_image_write.h"

using namespace std;


// Read a triangle mesh from an off file
//Load the same scenes used in Assignment 3
void load_off(const std::string &filename, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
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

int main() 
{

	// The Framebuffer storing the image rendered by the rasterizer
	Eigen::Matrix<FrameBufferAttributes,Eigen::Dynamic,Eigen::Dynamic> frameBuffer(1000,1000);

	// Global Constants (empty in this example)
	UniformAttributes uniform;

	// Basic rasterization program
	Program program;

	// The vertex shader is the identity
	program.VertexShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
	{
		VertexAttributes out;
		out.position = uniform.view * va.position;
		return out;
	};

	// The fragment shader uses a fixed color
	program.FragmentShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
	{
		return FragmentAttributes(1,0,0);
	};

	// The blending shader converts colors between 0 and 1 to uint8
	program.BlendingShader = [](const FragmentAttributes& fa, const FrameBufferAttributes& previous)
	{
		return FrameBufferAttributes(fa.color[0]*255,fa.color[1]*255,fa.color[2]*255,fa.color[3]*255);
	};

	// One triangle in the center of the screen
	vector<VertexAttributes> vertices;

    //Load the bunny image
	Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    load_off("../data/bunny.off", V, F);

    //take all the vertices 
    for(int i = 0 ; i < F.rows() ; i++)
    {
        vertices.push_back(VertexAttributes(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2)));
        vertices.push_back(VertexAttributes(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2)));
        vertices.push_back(VertexAttributes(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2)));
    }

    Eigen::Vector3f barycenter(V.col(0).sum() / V.rows(), V.col(1).sum() / V.rows(), V.col(2).sum() / V.rows());

    //Translate the image in the origin of the coordinates 
    Eigen::Matrix4f M_translation;
    M_translation << 
            1, 0, 0, -barycenter(0),
            0, 1, 0, -barycenter(1),
            0, 0, 1, -barycenter(2),
            0, 0, 0, 1;

    //Enlarge the bunny by 5        
    Eigen::Matrix4f M_scale;
    M_scale << 
            5, 0, 0, 0,
            0, 5, 0, 0,
            0, 0, 5, 0,
            0, 0, 0, 1;

    //Model Matrix
    Eigen::Matrix4f M_model = M_scale * M_translation;

    Eigen::Matrix4f M_orth;
    float l = -1, b = -1, n = -4;
    float r = 1, t = 1, f = -6;
    M_orth << 
            2 / (r - l), 0, 0, -(r + l) / (r - l),
            0, 2 / (t - b), 0, -(t + b) / (t - b),
            0, 0, 2 / (n - f), -(n + f) / (n - f),
            0, 0, 0, 1;

    Eigen::Vector3f E(0, 0, 5);    //camera position or the eye position
    Eigen::Vector3f G(0, 0, -1);   //gaze direction
    Eigen::Vector3f T(0, 1, 0);   //view up vector

    Eigen::Vector3f w = -G.normalized();
    Eigen::Vector3f u = T.cross(w).normalized();
    Eigen::Vector3f v = w.cross(u);

    //Camera Matrix
    Eigen::Matrix4f M_cam;
    M_cam << 
            u(0), v(0), w(0), E(0),
            u(1), v(1), w(1), E(1),
            u(2), v(2), w(2), E(2),
            0, 0, 0, 1;
    M_cam = M_cam.inverse().eval();
    uniform.view = M_orth * M_cam * M_model;


	rasterize_triangles(program,uniform,vertices,frameBuffer);

	vector<uint8_t> image;
	framebuffer_to_uint8(frameBuffer,image);
	stbi_write_png("triangle.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows()*4);
	
	return 0;
}



