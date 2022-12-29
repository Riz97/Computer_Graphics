// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"


// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

void raytrace_sphere() {
	std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

	const std::string filename("sphere_orthographic.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// Single light source
	const Vector3d light_position(-1,1,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);

			// Intersect with the sphere
			// NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
			Vector2d ray_on_xy(ray_origin(0),ray_origin(1));
			const double sphere_radius = 0.9;

			if (ray_on_xy.norm() < sphere_radius) {
				// The ray hit the sphere, compute the exact intersection point
				Vector3d ray_intersection(ray_on_xy(0),ray_on_xy(1),sqrt(sphere_radius*sphere_radius - ray_on_xy.squaredNorm()));

				// Compute normal at the intersection point
				Vector3d ray_normal = ray_intersection.normalized();

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;
				 
				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);

}

bool pgram_intersections_func(const Vector3d &ray_origin,const Vector3d &ray_direction,const Vector3d pgram_origin, 
	const Vector3d pgram_u,const Vector3d pgram_v, Vector3d &result) 
{
Matrix3d A;
	     //a-b a-c d
    A << -pgram_u, -pgram_v, ray_direction;
               //(a - e)	
    Vector3d b(pgram_origin - ray_origin);
	//Ax = b , we need x
    Vector3d x = A.colPivHouseholderQr().solve(b);
	//Save the result in a variable
    result = x;
	//Conditions for having an intersection
    return (x(2) > 0 && (0 <= x(0) && x(0) <= 1) && (0 <= x(1) && x(1) <= 1)); 
}

void raytrace_parallelogram() {
	std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

	const std::string filename("plane_orthographic.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// DONE: Parameters of the parallelogram (position of the lower-left corner + two sides)
	Vector3d pgram_origin(-0.7, -0.5, 0);
	Vector3d pgram_u(0.5, 1.0, 0);
	Vector3d pgram_v(1.0, 0, 0);

	// Single light source
	const Vector3d light_position(-1,1,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);
			// DONE: Check if the ray intersects with the parallelogram
			//If true the ray hit the parallelogram
			Vector3d result;
			if (pgram_intersections_func(ray_origin,ray_direction,pgram_origin,pgram_u,pgram_v,result)) {
				// DONE: The ray hit the parallelogram, compute the exact intersection point
				Vector3d ray_intersection(pgram_origin + result(0) * pgram_u + result(1) * pgram_v);

					// DONE: Compute normal at the intersection point
				//Rule of the normal (use cross product)
				Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);
}

bool sphere_intersection(const Vector3d& ray_direction,const Vector3d& center ,const Vector3d& ray_origin,
	 double radius,double& t )
{
	double a = ray_direction.dot(ray_direction);
	double b = 2*ray_direction.dot(ray_origin - center);
	double c = (ray_origin - center).dot(ray_origin - center) - radius*radius;

	if (b * b - 4 * a * c >= 0)
	{
		t =  (- b - sqrt(b * b - 4 * a * c)) / (2 * a);
		return true;
		
	}
	else
	{
		return false;
	}
}

void sphere_perspective() {
	std::cout << "Simple ray tracer, one sphere with perspective projection" << std::endl;

	const std::string filename("sphere_perspective.png");
	MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
	MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

	// The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1, 1, 1);
	Vector3d x_displacement(2.0 / C.cols(), 0, 0);
	Vector3d y_displacement(0, -2.0 / C.rows(), 0);

	// TODO: Parameters of the sphere
	// Single light source
	const Vector3d light_position(-1, 1, 1);
	const double radius = 0.5;
	Vector3d center(0.5, 0.5, 0);


	for (unsigned i = 0; i < C.cols(); ++i) {
		for (unsigned j = 0; j < C.rows(); ++j) {
			// DONE: Prepare the ray (origin point and direction)
			Vector3d ray_origin(0, 0, 2);
			Vector3d ray_direction = (origin + double(i) * x_displacement + double(j) * y_displacement -
				ray_origin).normalized();

			double t = INT_MAX;
			// DONE: Check if the ray intersects with the sphere
			if (sphere_intersection(ray_direction, center, ray_origin, radius, t)) {

				


				// DONE: The ray hit the parallelogram, compute the exact intersection point
				Vector3d ray_intersection = ray_origin + t * ray_direction;

				// DONE: Compute normal at the intersection point
				Vector3d ray_normal = (ray_intersection - center).normalized();


				// Simple diffuse model
				C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i, j) = std::max(C(i, j), 0.);

				// Disable the alpha mask for this pixel
				A(i, j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C, C, C, A, filename);
}
void raytrace_perspective() {
	std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

	const std::string filename("plane_perspective.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
	Vector3d pgram_origin(-0.5, -0.2, 0);
	Vector3d pgram_u(0.3, 1.0, 0.3);
	Vector3d pgram_v(1.0, 0, 0.2);

	

	// Single light source
	const Vector3d light_position(-1,1,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// DONE: Prepare the ray (origin point and direction)
			// p - e (p pixel , e ray direction)
			Vector3d ray_origin(0, 0, 2);
			Vector3d ray_direction = (origin + double(i) * x_displacement + double(j) * y_displacement -
				ray_origin).normalized();
			Vector3d result;
			
			// DONE: Check if the ray intersects with the parallelogram
			if (pgram_intersections_func(ray_origin, ray_direction, pgram_origin, pgram_u, pgram_v, result)) {
				// DONE: The ray hit the parallelogram, compute the exact intersection point
				
				
				Vector3d ray_intersection = ray_origin + result(2) * ray_direction;
				
				// TODO: Compute normal at the intersection point
				Vector3d ray_normal = (pgram_v.cross(pgram_u)).normalized();
				

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);
}



void raytrace_shading(){
	std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

	const std::string filename("shading_p_100.png");
	MatrixXd R = MatrixXd::Zero(800,800);// Store the color
	MatrixXd G = MatrixXd::Zero(800, 800);
	MatrixXd B = MatrixXd::Zero(800, 800);	
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/R.cols(),0,0);
	Vector3d y_displacement(0,-2.0/R.rows(),0);

	// Single light source
	const Vector3d light_position(-1,1,1);
	double ambient = 0.1;
	MatrixXd diffuse = MatrixXd::Zero(800, 800);
	MatrixXd specular = MatrixXd::Zero(800, 800);

	for (unsigned i=0; i < R.cols(); ++i) {
		for (unsigned j=0; j < R.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);
			Vector3d center(0.5, 0.5, 0);

			// Intersect with the sphere
			// NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
			Vector2d ray_on_xy(ray_origin(0),ray_origin(1));
			const double sphere_radius = 0.9;

			if (ray_on_xy.norm() < sphere_radius) {
				// The ray hit the sphere, compute the exact intersection point
				Vector3d ray_intersection(ray_on_xy(0),ray_on_xy(1),sqrt(sphere_radius*sphere_radius - ray_on_xy.squaredNorm()));

				// Compute normal at the intersection point
				// p - c / ||p-c||
				Vector3d ray_normal = (ray_intersection - center).normalized();

				// DONE: Add shading parameter here
				diffuse(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;
				specular(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

				
				
				// Simple diffuse model
				R(i, j) = 0.1 * ambient + 0.6 * std::max(diffuse(i, j), 0.) + 0.5 * pow(std::max(specular(i, j), 0.), 100);
				G(i, j) = 0.1 * ambient + 0.6 * std::max(diffuse(i, j), 0.) + 0.5 * pow(std::max(specular(i, j), 0.), 100);
				B(i, j) = 0.1 * ambient + 0.6 * std::max(diffuse(i, j), 0.) + 0.5 * pow(std::max(specular(i, j), 0.), 100);
				

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(R,G,B,A,filename);
}

int main() {
	raytrace_sphere();
	raytrace_parallelogram();
	raytrace_perspective();
	sphere_perspective();
	raytrace_shading();

	return 0;
}






