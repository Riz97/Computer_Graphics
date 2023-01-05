////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <stack>
#include <queue>

// Eigen for matrix operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!

#include "stb_image_write.h"
#include "utils.h"

// JSON parser library (https://github.com/nlohmann/json)
#include "json.hpp"

using json = nlohmann::json;

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

double epsilon = pow(10, -5);
int totalNum = 0;

////////////////////////////////////////////////////////////////////////////////
// Define types & classes
////////////////////////////////////////////////////////////////////////////////

struct Ray {
    Vector3d origin;
    Vector3d direction;

    Ray() {}

    Ray(Vector3d o, Vector3d d) : origin(o), direction(d) {}
};

struct Light {
    Vector3d position;
    Vector3d intensity;
};

struct Intersection {
    Vector3d position;
    Vector3d normal;
    double ray_param;
};

struct Camera {
    bool is_perspective;
    Vector3d position;
    double field_of_view; // between 0 and PI
    double focal_length;
    double lens_radius; // for depth of field
};

struct Material {
    Vector3d ambient_color;
    Vector3d diffuse_color;
    Vector3d specular_color;
    double specular_exponent; // Also called "shininess"

    Vector3d reflection_color;
    Vector3d refraction_color;
    double refraction_index;
};

struct Object {
    Material material;

    virtual ~Object() = default; // Classes with virtual methods should have a virtual destructor!
    virtual bool intersect(const Ray &ray, Intersection &hit) = 0;
};

// We use smart pointers to hold objects as this is a virtual class
typedef std::shared_ptr<Object> ObjectPtr;

struct Sphere : public Object {
    Vector3d position;
    double radius;

    virtual ~Sphere() = default;

    virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Parallelogram : public Object {
    Vector3d origin;
    Vector3d u;
    Vector3d v;

    virtual ~Parallelogram() = default;

    virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct AABBTree {
    struct Node {
        AlignedBox3d bbox;
        int parent; // Index of the parent node (-1 for root)
        int left; // Index of the left child (-1 for a leaf)
        int right; // Index of the right child (-1 for a leaf)
        int triangle; // Index of the node triangle (-1 for internal nodes)
    };

    std::vector<Node> nodes;
    int root;

    AABBTree() = default; // Default empty constructor
    AABBTree(const MatrixXd &V, MatrixXi &F, int left, int right); // Build a BVH from an existing mesh
};

struct Mesh : public Object {
    MatrixXd vertices; // n x 3 matrix (n points)
    MatrixXi facets; // m x 3 matrix (m triangles)

    AABBTree bvh;

    Mesh() = default; // Default empty constructor
    Mesh(const std::string &filename);

    virtual ~Mesh() = default;

    virtual bool intersect(const Ray &ray, Intersection &hit) override;

};

struct Scene {
    Vector3d background_color;
    Vector3d ambient_light;

    Camera camera;
    std::vector<Material> materials;
    std::vector<Light> lights;
    std::vector<ObjectPtr> objects;
};

struct Triangle_Centroid {
    Vector3i triangle;
    Vector3d centroid;
    int dim = 0;
};

struct cmp {
    bool operator()(Triangle_Centroid a, Triangle_Centroid b) {
        return a.centroid(a.dim) > b.centroid(b.dim);
    }
};

////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, MatrixXd &V, MatrixXi &F) {
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
    totalNum = nf;
}

Mesh::Mesh(const std::string &filename) {
    // Load a mesh from a file (assuming this is a .off file), and create a bvh
    load_off(filename, vertices, facets);
    bvh = AABBTree(vertices, facets, 0, facets.rows() - 1);
}

////////////////////////////////////////////////////////////////////////////////
// BVH Implementation
////////////////////////////////////////////////////////////////////////////////

// Bounding box of a triangle
AlignedBox3d bbox_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
    AlignedBox3d box;
    box.extend(a);
    box.extend(b);
    box.extend(c);
    return box;
}

AABBTree::AABBTree(const MatrixXd &V, MatrixXi &F, int left, int right) {
    
	//base case for construction of the three 
    int n = right - left + 1;
    if (n == 1) {
        Node node;
        node.parent = -1;
        node.left = -1;
        node.right = -1;
        node.triangle = left;
        node.bbox = bbox_triangle(V.row(F(node.triangle, 0)), V.row(F(node.triangle, 1)), V.row(F(node.triangle, 2)));
        this->nodes.push_back(node);
        this->root = 0;
        return;
    }

    int mid = left + n / 2;
    if (n == totalNum) {
	// Compute the centroids of all the triangles in the input mesh
        MatrixXd centroids(F.rows(), V.cols());
        centroids.setZero();
        double x_max = 0, y_max = 0, z_max = 0;
        double x_min = INFINITY, y_min = INFINITY, z_min = INFINITY;
        for (int i = 0; i < F.rows(); ++i) {
            for (int k = 0; k < F.cols(); ++k) {
                centroids.row(i) += V.row(F(i, k));
            }
            centroids.row(i) /= F.cols();

			//Comparing centroids and bounding box

            if (centroids(i, 0) > x_max) x_max = centroids(i, 0);
            if (centroids(i, 0) < x_min) x_min = centroids(i, 0);
            if (centroids(i, 1) > y_max) y_max = centroids(i, 1);
            if (centroids(i, 1) < y_min) y_min = centroids(i, 1);
            if (centroids(i, 2) > z_max) z_max = centroids(i, 2);
            if (centroids(i, 2) < z_min) z_min = centroids(i, 2);
        }

		//Splitting dimension

        int dim;
        if (x_max - x_min >= y_max - y_min && x_max - x_min >= z_max - z_min)
            dim = 0;
        else if (y_max - y_min >= x_max - x_min && y_max - y_min >= z_max - z_min)
            dim = 1;
        else if (z_max - z_min >= x_max - x_min && z_max - z_min >= y_max - y_min)
            dim = 2;

        // TODO (Assignment 3)

        // Method (1): Top-down approach.
        // Split each set of primitives into 2 sets of roughly equal size,
        // based on sorting the centroids along one direction or another.


        std::priority_queue<Triangle_Centroid, std::vector<Triangle_Centroid>, cmp> priorityQueue;
        for (int i = 0; i < centroids.rows(); i++) {
            Triangle_Centroid tc;
            tc.triangle = Vector3i(F(i, 0), F(i, 1), F(i, 2));
            tc.centroid = Vector3d(centroids(i, 0), centroids(i, 1), centroids(i, 2));
            tc.dim = dim;
            priorityQueue.push(tc);
        }

		

        for (int i = 0; i < F.rows(); i++) {
            F(i, 0) = priorityQueue.top().triangle(0);
            F(i, 1) = priorityQueue.top().triangle(1);
            F(i, 2) = priorityQueue.top().triangle(2);
            priorityQueue.pop();
        }
    }

    AABBTree bvh_left(V, F, left, mid - 1);
    AABBTree bvh_right(V, F, mid, right);

    Node node;
    node.bbox.extend(bvh_left.nodes[bvh_left.root].bbox);
    node.bbox.extend(bvh_right.nodes[bvh_right.root].bbox);
    node.parent = -1;
    node.left = bvh_left.root;
    node.right = bvh_right.root + bvh_left.nodes.size();
    node.triangle = -1;

    int offset = (int) bvh_left.nodes.size();

    for (int i = 0; i < bvh_left.nodes.size(); i++) {
        this->nodes.push_back(bvh_left.nodes[i]);
    }
    for (int i = 0; i < bvh_right.nodes.size(); i++) {
        if (bvh_right.nodes[i].left != -1) {
            bvh_right.nodes[i].left += offset;
            bvh_right.nodes[i].right += offset;
        }
        bvh_right.nodes[i].parent += offset;
        this->nodes.push_back(bvh_right.nodes[i]);
    }

    this->nodes.push_back(node);
    this->nodes[bvh_left.root].parent = this->nodes.size() - 1;
    this->nodes[bvh_right.root + offset].parent = this->nodes.size() - 1;
    this->root = this->nodes.size() - 1;

    int k = 0;

    // Method (2): Bottom-up approach.
    // Merge nodes 2 by 2, starting from the leaves of the forest, until only 1 tree is left.
}

////////////////////////////////////////////////////////////////////////////////

bool Sphere::intersect(const Ray &ray, Intersection &hit) {
    // DONE (Assignment 2)
    double A = ray.direction.dot(ray.direction);
    double B = 2 * ray.direction.dot(ray.origin - position);
    double C = (ray.origin - position).dot(ray.origin - position) - radius * radius;
    if (B * B - 4 * A * C >= 0) {
        hit.ray_param = (-B - sqrt(B * B - 4 * A * C)) / (2 * A);
        if (hit.ray_param < 0)
            hit.ray_param = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
        hit.position = ray.origin + hit.ray_param * ray.direction;
        hit.normal = (hit.position - position).normalized();
        if (hit.ray_param > epsilon) return true;
        else return false;
    } else return false;
}

bool Parallelogram::intersect(const Ray &ray, Intersection &hit) {
    // DONE (Assignment 2)
    Matrix3d A;


	//a-b a-c d
	A << -u, -v, ray.direction;
               //(a-e)
	Vector3d b(origin - ray.origin);
            //Ax = b , we need x
    Vector3d x = A.colPivHouseholderQr().solve(b);

	//Conditions for having an intersection
    if (x(2) > epsilon && (0 <= x(0) && x(0) <= 1) && (0 <= x(1) && x(1) <= 1)) {
        hit.ray_param = x(2);
        hit.position = ray.origin + hit.ray_param * ray.direction;
        hit.normal = u.cross(v).normalized();
        return true;
    } else return false;
}

// -----------------------------------------------------------------------------

bool intersect_triangle(const Ray &ray, const Vector3d &a, const Vector3d &b, const Vector3d &c, Intersection &hit) {
    // TODO (Assignment 3)
    //
    // Compute whether the ray intersects the given triangle.
    // If you have done the parallelogram case, this should be very similar to it.
    Matrix3d A;
    Vector3d u = b - a;
    Vector3d v = c - a;
    A << -u, -v, ray.direction;
    Vector3d B(a - ray.origin);
    Vector3d x = A.colPivHouseholderQr().solve(B);
    if (x(2) > epsilon && x(0) >= 0 && x(1) >= 0 && (0 <= x(0) + x(1) && x(0) + x(1) <= 1)) {
        hit.ray_param = x(2);
        hit.position = ray.origin + hit.ray_param * ray.direction;
        hit.normal = u.cross(v).normalized();
        return true;
    } else return false;
}

bool intersect_box(const Ray &ray, const AlignedBox3d &box) {
    // DONE (Assignment 3)
    //
    // Compute whether the ray intersects the given box.
    // There is no need to set the resulting normal and ray parameter, since
    // we are not testing with the real surface here anyway.

	//3D bounding box definition for having the intersection
    double x_min = box.min()(0);
    double y_min = box.min()(1);
    double z_min = box.min()(2);
    double x_max = box.max()(0);
    double y_max = box.max()(1);
    double z_max = box.max()(2);

	//Computation of the ray paramete where the ray hits lines x,y,z
	//Algorithm page 308 of the book
    double tx_max, tx_min, ty_max, ty_min, tz_max, tz_min, t_max, t_min;
    Vector3d e = ray.origin;
    Vector3d d = ray.direction;
    if (d(0) > 0) {
        tx_min = (x_min - e(0)) / d(0);
        tx_max = (x_max - e(0)) / d(0);
    } else if (d(0) < 0) {
        tx_max = (x_min - e(0)) / d(0);
        tx_min = (x_max - e(0)) / d(0);
    }
    if (d(1) > 0) {
        ty_min = (y_min - e(1)) / d(1);
        ty_max = (y_max - e(1)) / d(1);
    } else if (d(1) < 0) {
        ty_max = (y_min - e(1)) / d(1);
        ty_min = (y_max - e(1)) / d(1);
    }
    if (d(2) > 0) {
        tz_min = (z_min - e(2)) / d(2);
        tz_max = (z_max - e(2)) / d(2);
    } else if (d(2) < 0) {
        tz_max = (z_min - e(2)) / d(2);
        tz_min = (z_max - e(2)) / d(2);
    }
    t_max = std::min(tx_max, std::min(ty_max, tz_max));
    t_min = std::max(tx_min, std::max(ty_min, tz_min));
    return t_max > t_min;
}

bool Mesh::intersect(const Ray &ray, Intersection &closest_hit) {
    // DONE (Assignment 3)

    // Method (1): Traverse every triangle and return the closest hit.
//    double ray_param = INFINITY;
//    bool res = false;
//    for (int i = 0; i < facets.rows(); i++) {
//        Intersection hit;
//        Vector3d a(vertices(facets(i, 0), 0), vertices(facets(i, 0), 1), vertices(facets(i, 0), 2));
//        Vector3d b(vertices(facets(i, 1), 0), vertices(facets(i, 1), 1), vertices(facets(i, 1), 2));
//        Vector3d c(vertices(facets(i, 2), 0), vertices(facets(i, 2), 1), vertices(facets(i, 2), 2));
//        if (intersect_triangle(ray, a, b, c, hit)) {
//            res = true;
//            if (hit.ray_param < ray_param) {
//                ray_param = hit.ray_param;
//                closest_hit = hit;
//            }
//        }
//    }
//
//    return res;

    //DONE :  Method (2): Traverse the BVH tree and test the intersection with a
    // triangles at the leaf nodes that intersects the input ray.

    int triangle_index = this->bvh.nodes[this->bvh.root].triangle;

	//If the triangle index  is a leaf 
    if (triangle_index != -1) {
        Vector3d a = vertices.row(facets(triangle_index, 0));
        Vector3d b = vertices.row(facets(triangle_index, 1));
        Vector3d c = vertices.row(facets(triangle_index, 2));
		// a triangle case
        if (intersect_triangle(ray, a, b, c, closest_hit))
            return true;
		//Bounding box case
    } else if (intersect_box(ray, this->bvh.nodes[this->bvh.root].bbox)) {
        Intersection hit1, hit2;
        this->bvh.root = this->bvh.nodes[this->bvh.root].left;
        bool flag1 = intersect(ray, hit1);
        this->bvh.root = this->bvh.nodes[this->bvh.root].parent;
        this->bvh.root = this->bvh.nodes[this->bvh.root].right;
        bool flag2 = intersect(ray, hit2);
        this->bvh.root = this->bvh.nodes[this->bvh.root].parent;
        if (flag1 && flag2) {
            closest_hit = hit1.ray_param < hit2.ray_param ? hit1 : hit2;
            return true;
        } else if (flag1) {
            closest_hit = hit1;
            return true;
        } else if (flag2) {
            closest_hit = hit2;
            return true;
        } else return false;

    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////
// Define ray-tracing functions
////////////////////////////////////////////////////////////////////////////////

// Function declaration here (could be put in a header file)
Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &object, const Intersection &hit, int max_bounce);

Object *find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit);

bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light);

Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce);

// -----------------------------------------------------------------------------

Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &obj, const Intersection &hit, int max_bounce) {
    // Material for hit object
    const Material &mat = obj.material;

    // Ambient light contribution
    Vector3d ambient_color = obj.material.ambient_color.array() * scene.ambient_light.array();

    // Punctual lights contribution (direct lighting)
    Vector3d lights_color(0, 0, 0);
    for (const Light &light: scene.lights) {
        Vector3d Li = (light.position - hit.position).normalized();
        Vector3d N = hit.normal;

        // TODO (Assignment 2, shadow rays)

        // Diffuse contribution
        Vector3d diffuse = mat.diffuse_color * std::max(Li.dot(N), 0.0);

        // TODO (Assignment 2, specular contribution)
        Vector3d specular(0, 0, 0);

        // Attenuate lights according to the squared distance to the lights
        Vector3d D = light.position - hit.position;
        lights_color += (diffuse + specular).cwiseProduct(light.intensity) / D.squaredNorm();
    }

    // TODO (Assignment 2, reflected ray)
    Vector3d reflection_color(0, 0, 0);

    // TODO (Assignment 2, refracted ray)
    Vector3d refraction_color(0, 0, 0);

    // Rendering equation
    Vector3d C = ambient_color + lights_color + reflection_color + refraction_color;

    return C;
}

// -----------------------------------------------------------------------------

Object *find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit) {
    int closest_index = -1;
     // DONE:
    //
    // Find the object in the scene that intersects the ray first
    // The function must return 'nullptr' if no object is hit, otherwise it must
    // return a pointer to the hit object, and set the parameters of the argument
    // 'hit' to their expected values.
    double ray_param = INFINITY;
    for (int i = 0; i < scene.objects.size(); i++) {
        Intersection hit;
        if (scene.objects[i]->intersect(ray, hit)) {
            if (hit.ray_param < ray_param) {
                ray_param = hit.ray_param;
                closest_hit = hit;
                closest_index = i;
            }
        }
    }

    if (closest_index < 0) {
        // Return a NULL pointer
        return nullptr;
    } else {
        // Return a pointer to the hit object. Don't forget to set 'closest_hit' accordingly!
        return scene.objects[closest_index].get();
    }
}

bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light) {
    // TODO (Assignment 2, shadow ray) : Determine if the light is visible here
    for (auto &obj: scene.objects) {
        Intersection hit;
        if (obj->intersect(ray, hit) && ((hit.position - ray.origin).norm() < (light.position - ray.origin).norm()))
            return false;
    }
    return true;
}

Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce) {
    Intersection hit;
    if (Object *obj = find_nearest_object(scene, ray, hit)) {
        // 'obj' is not null and points to the object of the scene hit by the ray
        return ray_color(scene, ray, *obj, hit, max_bounce);
    } else {
        // 'obj' is null, we must return the background color
        return scene.background_color;
    }
}

////////////////////////////////////////////////////////////////////////////////

void render_scene(const Scene &scene) {
    std::cout << "Simple ray tracer." << std::endl;

    int w = 640;
    int h = 480;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);
    // DONE: Stretch the pixel grid by the proper amount here
    double scale_y = tan(scene.camera.field_of_view / 2) * scene.camera.focal_length;
    double scale_x = tan(scene.camera.field_of_view / 2) * scene.camera.focal_length * aspect_ratio; //

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    // from the sensor, and is scaled from the canonical [-1,1] in order
    // to produce the target field of view.
    Vector3d grid_origin(-scale_x, scale_y, -scene.camera.focal_length);
    Vector3d x_displacement(2.0 / w * scale_x, 0, 0);
    Vector3d y_displacement(0, -2.0 / h * scale_y, 0);

    for (unsigned i = 0; i < w; ++i) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Ray tracing: " << (100.0 * i) / w << "%\r" << std::flush;
        for (unsigned j = 0; j < h; ++j) {
            // DONE (Assignment 2, depth of field)
            Vector3d shift = grid_origin + (i + 0.5) * x_displacement + (j + 0.5) * y_displacement;

            // Prepare the ray
            Ray ray;

            if (scene.camera.is_perspective) {
                // Perspective camera
                // DONE (Assignment 2, perspective camera)
                ray.origin = scene.camera.position;
                ray.direction = Vector3d(shift[0], shift[1], 0) - ray.origin;
            } else {
                // Orthographic camera
                ray.origin = scene.camera.position + Vector3d(shift[0], shift[1], 0);
                ray.direction = Vector3d(0, 0, -1);
            }

            int max_bounce = 5;
            Vector3d C = shoot_ray(scene, ray, max_bounce);
            R(i, j) = C(0);
            G(i, j) = C(1);
            B(i, j) = C(2);
            A(i, j) = 1;
        }
    }

    std::cout << "Ray tracing: 100%  " << std::endl;

    // Save to png
    const std::string filename("raytrace.png");
    write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

Scene load_scene(const std::string &filename) {
    Scene scene;

    // Load json data from scene file
    json data;
    std::ifstream in(filename);
    in >> data;

    // Helper function to read a Vector3d from a json array
    auto read_vec3 = [](const json &x) {
        return Vector3d(x[0], x[1], x[2]);
    };

    // Read scene info
    scene.background_color = read_vec3(data["Scene"]["Background"]);
    scene.ambient_light = read_vec3(data["Scene"]["Ambient"]);

    // Read camera info
    scene.camera.is_perspective = data["Camera"]["IsPerspective"];
    scene.camera.position = read_vec3(data["Camera"]["Position"]);
    scene.camera.field_of_view = data["Camera"]["FieldOfView"];
    scene.camera.focal_length = data["Camera"]["FocalLength"];
    scene.camera.lens_radius = data["Camera"]["LensRadius"];

    // Read materials
    for (const auto &entry: data["Materials"]) {
        Material mat;
        mat.ambient_color = read_vec3(entry["Ambient"]);
        mat.diffuse_color = read_vec3(entry["Diffuse"]);
        mat.specular_color = read_vec3(entry["Specular"]);
        mat.reflection_color = read_vec3(entry["Mirror"]);
        mat.refraction_color = read_vec3(entry["Refraction"]);
        mat.refraction_index = entry["RefractionIndex"];
        mat.specular_exponent = entry["Shininess"];
        scene.materials.push_back(mat);
    }

    // Read lights
    for (const auto &entry: data["Lights"]) {
        Light light;
        light.position = read_vec3(entry["Position"]);
        light.intensity = read_vec3(entry["Color"]);
        scene.lights.push_back(light);
    }

    // Read objects
    for (const auto &entry: data["Objects"]) {
        ObjectPtr object;
        if (entry["Type"] == "Sphere") {
            auto sphere = std::make_shared<Sphere>();
            sphere->position = read_vec3(entry["Position"]);
            sphere->radius = entry["Radius"];
            object = sphere;
        } else if (entry["Type"] == "Parallelogram") {
            // TODO
            auto parallelogram = std::make_shared<Parallelogram>();
            parallelogram->origin = read_vec3(entry["Origin"]);
            parallelogram->u = read_vec3(entry["U"]);
            parallelogram->v = read_vec3(entry["V"]);
            object = parallelogram;
        } else if (entry["Type"] == "Mesh") {
            // Load mesh from a file
            std::string filename = std::string(DATA_DIR) + entry["Path"].get<std::string>();
            object = std::make_shared<Mesh>(filename);
        }
        object->material = scene.materials[entry["Material"]];
        scene.objects.push_back(object);
    }

    return scene;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " scene.json" << std::endl;
        return 1;
    }
    Scene scene = load_scene(argv[1]);
    render_scene(scene);
    return 0;
}