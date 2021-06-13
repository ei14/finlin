// Copyright (c) 2021 Thomas Kaldahl

#ifndef FINLIN_HPP
#define FINLIN_HPP

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

class FinLin {
	friend class Vec;
	friend class Mat;

	static void setArg(cl_kernel kernel, int argno, cl_mem obj);
	static void setArg(cl_kernel kernel, int argno, double obj);
	static void setArg(cl_kernel kernel, int argno, int obj);
	static void writeBuffer(
		cl_mem buffer,
		size_t offset,
		size_t cb,
		const void *ptr
	);
	static void execKernel(
		cl_kernel kernel,
		size_t offset,
		size_t globalWorkSize,
		size_t localWorkSize
	);
	static void execKernel(
		cl_kernel kernel,
		size_t offset,
		size_t globalSizeX,
		size_t globalSizeY,
		size_t localWorkSize
	);
	static void readBuffer(
		cl_mem buffer,
		size_t offset,
		size_t cb,
		void *ptr
	);

	static const char *SRC; // Kernel source code
	static int err; // Error code output

	static cl_platform_id *platforms;
	static cl_device_id *devices;
	static cl_uint *platformCount;
	static cl_uint *deviceCount;

	static int platformID;
	static int deviceID;

	static cl_context context;
	static cl_command_queue commandQueue;
	static cl_program program;

	// Kernels
	static cl_kernel scale; // Scale an array
	static cl_kernel add; // Add two arrays element-wise
	static cl_kernel addScaled; // Add a scalar multiple of an array to another
	static cl_kernel hadamard; // Multiply two arrays element-wise
	static cl_kernel sigmoid; // Perform fast sigmoid on each element
	static cl_kernel dsigmoid; // Perform derivative of sigmoid on each element
	static cl_kernel reduce; // Halves an even-length array, preserving sum.
	static cl_kernel matVec; // Matrix and vector multiplication
	static cl_kernel matMul; // Matrix multiplication

	static void checkErr(); // Stops the program if there is an error

	static double *res; // Results from kernels
	static cl_mem memRes; // Memory object for res

	public:

	static void init(int platform, int device); // Sets up all the OpenCL stuff.
												// Must be called before
												// creating any objects.
};

class Vec { // Vector, real components, double precision, on the GPU.
	friend class Mat;

	int d; // Dimension
	double *data; // Components
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Vec randomUniform(int dim, double min, double max);

	// Constructors
	Vec(int dimension); // Zero vector
	Vec(int dimension, double* components);
	Vec(int dimension, double value); // Populates all components with value

	// Accessors
	int dim() const; // Dimension

	double comp(int index) const; // Component
	char *string() const; // As a string

	// Unary operations
	Vec operator-() const;
	double norm() const; // Magnitude
	Vec normal() const; // Unit vector

	Vec sigmoid() const; // Fast sigmoid function
	Vec dsigmoid() const; // Derivative of fast sigmoid function

	// Binary operations
	Vec operator*(double scalar) const;
	Vec operator/(double divisor) const;

	Vec operator+(Vec addend) const; // Throws error if dimensions mis-match
	Vec operator-(Vec subtrahend) const; // ''
	Vec operator%(Vec multiplier) const; // Hadamard product
	double operator*(Vec multiplier) const; // Dot product

	// In-place operations
	Vec operator*=(double scalar);
	Vec operator/=(double divisor);
	Vec operator+=(Vec addend);
	Vec operator-=(Vec subtrahend);
	Vec operator%=(Vec multiplier); // Hadamard product

	Vec normalize();
	Vec setSigmoid(); // Fast sigmoid function
	Vec setDsigmoid(); // Derivative of fast sigmoid function

	// Mutators
	double setComp(int index, double value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Vec copy() const;
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Vector operations should do this automatically.
};
Vec operator*(double scalar, Vec vector);

class Mat { // Matrix, real components, double precision, on the GPU.
	double *data; // Components, row by row
	int h; // Height
	int w; // Width
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Mat randomUniform(int height, int width, double min, double max);
	static Mat fromRowVecs(int numVecs, Vec *vecs); // Throws error if
	static Mat fromColVecs(int numVecs, Vec *vecs); // dimensions don't match.

	// Constructors
	Mat(int size); // Identity matrix
	Mat(int size, double scalar); // Scalar multiple of identity matrix
	Mat(int height, int width); // Zero matrix
	Mat(int height, int width, double *data);

	// Accessors
	int height() const;
	int width() const;

	double comp(int r, int c) const; // Component
	char *string() const; // As a string

	// Unary operations
	bool invertible() const;

	double det() const; // Determinant
	double trace() const;

	Mat operator-() const;
	Mat T() const; // Transpose
	Mat cofactor() const;
	Mat adj() const; // Adjoint
	Mat inv() const; // Inverse. Throws error if not invertible.

	// Misc operations
	Vec rowVec(int row) const;
	Vec colVec(int col) const;
	double minor(int r, int c) const;

	// Binary operations
	Mat operator*(double scalar) const;
	Mat operator/(double divisor) const;

	Vec operator*(Vec multiplier); // Throws error if dimensions mis-match

	Mat operator*(Mat multiplier); // ''
	Mat operator+(Mat addend) const;
	Mat operator-(Mat subtrahend) const;

	// In-place operations
	Mat operator*=(double scalar);
	Mat operator/=(double divisor);

	Mat operator+=(Mat addend);
	Mat operator-=(Mat subtrahend);

	// Mutators
	double setComp(int r, int c, double value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Mat copy() const;
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Matrix operations should do this automatically.
};
Mat operator*(double scalar, Mat matrix);

#endif
