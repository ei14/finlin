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
	friend void setArg(cl_kernel kernel, int argno, cl_mem obj);
	friend void setArg(cl_kernel kernel, int argno, double obj);
	friend void setArg(cl_kernel kernel, int argno, int obj);
	friend void writeBuffer(
		cl_mem buffer,
		size_t offset,
		size_t cb,
		const void *ptr
	);
	friend void execKernel(
		cl_kernel kernel,
		int dim,
		size_t offset,
		size_t globalWorkSize,
		size_t localWorkSize
	);
	friend void readBuffer(
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
	static cl_kernel hadamard; // Multiply two arrays element-wise
	static cl_kernel sigmoid; // Perform fast sigmoid on each element
	static cl_kernel dsigmoid; // Perform derivative of sigmoid on each element
	static cl_kernel reduce; // Halves an even-length array, preserving sum.
	static cl_kernel matMul; // Matrix multiplication
	static cl_kernel adjoint; // Find adjoint matrix

	static void checkErr(); // Stops the program if there is an error

	static double *res; // Results from kernels
	static cl_mem memRes; // Memory object for res

	public:

	static void init(int platform, int device); // Sets up all the OpenCL stuff.
												// Must be called before
												// creating any objects.
};

class Vec { // Vector, real components, double precision, on the GPU.
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

	// Accessors
	int dim() const; // Dimension

	double comp(int index) const; // Component
	char *string() const; // As a string

	// Unary operations
	Vec operator-();
	double norm(); // Magnitude
	Vec normal(); // Unit vector

	Vec sigmoid(); // Fast sigmoid function
	Vec dsigmoid(); // Derivative of fast sigmoid function

	// Binary operations
	Vec operator*(double scalar);
	Vec operator/(double divisor);

	Vec operator+(Vec addend); // Throws error if dimensions mis-match
	Vec operator-(Vec subtrahend); // ''
	Vec operator%(Vec multiplier); // Hadamard product
	double operator*(Vec multiplier); // Dot product

	// In-place operations
	Vec operator*=(double scalar);
	Vec operator/=(double divisor);
	Vec operator+=(Vec addend);
	Vec operator-=(Vec subtrahend);
	Vec operator%=(Vec multiplier); // Hadamard product

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
	double data; // Components, row by row
	int h; // Height
	int w; // Width
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	public:

	// Statics
	static Mat randomUniform(int height, int width, double min, double max);

	// Constructors
	Mat(int size); // Identity matrix
	Mat(int height, int width); // Zero matrix
	Mat(int height, int width, double *data);

	// Accessors
	int height() const;
	int width() const;

	double comp(int r, int c) const; // Component
	char *string() const; // As a string

	// Unary operations
	bool invertible();

	double det(); // Determinant
	double trace() const;

	Mat operator-();
	Mat T(); // Transpose
	Mat adj(); // Adjoint
	Mat inv(); // Inverse. Throws error if not invertible.

	// Binary operations
	Mat operator*(double scalar);
	Mat operator/(double divisor);

	Vec operator*(Vec multiplier); // Throws error if dimensions mis-match

	Mat operator*(Mat multiplier); // ''
	Mat operator+(Mat multiplier);
	Mat operator-(Mat multiplier);

	// In-place operations
	Mat operator*=(double scalar);
	Mat operator/=(double divisor);

	Mat operator*=(Mat multiplier); // Multiply as usual
	Mat operator%=(Mat multiplicand); // Multiply in reverse order

	// Mutators
	double setComp(int r, int c, double value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Mat copy();
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Matrix operations should do this automatically.
};
double operator*(double scalar, Mat matrix);

#endif
