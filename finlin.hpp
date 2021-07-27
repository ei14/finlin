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

	friend class Veci;
	friend class Mati;

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
	static cl_kernel compNot; // Replace zeros with ones, non-zeros with zeros.

	// Integer kernels
	static cl_kernel scalei; // Scale an array
	static cl_kernel dividei; // Divide an array by a scalar
	static cl_kernel modulo; // Perform modulo on integer array
	static cl_kernel addi; // Add two arrays element-wise
	static cl_kernel addScaledi; // Add a scalar multiple of an array to another
	static cl_kernel hadamardi; // Multiply two arrays element-wise
	static cl_kernel reducei; // Halves an even-length array, preserving sum.
	static cl_kernel matVeci; // Matrix and vector multiplication
	static cl_kernel matMuli; // Matrix multiplication
	static cl_kernel compNoti; // Replace zeros with ones, non-zeros with zeros.

	static void checkErr(); // Stops the program if there is an error

	static double *res; // Results from kernels
	static cl_mem memRes; // Memory object for res

	public:

	static void init(int platform, int device); // Sets up all the OpenCL stuff.
												// Must be called before
												// creating any objects.
};

class Veci;
class Vec { // Vector, real components, double precision, on the GPU.
	friend class Mat;
	friend class Veci;

	int d; // Dimension
	double *data; // Components
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Vec randomUniform(int dim, double min, double max);
	static Vec *gramSchmidt(int numVecs, Vec *vecs); // Mutates input.

	// Constructors
	Vec(int dimension); // Zero vector
	Vec(int dimension, double* components);
	Vec(int dimension, double value); // Populates all components with value
	Vec(Veci vec); // Convert integers to doubles

	// Accessors
	int dim() const; // Dimension

	double comp(int index) const; // Component
	char *string() const; // As a string

	// Unary operations
	Vec operator-() const;
	double norm() const; // Magnitude
	Vec normal() const; // Unit vector

	double sum() const; // Sum of components
	Vec operator~() const; // Replace zeros with ones, non-zeros with zeros.

	Vec sigmoid() const; // Fast sigmoid function
	Vec dsigmoid() const; // Derivative of fast sigmoid function

	// Binary operations
	Vec operator*(double scalar) const;
	Vec operator/(double divisor) const;
	double operator^(double exponent) const; // Magnitude raised to power

	Vec operator+(Vec addend) const; // Throws error if dimensions mis-match
	Vec operator-(Vec subtrahend) const; // ''
	Vec operator&(Vec multiplier) const; // Hadamard product
	double operator*(Vec multiplier) const; // Dot product

	// In-place operations
	Vec operator*=(double scalar);
	Vec operator/=(double divisor);
	Vec operator+=(Vec addend);
	Vec operator-=(Vec subtrahend);
	Vec operator&=(Vec multiplier); // Hadamard product

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

class Mati;
class Mat { // Matrix, real components, double precision, on the GPU.
	friend class Mati;

	double *data; // Components, row by row
	int h; // Height
	int w; // Width
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Mat randomUniform(int height, int width, double min, double max);
	static Mat fromRowVec(Vec row);
	static Mat fromColVec(Vec col);
	static Mat fromRowVecs(int numVecs, Vec *vecs); // Throws error if
	static Mat fromColVecs(int numVecs, Vec *vecs); // dimensions don't match.

	// Constructors
	Mat(int size); // Identity matrix
	Mat(int size, double scalar); // Scalar multiple of identity matrix
	Mat(int height, int width); // Zero matrix
	Mat(int height, int width, double *data);
	Mat(Mati mat); // Convert integers to doubles

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
	Mat inv() const; // Inverse. Throws error if not invertible.

	Mat operator~() const; // Replace zeros with ones, non-zeros with zeros.

	// Misc operations
	Vec rowVec(int row) const;
	Vec colVec(int col) const;

	// Binary operations
	Mat operator*(double scalar) const;
	Mat operator/(double divisor) const;
	Mat operator^(int exponent) const; // Exponentiation

	Vec operator*(Vec multiplier); // Throws error if dimensions mis-match

	Mat operator*(Mat multiplier); // ''
	Mat operator&(Mat multiplier) const; // Hadamard product
	Mat operator+(Mat addend) const;
	Mat operator-(Mat subtrahend) const;

	// In-place operations
	Mat operator*=(double scalar);
	Mat operator/=(double divisor);
	Mat operator^=(int exponent); // Exponentiation

	Mat operator&=(Mat multiplier); // Hadamard product
	Mat operator+=(Mat addend);
	Mat operator-=(Mat subtrahend);

	Mat RREF();

	// Mutators
	double setComp(int r, int c, double value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Mat copy() const;
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Matrix operations should do this automatically.
};
Mat operator*(double scalar, Mat matrix);

class Veci { // Vector, integer components, on the GPU.
	friend class Vec;
	friend class Mati;

	int d; // Dimension
	int *data; // Components
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Veci randomUniform(int dim, int min, int max);

	// Constructors
	Veci(int dimension); // Zero vector
	Veci(int dimension, int* components);
	Veci(int dimension, int value); // Populates all components with value
	Veci(Vec vec); // Round doubles down

	// Accessors
	int dim() const; // Dimension

	int comp(int index) const; // Component
	char *string() const; // As a string

	// Unary operations
	int sum() const; // Sum of components
	Veci operator~() const; // Replace zeros with ones, non-zeros with zeros.

	Veci operator-() const;

	// Binary operations
	Veci operator*(int scalar) const;
	Veci operator/(int divisor) const; // Round down
	Veci operator%(int modulus) const; // Modulo
	int operator^(int exponent) const; // Magnitude raised to even power

	Veci operator+(Veci addend) const; // Throws error if dimensions mis-match
	Veci operator-(Veci subtrahend) const; // ''
	Veci operator&(Veci multiplier) const; // Hadamard product
	int operator*(Veci multiplier) const; // Dot product

	// In-place operations
	Veci operator*=(int scalar);
	Veci operator/=(int divisor); // Round down
	Veci operator%=(int modulus); // Modulo
	Veci operator+=(Veci addend);
	Veci operator-=(Veci subtrahend);
	Veci operator&=(Veci multiplier); // Hadamard product

	// Mutators
	int setComp(int index, int value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Veci copy() const;
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Vecitor operations should do this automatically.
};
Veci operator*(int scalar, Veci vector);

class Mati { // Matrix, integer components, on the GPU.
	int *data; // Components, row by row
	int h; // Height
	int w; // Width
	cl_mem clmem; // OpenCL memory object

	bool dirty; // Changes have been made in RAM but not in GPU RAM

	void createMem();

	public:

	// Statics
	static Mati randomUniform(int height, int width, int min, int max);
	static Mati fromRowVec(Veci row);
	static Mati fromColVec(Veci col);
	static Mati fromRowVecs(int numVecs, Veci *vecs); // Throws error if
	static Mati fromColVecs(int numVecs, Veci *vecs); // dimensions don't match.

	// Constructors
	Mati(int size); // Identity matrix
	Mati(int height, int width); // Zero matrix
	Mati(int height, int width, int *data);
	Mati(Mat mat); // Round doubles down

	// Accessors
	int height() const;
	int width() const;

	int comp(int r, int c) const; // Component
	char *string() const; // As a string

	// Unary operations
	int trace() const;

	Mati operator-() const;
	Mati T() const; // Transpose

	Mati operator~() const; // Replace zeros with ones, non-zeros with zeros.

	// Misc operations
	Veci rowVeci(int row) const;
	Veci colVeci(int col) const;

	// Binary operations
	Mati operator*(int scalar) const;
	Mati operator/(int divisor) const; // Rounds down
	Mati operator%(int modulus) const; // Modulo
	Mati operator^(int exponent) const; // Exponentiation

	Veci operator*(Veci multiplier); // Throws error if dimensions mis-match

	Mati operator*(Mati multiplier); // ''
	Mati operator&(Mati multiplier) const; // Hadamard product
	Mati operator+(Mati addend) const;
	Mati operator-(Mati subtrahend) const;

	// In-place operations
	Mati operator*=(int scalar);
	Mati operator/=(int divisor); // Rounds down
	Mati operator%=(int modulus); // Modulo
	Mati operator^=(int exponent); // Exponentiation

	Mati operator&=(Mati multiplier); // Hadamard product
	Mati operator+=(Mati addend);
	Mati operator-=(Mati subtrahend);

	// Mutators
	int setComp(int r, int c, int value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Mati copy() const;
	bool update();	// If necessary, updates the GPU memory and returns true.
					// Matirix operations should do this automatically.
};
Mati operator*(int scalar, Mati matrix);

#endif
