#ifndef FINLIN_HPP
#define FINLIN_HPP

#include <CL/cl.h>

class Vec { // Vector, real components, double precision.
	int d; // Dimension
	double *data; // Components
	cl_mem clmem; // OpenCL memory object

	bool cpuDirty; // Unapplied changes have been made in general RAM
	bool gpuDirty; // Unapplied changes have been made in GPU RAM

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
	Vec hadamard(Vec multiplier); // Hadamard product
	double operator*(Vec multiplier); // Dot product

	// In-place operations
	Vec operator*=(double scalar);
	Vec operator/=(double divisor);
	Vec operator+=(Vec addend);
	Vec operator-=(Vec subtrahend);

	// Mutators
	double setComp(int index, double value);	// Sets component.
												// Returns previous value.
	// Technical methods
	Vec copy();
	bool updateMemory();	// Corrects discrepancies between CPU/GPU memory.
							// Returns true if there was any discrepancy.
};

class Mat { // Matrix, real components, double precision.
	double data; // Components, row by row
	int h; // Height
	int w; // Width
	cl_mem clmem; // OpenCL memory object

	bool cpuDirty; // Unapplied changes have been made in general RAM
	bool gpuDirty; // Unapplied changes have been made in GPU RAM

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
	Mat transpose(); // Adjoint
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
	bool updateMemory();	// Corrects discrepancies between CPU/GPU memory.
							// Returns true if there was any discrepancy.
};

#endif
