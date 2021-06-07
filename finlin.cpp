#include "finlin.hpp"

const char *FinLin::SRC = R"(
__kernel void scale(__global double *vector, const double scalar) {
	int i = get_global_id(0);
	vector[i] *= scalar;
}

__kernel void add(__global double *augend, __global const double *addend) {
	int i = get_global_id(0);
	augend[i] += addend[i];
}

__kernel void hadamard(
	__global double *multiplicand,
	__global const double *multiplier
) {
	int i = get_global_id(0);
	multiplicand[i] *= multiplier[i];
}

__kernel void reduce(__global double *arr, const int newlen) {
	int i = get_global_id(0);
	arr[i] += arr[i + newlen];
}

__kernel void sigmoid(__global double *arr) {
	int i = get_global_id(0);
	arr[i] = arr[i] / (1.0 + fabs(2.0 * arr[i])) + 0.5;
}
__kernel void dsigmoid(__global double *arr) {
	int i = get_global_id(0);
	arr[i] = pown(1.0 + fabs(2.0 * arr[i]), -2);
}
)";

int FinLin::err;

cl_platform_id *FinLin::platforms;
cl_device_id *FinLin::devices;
cl_uint *FinLin::platformCount;
cl_uint *FinLin::deviceCount; int FinLin::platformID; int FinLin::deviceID;
cl_context FinLin::context;
cl_command_queue FinLin::commandQueue;
cl_program FinLin::program;

cl_kernel FinLin::reduce;
cl_kernel FinLin::scale;
cl_kernel FinLin::add;
cl_kernel FinLin::hadamard;
cl_kernel FinLin::sigmoid;
cl_kernel FinLin::dsigmoid;
cl_kernel FinLin::matMul;
cl_kernel FinLin::adjoint;

double *res;
cl_mem memRes;

void FinLin::checkErr() {
	if(err != 0) {
		switch(err) {
			case -1: fprintf(stderr, "Device not found\n"); break;
			case -2: fprintf(stderr, "Device not available\n"); break;
			case -3: fprintf(stderr, "Compiler not available\n"); break;
			case -4: fprintf(stderr, "Mem object allocation failure\n"); break;
			case -5: fprintf(stderr, "Out of resources\n"); break;
			case -6: fprintf(stderr, "Out of host memory\n"); break;
			case -7: fprintf(stderr, "Profiling info not available\n"); break;
			case -8: fprintf(stderr, "Mem copy overlap\n"); break;
			case -9: fprintf(stderr, "Image format mismatch\n"); break;
			case -10: fprintf(stderr, "Image format not supported\n"); break;
			case -12: fprintf(stderr, "Map failure\n"); break;
			case -13: fprintf(stderr, "Misaligned sub buffer offset\n"); break;
			case -14: fprintf(stderr, "Waitlist event status error\n"); break;
			case -15: fprintf(stderr, "Compile program failure\n"); break;
			case -16: fprintf(stderr, "Linker not available\n"); break;
			case -17: fprintf(stderr, "Link program failure\n"); break;
			case -18: fprintf(stderr, "Device partition failed\n"); break;
			case -19: fprintf(stderr, "Kernel arg info not available\n"); break;
			case -30: fprintf(stderr, "Invalid value\n"); break;
			case -31: fprintf(stderr, "Invalid device type\n"); break;
			case -32: fprintf(stderr, "Invalid platform\n"); break;
			case -33: fprintf(stderr, "Invalid device\n"); break;
			case -34: fprintf(stderr, "Invalid context\n"); break;
			case -35: fprintf(stderr, "Invalid queue properties\n"); break;
			case -36: fprintf(stderr, "Invalid command queue\n"); break;
			case -37: fprintf(stderr, "Invalid host ptr\n"); break;
			case -38: fprintf(stderr, "Invalid mem object\n"); break;
			case -39: fprintf(stderr, "Invalid image format\n"); break;
			case -40: fprintf(stderr, "Invalid image size\n"); break;
			case -41: fprintf(stderr, "Invalid sampler\n"); break;
			case -42: fprintf(stderr, "Invalid binary\n"); break;
			case -43: fprintf(stderr, "Invalid build options\n"); break;
			case -44: fprintf(stderr, "Invalid program\n"); break;
			case -45: fprintf(stderr, "Invalid program executable\n"); break;
			case -46: fprintf(stderr, "Invalid kernel name\n"); break;
			case -47: fprintf(stderr, "Invalid kernel definition\n"); break;
			case -48: fprintf(stderr, "Invalid kernel\n"); break;
			case -49: fprintf(stderr, "Invalid arg index\n"); break;
			case -50: fprintf(stderr, "Invalid arg value\n"); break;
			case -51: fprintf(stderr, "Invalid arg size\n"); break;
			case -52: fprintf(stderr, "Invalid kernel args\n"); break;
			case -53: fprintf(stderr, "Invalid work dimension\n"); break;
			case -54: fprintf(stderr, "Invalid work group size\n"); break;
			case -55: fprintf(stderr, "Invalid work item size\n"); break;
			case -56: fprintf(stderr, "Invalid global offset\n"); break;
			case -57: fprintf(stderr, "Invalid event wait list\n"); break;
			case -58: fprintf(stderr, "Invalid event\n"); break;
			case -59: fprintf(stderr, "Invalid operation\n"); break;
			case -60: fprintf(stderr, "Invalid GL object\n"); break;
			case -61: fprintf(stderr, "Invalid buffer size\n"); break;
			case -62: fprintf(stderr, "Invalid MIP level\n"); break;
			case -63: fprintf(stderr, "Invalid global work size\n"); break;
			case -64: fprintf(stderr, "Invalid property\n"); break;
			case -65: fprintf(stderr, "Invalid image descriptor\n"); break;
			case -66: fprintf(stderr, "Invalid compiler options\n"); break;
			case -67: fprintf(stderr, "Invalid linker options\n"); break;
			case -68: fprintf(stderr, "Invalid device partition no.\n"); break;
			case -1000: fprintf(stderr, "Invalid GL sharegroup\n"); break;
			case -1001: fprintf(stderr, "Platform not found KHR\n"); break;
			case -1002: fprintf(stderr, "Invalid D3D10 device KHR\n"); break;
			case -1003: fprintf(stderr, "Invalid D3D10 resource KHR\n"); break;
			case -1004: fprintf(stderr, "D3D10 resource taken\n"); break;
			case -1005: fprintf(stderr, "D3D10 resource not acquired\n"); break;
			case -11:
				fprintf(stderr, "Build program failure\n");
				if(err != 0) {
					fprintf(stderr, "\nCould not build program.\nOpenCL failed with exit code %d\n\n", err);
					char *errLog;
					size_t errLen;
					clGetProgramBuildInfo(program, devices[deviceID], CL_PROGRAM_BUILD_LOG, 0, NULL, &errLen);
					errLog = (char*)malloc((errLen + 1) * sizeof(char));
					clGetProgramBuildInfo(program, devices[deviceID], CL_PROGRAM_BUILD_LOG, errLen, errLog, NULL);
					errLog[errLen] = 0;
					fprintf(stderr, "\nBuild Log:\n%s\n", errLog);
				}
				break;
			default: fprintf(stderr, "Unknown OpenCL error\n");
		}
		exit(err);
	}
}

void FinLin::init(int platformID, int deviceID) {
	FinLin::platformID = platformID;
	FinLin::deviceID = deviceID;
	platforms = (cl_platform_id*)malloc((platformID + 1) * sizeof(cl_platform_id));
	devices = (cl_device_id*)malloc((deviceID + 1) * sizeof(cl_device_id));
	platformCount = (cl_uint*)malloc(sizeof(cl_uint));
	deviceCount = (cl_uint*)malloc(sizeof(cl_uint));

	err = clGetPlatformIDs(platformID + 1, platforms, platformCount);
	checkErr();

	err = clGetDeviceIDs(
		platforms[platformID],
		CL_DEVICE_TYPE_ALL,
		deviceID + 1,
		devices,
		deviceCount
	);
	checkErr();

	context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
	checkErr();

	commandQueue = clCreateCommandQueueWithProperties(
		context,
		devices[deviceID],
		NULL,
		&err
	);
	checkErr();

	program = clCreateProgramWithSource(
		context,
		1,
		(const char**)&SRC,
		0,
		&err
	);
	checkErr();

	err = clBuildProgram(program, 1, devices + deviceID, NULL, NULL, NULL);
	checkErr();

	scale = clCreateKernel(program, "scale", &err); checkErr();
	add = clCreateKernel(program, "add", &err); checkErr();
	hadamard = clCreateKernel(program, "hadamard", &err); checkErr();
	reduce = clCreateKernel(program, "reduce", &err); checkErr();
	sigmoid = clCreateKernel(program, "sigmoid", &err); checkErr();
	dsigmoid = clCreateKernel(program, "dsigmoid", &err); checkErr();
}

// General helper functions
void setArg(cl_kernel kernel, int argno, cl_mem obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(cl_mem),
		(void*)&obj
	);
	FinLin::checkErr();
}
void setArg(cl_kernel kernel, int argno, double obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(double),
		(void*)&obj
	);
	FinLin::checkErr();
}
void setArg(cl_kernel kernel, int argno, int obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(int),
		(void*)&obj
	);
	FinLin::checkErr();
}
void writeBuffer(cl_mem buffer, size_t offset, size_t cb, const void *ptr) {
	FinLin::err = clEnqueueWriteBuffer(
		FinLin::commandQueue,
		buffer,
		CL_TRUE,
		offset,
		cb,
		ptr,
		0,
		NULL,
		NULL
	);
	FinLin::checkErr();
}
void execKernel(
	cl_kernel kernel,
	int dim,
	size_t offset,
	size_t globalSize,
	size_t localSize // 0 for NULL
) {
	size_t workOffset = offset;
	size_t globalWorkSize = globalSize;
	size_t localWorkSize = localSize;
	if(localSize == 0) {
		FinLin::err = clEnqueueNDRangeKernel(
			FinLin::commandQueue,
			kernel,
			dim,
			&workOffset,
			&globalWorkSize,
			NULL,
			0,
			NULL,
			NULL
		);
	} else {
		FinLin::err = clEnqueueNDRangeKernel(
			FinLin::commandQueue,
			kernel,
			dim,
			&workOffset,
			&globalWorkSize,
			&localWorkSize,
			0,
			NULL,
			NULL
		);
	}
	FinLin::checkErr();
}
void readBuffer(cl_mem buffer, size_t offset, size_t cb, void *ptr) {
	FinLin::err = clEnqueueReadBuffer(
		FinLin::commandQueue,
		buffer,
		CL_TRUE,
		offset,
		cb,
		ptr,
		0,
		NULL,
		NULL
	);
	FinLin::checkErr();
}

/////////////////////////////////// VECTOR /////////////////////////////////////

// Helper functions
void ensureSameVecDim(int d1, int d2, const char *operation) {
	if(d1 != d2) {
		fprintf(
			stderr,
			"Dimension mismatch. Cannot %s vectors of length %d and %d.\n",
			operation,
			d1,
			d2
		);
		exit(1);
	}
}

// Technical methods
void Vec::createMem() {
	clmem = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		d * sizeof(double),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();
	dirty = true;
}
Vec Vec::copy() const {
	Vec res = Vec(d);
	memcpy(res.data, data, d * sizeof(double));
	return res;
}
bool Vec::update() {
	if(!dirty) return false;
	writeBuffer(clmem, 0, d*sizeof(double), data);
	dirty = false;
	return true;
}

// Constructors
Vec::Vec(int dimension, double* components) {
	d = dimension;
	data = components;
	createMem();
}
Vec::Vec(int dimension) {
	d = dimension;
	data = (double*)malloc(d * sizeof(double));
	memset(data, 0, d * sizeof(double));
	createMem();
}

// Statics
Vec Vec::randomUniform(int dim, double min, double max) {
	double *components = (double*)malloc(dim * sizeof(double));
	for(int i = 0; i < dim; i++) {
		components[i] = (max - min) * rand() / RAND_MAX + min;
	}
	return Vec(dim, components);
}

// Accessors
int Vec::dim() const {
	return d;
}

double Vec::comp(int index) const {
	return data[index];
}

char *Vec::string() const {
	const int MAXLEN = 12;
	char *res = (char*)malloc(MAXLEN * d * sizeof(char));
	res[0] = '<';
	char *resp = res + 1;
	for(int i = 0; i < d; i++) {
		snprintf(resp, MAXLEN, "%0.3f, ", data[i]);
		resp += strlen(resp);
	}
	resp -= strlen(", ");
	*resp = '>';
	return res;
}

// In-place operations
Vec Vec::operator*=(double scalar) {
	update();

	setArg(FinLin::scale, 0, clmem);
	setArg(FinLin::scale, 1, scalar);
	execKernel(FinLin::scale, 1, 0, d, 0);
	readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::operator/=(double scalar) {
	return *this *= 1.0/scalar;
}

Vec Vec::operator+=(Vec addend) {
	ensureSameVecDim(d, addend.d, "add");

	update();
	addend.update();

	setArg(FinLin::add, 0, clmem);
	setArg(FinLin::add, 1, addend.clmem);
	execKernel(FinLin::add, 1, 0, d, 0);
	readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::operator-=(Vec subtrahend) {
	ensureSameVecDim(d, subtrahend.d, "subtract");
	return *this += -subtrahend;
}

Vec Vec::operator%=(Vec multiplier) {
	ensureSameVecDim(d, multiplier.d, "multiply");
	update();
	multiplier.update();

	setArg(FinLin::hadamard, 0, clmem);
	setArg(FinLin::hadamard, 1, multiplier.clmem);
	execKernel(FinLin::hadamard, 1, 0, d, 0);
	readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::setSigmoid() {
	update();

	setArg(FinLin::sigmoid, 0, clmem);
	execKernel(FinLin::sigmoid, 1, 0, d, 0);
	readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}
Vec Vec::setDsigmoid() {
	update();

	setArg(FinLin::dsigmoid, 0, clmem);
	execKernel(FinLin::dsigmoid, 1, 0, d, 0);
	readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

// Binary operations
Vec Vec::operator*(double scalar) {
	Vec vector = copy();
	vector *= scalar;
	return vector;
}
Vec operator*(double scalar, Vec vector) {
	return vector * scalar;
}
Vec Vec::operator/(double divisor) {
	Vec dividend = copy();
	dividend /= divisor;
	return dividend;
}

Vec Vec::operator+(Vec addend) {
	ensureSameVecDim(d, addend.d, "add");
	Vec augend = copy();
	augend += addend;
	return augend;
}
Vec Vec::operator-(Vec subtrahend) {
	ensureSameVecDim(d, subtrahend.d, "subtract");
	return *this + -subtrahend;
}
Vec Vec::operator%(Vec multiplier) {
	ensureSameVecDim(d, multiplier.d, "multiply");
	Vec multiplicand = copy();
	multiplicand %= multiplier;
	return multiplicand;
}
double Vec::operator*(Vec multiplier) {
	if(d == 0 || multiplier.d == 0) return 0;

	Vec hdm = *this % multiplier; // hdm is for Hadamard

	int len = d; // Is cut in half until down to 1.

	setArg(FinLin::reduce, 0, hdm.clmem);

	while(len != 1) {
		// Include the last element in the event of an odd length
		if(len % 2 == 1) {
			hdm.data[0] += hdm.data[len - 1]; // Add it to the first component
			writeBuffer(hdm.clmem, 0, sizeof(double), hdm.data);
		}

		len /= 2;
		setArg(FinLin::reduce, 1, len);

		execKernel(FinLin::reduce, 1, 0, len, 0);
	}

	// Only the first element is read. It should equal the result.
	readBuffer(hdm.clmem, 0, sizeof(double), hdm.data);

	return hdm.data[0];
}

// Unary operations
Vec Vec::operator-() {
	return -1 * *this;
}

double Vec::norm() {
	return sqrt(*this * *this);
}
Vec Vec::normal() {
	return *this / norm();
}

Vec Vec::sigmoid() {
	Vec res = copy();
	res.setSigmoid();
	return res;
}
Vec Vec::dsigmoid() {
	Vec res = copy();
	res.setDsigmoid();
	return res;
}

// Mutators
double Vec::setComp(int index, double value) {
	double prev = data[index];
	data[index] = value;
	dirty = true;
	return prev;
}

/////////////////////////////////// MATRIX /////////////////////////////////////

// Helper functions
void ensureSameMatDim(int h1, int w1, int h2, int w2, const char *operation) {
	if(w1 != w2) {
		fprintf(
			stderr,
			"Dimension mismatch. Cannot %s matrices of width %d and %d.\n",
			operation,
			w1,
			w2
		);
		exit(1);
	} else if(h1 != h2) {
		fprintf(
			stderr,
			"Dimension mismatch. Cannot %s matrices of height %d and %d.\n",
			operation,
			h1,
			h2
		);
		exit(1);
	}
}
void ensureMulMatDims(int w1, int h2, const char *operation) {
	if(w1 != h2) {
		fprintf(
			stderr,
			"Dimension mismatch. "
			"Cannot %s matrix of width %d with matrix of height %d.\n",
			operation,
			w1,
			h2
		);
		exit(1);
	}
}
