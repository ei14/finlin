// Copyright (c) 2021 Thomas Kaldahl

#include "finlin.hpp"

const char *FinLin::SRC = R"(
// DOUBLE KERNELS

__kernel void scale(__global double *vector, const double scalar) {
	int i = get_global_id(0);
	vector[i] *= scalar;
}

__kernel void add(__global double *augend, __global const double *addend) {
	int i = get_global_id(0);
	augend[i] += addend[i];
}

__kernel void addScaled(
	__global double *augend,
	__global const double *addend,
	const double coeff
) {
	int i = get_global_id(0);
	augend[i] += coeff * addend[i];
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

__kernel void matVec(
	__global const double *matrix,
	__global const double *vector,
	__global double *prod,
	const int depth
) {
	int r = get_global_id(0);
	int h = get_global_size(0);

	prod[r] = 0;

	for(int i = 0; i < depth; i++) {
		prod[r] += matrix[r*depth + i] * vector[i];
	}
}

__kernel void matMul(
	__global const double *mplcnd,
	__global const double *mplier,
	__global double *prod,
	const int depth
) {
	int r = get_global_id(0);
	int c = get_global_id(1);
	int h = get_global_size(0);
	int w = get_global_size(1);

	prod[r*w + c] = 0;

	for(int i = 0; i < depth; i++) {
		prod[r*w + c] += mplcnd[r*depth + i] * mplier[c + i*w];
	}
}

// INTEGER KERNELS

__kernel void scalei(__global int *vector, const int scalar) {
	int i = get_global_id(0);
	vector[i] *= scalar;
}
__kernel void dividei(__global int *vector, const int divisor) {
	int i = get_global_id(0);
	vector[i] /= divisor;
}
__kernel void modulo(__global int *vector, const int modulus) {
	int i = get_global_id(0);
	vector[i] %= modulus;
}

__kernel void addi(__global int *augend, __global const int *addend) {
	int i = get_global_id(0);
	augend[i] += addend[i];
}

__kernel void addScaledi(
	__global int *augend,
	__global const int *addend,
	const int coeff
) {
	int i = get_global_id(0);
	augend[i] += coeff * addend[i];
}

__kernel void hadamardi(
	__global int *multiplicand,
	__global const int *multiplier
) {
	int i = get_global_id(0);
	multiplicand[i] *= multiplier[i];
}

__kernel void reducei(__global int *arr, const int newlen) {
	int i = get_global_id(0);
	arr[i] += arr[i + newlen];
}

__kernel void matVeci(
	__global const int *matrix,
	__global const int *vector,
	__global int *prod,
	const int depth
) {
	int r = get_global_id(0);
	int h = get_global_size(0);

	prod[r] = 0;

	for(int i = 0; i < depth; i++) {
		prod[r] += matrix[r*depth + i] * vector[i];
	}
}

__kernel void matMuli(
	__global const int *mplcnd,
	__global const int *mplier,
	__global int *prod,
	const int depth
) {
	int r = get_global_id(0);
	int c = get_global_id(1);
	int h = get_global_size(0);
	int w = get_global_size(1);

	prod[r*w + c] = 0;

	for(int i = 0; i < depth; i++) {
		prod[r*w + c] += mplcnd[r*depth + i] * mplier[c + i*w];
	}
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
cl_kernel FinLin::addScaled;
cl_kernel FinLin::hadamard;
cl_kernel FinLin::sigmoid;
cl_kernel FinLin::dsigmoid;
cl_kernel FinLin::matMul;
cl_kernel FinLin::matVec;

cl_kernel FinLin::reducei;
cl_kernel FinLin::scalei;
cl_kernel FinLin::dividei;
cl_kernel FinLin::modulo;
cl_kernel FinLin::addi;
cl_kernel FinLin::addScaledi;
cl_kernel FinLin::hadamardi;
cl_kernel FinLin::matMuli;
cl_kernel FinLin::matVeci;

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
					fprintf(stderr, "\n\nExit code %d\n\n", err);
					char *errLog;
					size_t errLen;
					clGetProgramBuildInfo(
						program,
						devices[deviceID],
						CL_PROGRAM_BUILD_LOG,
						0,
						NULL,
						&errLen
					);
					errLog = (char*)malloc((errLen + 1) * sizeof(char));
					clGetProgramBuildInfo(
						program,
						devices[deviceID],
						CL_PROGRAM_BUILD_LOG,
						errLen,
						errLog,
						NULL
					);
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
	platforms = (cl_platform_id*)malloc((platformID+1)*sizeof(cl_platform_id));
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
	addScaled = clCreateKernel(program, "addScaled", &err); checkErr();
	hadamard = clCreateKernel(program, "hadamard", &err); checkErr();
	reduce = clCreateKernel(program, "reduce", &err); checkErr();
	sigmoid = clCreateKernel(program, "sigmoid", &err); checkErr();
	dsigmoid = clCreateKernel(program, "dsigmoid", &err); checkErr();
	matVec = clCreateKernel(program, "matVec", &err); checkErr();
	matMul = clCreateKernel(program, "matMul", &err); checkErr();

	scalei = clCreateKernel(program, "scalei", &err); checkErr();
	dividei = clCreateKernel(program, "dividei", &err); checkErr();
	modulo = clCreateKernel(program, "modulo", &err); checkErr();
	addi = clCreateKernel(program, "addi", &err); checkErr();
	addScaledi = clCreateKernel(program, "addScaledi", &err); checkErr();
	hadamardi = clCreateKernel(program, "hadamardi", &err); checkErr();
	reducei = clCreateKernel(program, "reducei", &err); checkErr();
	matVeci = clCreateKernel(program, "matVeci", &err); checkErr();
	matMuli = clCreateKernel(program, "matMuli", &err); checkErr();
}

// General helper functions
void FinLin::setArg(cl_kernel kernel, int argno, cl_mem obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(cl_mem),
		(void*)&obj
	);
	FinLin::checkErr();
}
void FinLin::setArg(cl_kernel kernel, int argno, double obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(double),
		(void*)&obj
	);
	FinLin::checkErr();
}
void FinLin::setArg(cl_kernel kernel, int argno, int obj) {
	FinLin::err = clSetKernelArg(
		kernel,
		argno,
		sizeof(int),
		(void*)&obj
	);
	FinLin::checkErr();
}
void FinLin::writeBuffer(cl_mem buffer, size_t offset, size_t cb, const void *ptr) {
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
void FinLin::execKernel(
	cl_kernel kernel,
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
			1,
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
			1,
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
void FinLin::execKernel(
	cl_kernel kernel,
	size_t offset,
	size_t globalSizeX,
	size_t globalSizeY,
	size_t localSize // 0 for NULL
) {
	size_t workOffset = offset;
	size_t *globalWorkSize = (size_t*)malloc(2 * sizeof(size_t));
	globalWorkSize[0] = globalSizeX;
	globalWorkSize[1] = globalSizeY;
	size_t localWorkSize = localSize;
	if(localSize == 0) {
		FinLin::err = clEnqueueNDRangeKernel(
			FinLin::commandQueue,
			kernel,
			2,
			&workOffset,
			globalWorkSize,
			NULL,
			0,
			NULL,
			NULL
		);
	} else {
		FinLin::err = clEnqueueNDRangeKernel(
			FinLin::commandQueue,
			kernel,
			2,
			&workOffset,
			globalWorkSize,
			&localWorkSize,
			0,
			NULL,
			NULL
		);
	}
	FinLin::checkErr();
}
void FinLin::readBuffer(cl_mem buffer, size_t offset, size_t cb, void *ptr) {
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
