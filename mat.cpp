// Copyright (c) 2021 Thomas Kaldahl

#include "finlin.hpp"

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
void ensureSquare(int h, int w, const char *operation) {
	if(w != h) {
		fprintf(
			stderr,
			"Cannot %s of non-square matrix.\n",
			operation
		);
		exit(1);
	}
}
void ensureNonzero(int h, int w, const char *operation) {
	if(w == 0 || h == 0) {
		fprintf(
			stderr,
			"Cannot %s of matrix with zero elements.\n",
			operation
		);
		exit(1);
	}
}
void ensureInbound(int r, int c, int h, int w, const char *operation) {
	if(r < 0) {
		fprintf(
			stderr,
			"Cannot %s in negative row (row %d).\n",
			operation,
			r
		);
		exit(1);
	}
	if(c < 0) {
		fprintf(
			stderr,
			"Cannot %s in negative column (column %d).\n",
			operation,
			c
		);
		exit(1);
	}
	if(r >= h) {
		fprintf(
			stderr,
			"Matrix only contains %d rows. "
			"Cannot %s in row %d.\n",
			h,
			operation,
			r
		);
		exit(1);
	}
	if(c >= w) {
		fprintf(
			stderr,
			"Matrix only contains %d columns. "
			"Cannot %s in column %d.\n",
			w,
			operation,
			c
		);
		exit(1);
	}
}

// Technical methods
void Mat::createMem() {
	clmem = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		w*h * sizeof(double),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();
	dirty = true;
}

Mat Mat::copy() const {
	Mat res = Mat(h, w);
	memcpy(res.data, data, w*h * sizeof(double));
	return res;
}
bool Mat::update() {
	if(!dirty) return false;
	FinLin::writeBuffer(clmem, 0, w*h * sizeof(double), data);
	dirty = false;
	return true;
}

// Constructors
Mat::Mat(int height, int width, double *components) {
	h = height;
	w = width;
	data = components;
	createMem();
}
Mat::Mat(int height, int width) {
	h = height;
	w = width;
	data = (double*)malloc(w*h * sizeof(double));
	memset(data, 0, w*h * sizeof(double));
	createMem();
}
Mat::Mat(int size, double scalar) {
	h = size;
	w = size;
	data = (double*)malloc(w*h * sizeof(double));
	memset(data, 0, w*h * sizeof(double));
	for(int i = 0; i < w*h; i += w+1) {
		data[i] = scalar;
	}
	createMem();
}
Mat::Mat(int size) : Mat(size, 1.0) {}

// Statics
Mat Mat::randomUniform(int height, int width, double min, double max) {
	double *components = (double*)malloc(width*height * sizeof(double));
	for(int i = 0; i < width*height; i++) {
		components[i] = (max - min) * rand() / RAND_MAX + min;
	}
	return Mat(height, width, components);
}
Mat Mat::fromRowVecs(int numVecs, Vec *vecs) {
	if(numVecs == 0) return Mat(0);
	int width = vecs[0].d;
	double *components = (double*)malloc(numVecs*width * sizeof(double));
	for(int row = 0; row < numVecs; row++) {
		if(vecs[row].d != width) {
			fprintf(stderr, "Cannot construct matrix from vectors"
				" of varying dimension.\n");
			exit(1);
		}
		memcpy(components + row*width, vecs[row].data, width * sizeof(double));
	}
	return Mat(numVecs, width, components);
}
Mat Mat::fromColVecs(int numVecs, Vec *vecs) {
	return fromRowVecs(numVecs, vecs).T();
}

// Accessors
int Mat::height() const {
	return h;
}
int Mat::width() const {
	return w;
}

double Mat::comp(int r, int c) const {
	ensureInbound(r, c, h, w, "access component");
	return data[w*r + c];
}

char *Mat::string() const {
	const int MAXLEN = 12;
	char *res = (char*)malloc(MAXLEN * w*h * sizeof(char));
	snprintf(res, MAXLEN, "(\n");
	char *resp = res + 2;
	for(int r = 0; r < h; r++) {
		for(int c = 0; c < w; c++) {
			snprintf(resp, MAXLEN, "\t%0.3f", data[w*r + c]);
			resp += strlen(resp);
		}
		*resp = '\n';
		resp++;
	}
	snprintf(resp, 2, ")");
	return res;
}

// In-place operations
Mat Mat::operator*=(double scalar) {
	update();

	FinLin::setArg(FinLin::scale, 0, clmem);
	FinLin::setArg(FinLin::scale, 1, scalar);
	FinLin::execKernel(FinLin::scale, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(double), data);

	return *this;
}
Mat Mat::operator/=(double divisor) {
	return *this *= 1.0/divisor;
}

Mat Mat::operator+=(Mat addend) {
	ensureSameMatDim(h, w, addend.h, addend.w, "add");

	update();
	addend.update();

	FinLin::setArg(FinLin::add, 0, clmem);
	FinLin::setArg(FinLin::add, 1, addend.clmem);
	FinLin::execKernel(FinLin::add, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(double), data);

	return *this;
}
Mat Mat::operator-=(Mat subtrahend) {
	ensureSameMatDim(h, w, subtrahend.h, subtrahend.w, "subtract");

	update();
	subtrahend.update();

	FinLin::setArg(FinLin::addScaled, 0, clmem);
	FinLin::setArg(FinLin::addScaled, 1, subtrahend.clmem);
	FinLin::setArg(FinLin::addScaled, 2, -1.0);
	FinLin::execKernel(FinLin::addScaled, 0, w*h, 0);

	FinLin::readBuffer(clmem, 0, w*h * sizeof(double), data);

	return *this;
}

// Binary operations
Mat Mat::operator*(double scalar) const {
	Mat matrix = copy();
	matrix *= scalar;
	return matrix;
}
Mat operator*(double scalar, Mat matrix) {
	Mat product = matrix.copy();
	product *= scalar;
	return product;
}
Mat Mat::operator/(double divisor) const {
	Mat dividend = copy();
	dividend /= divisor;
	return dividend;
}

Vec Mat::operator*(Vec vector) {
	ensureMulMatDims(w, vector.d, "multiply");
	update();
	vector.update();

	cl_mem resBuff = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		h * sizeof(double),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();

	FinLin::setArg(FinLin::matVec, 0, clmem);
	FinLin::setArg(FinLin::matVec, 1, vector.clmem);
	FinLin::setArg(FinLin::matVec, 2, resBuff);
	FinLin::setArg(FinLin::matVec, 3, w);
	FinLin::execKernel(FinLin::matVec, 0, h, 0);

	double *resData = (double*)malloc(h * sizeof(double));
	FinLin::readBuffer(resBuff, 0, h * sizeof(double), resData);

	return Vec(h, resData);
}

Mat Mat::operator*(Mat multiplier) {
	ensureMulMatDims(w, multiplier.h, "multiply");
	update();
	multiplier.update();

	cl_mem resBuff = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		h * multiplier.w * sizeof(double),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();

	FinLin::setArg(FinLin::matMul, 0, clmem);
	FinLin::setArg(FinLin::matMul, 1, multiplier.clmem);
	FinLin::setArg(FinLin::matMul, 2, resBuff);
	FinLin::setArg(FinLin::matMul, 3, w);
	FinLin::execKernel(FinLin::matMul, 0, h, multiplier.w, 0);

	double *resData = (double*)malloc(h * multiplier.w * sizeof(double));
	FinLin::readBuffer(resBuff, 0, h * multiplier.w * sizeof(double), resData);

	return Mat(h, multiplier.w, resData);
}

Mat Mat::operator+(Mat addend) const {
	Mat augend = copy();
	augend += addend;
	return augend;
}
Mat Mat::operator-(Mat subtrahend) const {
	Mat minuend = copy();
	minuend -= subtrahend;
	return minuend;
}

// Misc operations
double Mat::minor(int r, int c) const {
	ensureNonzero(h, w, "find minor");
	ensureSquare(h, w, "find minor");

	double *dest = (double*)malloc((w-1)*(h-1) * sizeof(double));

	double *destp = dest;
	double *srcp = data;
	for(int row = 0; row < h-1; row++) {
		if(row == r) srcp += w;

		memcpy(destp, srcp, c*sizeof(double));
		memcpy(destp + c, srcp + c+1, (w-1 - c)*sizeof(double));

		destp += (w-1);
		srcp += w;
	}

	return Mat(h-1, w-1, dest).det();
}
Vec Mat::rowVec(int row) const {
	double *components = (double*)malloc(w * sizeof(double));
	memcpy(components, data + row*w, w * sizeof(double));
	return Vec(w, components);
}
Vec Mat::colVec(int col) const {
	double *components = (double*)malloc(h * sizeof(double));
	for(int r = 0; r < h; r++) {
		components[r] = comp(r, col);
	}
	return Vec(h, components);
}

// Unary operations
double Mat::det() const {
	ensureSquare(h, w, "take determinant");
	if(h == 0) return 0;
	if(h == 1) return data[0];

	double sum = 0;
	for(int x = 0; x < w; x++) {
		sum += (x % 2 == 0 ? 1 : -1) * data[x] * minor(0, x);
	}
	return sum;
}

bool Mat::invertible() const {
	return det() != 0;
}

double Mat::trace() const {
	ensureSquare(h, w, "find trace");
	double sum = 0;
	for(int i = 0; i < h; i++) {
		sum += comp(i, i);
	}
	return sum;
}

Mat Mat::operator-() const {
	return -1 * *this;
}

Mat Mat::T() const {
	double *res = (double*)malloc(w*h * sizeof(double));
	for(int r = 0; r < h; r++) {
		for(int c = 0; c < w; c++) {
			res[c*h + r] = data[r*w + c];
		}
	}
	return Mat(w, h, res);
}