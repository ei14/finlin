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
			"Matirix only contains %d rows. "
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
			"Matirix only contains %d columns. "
			"Cannot %s in column %d.\n",
			w,
			operation,
			c
		);
		exit(1);
	}
}

// Technical methods
void Mati::createMem() {
	clmem = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		w*h * sizeof(int),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();
	dirty = true;
}

Mati Mati::copy() const {
	Mati res = Mati(h, w);
	memcpy(res.data, data, w*h * sizeof(int));
	return res;
}
bool Mati::update() {
	if(!dirty) return false;
	FinLin::writeBuffer(clmem, 0, w*h * sizeof(int), data);
	dirty = false;
	return true;
}

// Constructors
Mati::Mati(int height, int width, int *components) {
	h = height;
	w = width;
	data = components;
	createMem();
}
Mati::Mati(int height, int width) {
	h = height;
	w = width;
	data = (int*)malloc(w*h * sizeof(int));
	memset(data, 0, w*h * sizeof(int));
	createMem();
}
Mati::Mati(int size) {
	h = size;
	w = size;
	data = (int*)malloc(w*h * sizeof(int));
	memset(data, 0, w*h * sizeof(int));
	for(int i = 0; i < w*h; i += w+1) {
		data[i] = 1;
	}
	createMem();
}
Mati::Mati(Mat mat) {
	w = mat.w;
	h = mat.h;
	data = (int*)malloc(w*h * sizeof(int));
	for(int i = 0; i < w*h; i++) {
		data[i] = (int)mat.data[i];
	}
	createMem();
}

// Statics
Mati Mati::randomUniform(int height, int width, int min, int max) {
	int *components = (int*)malloc(width*height * sizeof(int));
	for(int i = 0; i < width*height; i++) {
		components[i] = (max - min) * rand() / RAND_MAX + min;
	}
	return Mati(height, width, components);
}
Mati Mati::fromRowVec(Veci row) {
	return Mati(1, row.d, row.data);
}
Mati Mati::fromColVec(Veci col) {
	return Mati(col.d, 1, col.data);
}
Mati Mati::fromRowVecs(int numVecs, Veci *vecs) {
	if(numVecs == 0) return Mati(0);
	int width = vecs[0].d;
	int *components = (int*)malloc(numVecs*width * sizeof(int));
	for(int row = 0; row < numVecs; row++) {
		if(vecs[row].d != width) {
			fprintf(stderr, "Cannot construct matrix from vectors"
				" of varying dimension.\n");
			exit(1);
		}
		memcpy(components + row*width, vecs[row].data, width * sizeof(int));
	}
	return Mati(numVecs, width, components);
}
Mati Mati::fromColVecs(int numVecis, Veci *vecs) {
	return fromRowVecs(numVecis, vecs).T();
}

// Accessors
int Mati::height() const {
	return h;
}
int Mati::width() const {
	return w;
}

int Mati::comp(int r, int c) const {
	ensureInbound(r, c, h, w, "access component");
	return data[w*r + c];
}

char *Mati::string() const {
	const int MAXLEN = 12;
	char *res = (char*)malloc(MAXLEN * w*h * sizeof(char));
	snprintf(res, MAXLEN, "(\n");
	char *resp = res + 2;
	for(int r = 0; r < h; r++) {
		for(int c = 0; c < w; c++) {
			snprintf(resp, MAXLEN, "\t%d", data[w*r + c]);
			resp += strlen(resp);
		}
		*resp = '\n';
		resp++;
	}
	snprintf(resp, 2, ")");
	return res;
}

// In-place operations
Mati Mati::operator*=(int scalar) {
	update();

	FinLin::setArg(FinLin::scalei, 0, clmem);
	FinLin::setArg(FinLin::scalei, 1, scalar);
	FinLin::execKernel(FinLin::scalei, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}
Mati Mati::operator/=(int divisor) {
	update();

	FinLin::setArg(FinLin::dividei, 0, clmem);
	FinLin::setArg(FinLin::dividei, 1, divisor);
	FinLin::execKernel(FinLin::dividei, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}
Mati Mati::operator%=(int modulus) {
	update();

	FinLin::setArg(FinLin::modulo, 0, clmem);
	FinLin::setArg(FinLin::modulo, 1, modulus);
	FinLin::execKernel(FinLin::modulo, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}
Mati Mati::operator^=(int exponent) {
	for(int i = 1; i < exponent; i++) {
		*this = *this * *this;
	}
	return *this;
}

Mati Mati::operator&=(Mati multiplier) {
	ensureSameMatDim(h, w, multiplier.h, multiplier.w, "multiply");

	update();
	multiplier.update();

	FinLin::setArg(FinLin::hadamardi, 0, clmem);
	FinLin::setArg(FinLin::hadamardi, 1, multiplier.clmem);
	FinLin::execKernel(FinLin::hadamardi, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}
Mati Mati::operator+=(Mati addend) {
	ensureSameMatDim(h, w, addend.h, addend.w, "add");

	update();
	addend.update();

	FinLin::setArg(FinLin::addi, 0, clmem);
	FinLin::setArg(FinLin::addi, 1, addend.clmem);
	FinLin::execKernel(FinLin::addi, 0, w*h, 0);
	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}
Mati Mati::operator-=(Mati subtrahend) {
	ensureSameMatDim(h, w, subtrahend.h, subtrahend.w, "subtract");

	update();
	subtrahend.update();

	FinLin::setArg(FinLin::addScaledi, 0, clmem);
	FinLin::setArg(FinLin::addScaledi, 1, subtrahend.clmem);
	FinLin::setArg(FinLin::addScaledi, 2, -1);
	FinLin::execKernel(FinLin::addScaledi, 0, w*h, 0);

	FinLin::readBuffer(clmem, 0, w*h * sizeof(int), data);

	return *this;
}

// Binary operations
Mati Mati::operator*(int scalar) const {
	Mati matrix = copy();
	matrix *= scalar;
	return matrix;
}
Mati operator*(int scalar, Mati matrix) {
	Mati product = matrix.copy();
	product *= scalar;
	return product;
}
Mati Mati::operator/(int divisor) const {
	Mati dividend = copy();
	dividend /= divisor;
	return dividend;
}
Mati Mati::operator%(int modulus) const {
	Mati dividend = copy();
	dividend %= modulus;
	return dividend;
}
Mati Mati::operator^(int exponent) const {
	Mat base = copy();
	base ^= exponent;
	return base;
}

Veci Mati::operator*(Veci vector) {
	ensureMulMatDims(w, vector.d, "multiply");
	update();
	vector.update();

	cl_mem resBuff = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		h * sizeof(int),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();

	FinLin::setArg(FinLin::matVeci, 0, clmem);
	FinLin::setArg(FinLin::matVeci, 1, vector.clmem);
	FinLin::setArg(FinLin::matVeci, 2, resBuff);
	FinLin::setArg(FinLin::matVeci, 3, w);
	FinLin::execKernel(FinLin::matVeci, 0, h, 0);

	int *resData = (int*)malloc(h * sizeof(int));
	FinLin::readBuffer(resBuff, 0, h * sizeof(int), resData);

	return Veci(h, resData);
}

Mati Mati::operator*(Mati multiplier) {
	ensureMulMatDims(w, multiplier.h, "multiply");
	update();
	multiplier.update();

	cl_mem resBuff = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		h * multiplier.w * sizeof(int),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();

	FinLin::setArg(FinLin::matMuli, 0, clmem);
	FinLin::setArg(FinLin::matMuli, 1, multiplier.clmem);
	FinLin::setArg(FinLin::matMuli, 2, resBuff);
	FinLin::setArg(FinLin::matMuli, 3, w);
	FinLin::execKernel(FinLin::matMuli, 0, h, multiplier.w, 0);

	int *resData = (int*)malloc(h * multiplier.w * sizeof(int));
	FinLin::readBuffer(resBuff, 0, h * multiplier.w * sizeof(int), resData);

	return Mati(h, multiplier.w, resData);
}

Mati Mati::operator&(Mati multiplier) const {
	Mati multiplicand = copy();
	multiplicand &= multiplier;
	return multiplicand;
}
Mati Mati::operator+(Mati addend) const {
	Mati augend = copy();
	augend += addend;
	return augend;
}
Mati Mati::operator-(Mati subtrahend) const {
	Mati minuend = copy();
	minuend -= subtrahend;
	return minuend;
}

// Unary operations

int Mati::trace() const {
	ensureSquare(h, w, "find trace");
	int sum = 0;
	for(int i = 0; i < h; i++) {
		sum += comp(i, i);
	}
	return sum;
}

Mati Mati::operator-() const {
	return -1 * *this;
}
Mati Mati::operator~() const {
	Mati negated = copy();
	negated.update();

	FinLin::setArg(FinLin::compNoti, 0, negated.clmem);
	FinLin::execKernel(FinLin::compNoti, 0, w*h, 0);
	FinLin::readBuffer(negated.clmem, 0, w*h * sizeof(int), negated.data);

	return negated;
}
Mati Mati::T() const {
	int *res = (int*)malloc(w*h * sizeof(int));
	for(int r = 0; r < h; r++) {
		for(int c = 0; c < w; c++) {
			res[c*h + r] = data[r*w + c];
		}
	}
	return Mati(w, h, res);
}

// Misc operations
Veci Mati::rowVeci(int row) const {
	int *components = (int*)malloc(w * sizeof(int));
	memcpy(components, data + row*w, w * sizeof(int));
	return Veci(w, components);
}
Veci Mati::colVeci(int col) const {
	int *components = (int*)malloc(h * sizeof(int));
	for(int r = 0; r < h; r++) {
		components[r] = comp(r, col);
	}
	return Veci(h, components);
}
