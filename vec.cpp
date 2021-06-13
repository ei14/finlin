// Copyright (c) 2021 Thomas Kaldahl

#include "finlin.hpp"

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
	FinLin::writeBuffer(clmem, 0, d*sizeof(double), data);
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
	snprintf(res, 3, "< ");
	char *resp = res + strlen(res);
	for(int i = 0; i < d; i++) {
		snprintf(resp, MAXLEN, "%0.3f, ", data[i]);
		resp += strlen(resp);
	}
	resp -= strlen(", ");
	snprintf(resp, 3, " >");
	return res;
}

// In-place operations
Vec Vec::operator*=(double scalar) {
	update();

	FinLin::setArg(FinLin::scale, 0, clmem);
	FinLin::setArg(FinLin::scale, 1, scalar);
	FinLin::execKernel(FinLin::scale, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::operator/=(double divisor) {
	return *this *= 1.0/divisor;
}

Vec Vec::operator+=(Vec addend) {
	ensureSameVecDim(d, addend.d, "add");

	update();
	addend.update();

	FinLin::setArg(FinLin::add, 0, clmem);
	FinLin::setArg(FinLin::add, 1, addend.clmem);
	FinLin::execKernel(FinLin::add, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::operator-=(Vec subtrahend) {
	ensureSameVecDim(d, subtrahend.d, "subtract");

	update();
	subtrahend.update();

	FinLin::setArg(FinLin::addScaled, 0, clmem);
	FinLin::setArg(FinLin::addScaled, 1, subtrahend.clmem);
	FinLin::setArg(FinLin::addScaled, 2, -1.0);
	FinLin::execKernel(FinLin::addScaled, 0, d, 0);

	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::operator%=(Vec multiplier) {
	ensureSameVecDim(d, multiplier.d, "multiply");
	update();
	multiplier.update();

	FinLin::setArg(FinLin::hadamard, 0, clmem);
	FinLin::setArg(FinLin::hadamard, 1, multiplier.clmem);
	FinLin::execKernel(FinLin::hadamard, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

Vec Vec::normalize() {
	return *this /= norm();
}
Vec Vec::setSigmoid() {
	update();

	FinLin::setArg(FinLin::sigmoid, 0, clmem);
	FinLin::execKernel(FinLin::sigmoid, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}
Vec Vec::setDsigmoid() {
	update();

	FinLin::setArg(FinLin::dsigmoid, 0, clmem);
	FinLin::execKernel(FinLin::dsigmoid, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(double), data);

	return *this;
}

// Binary operations
Vec Vec::operator*(double scalar) const {
	Vec vector = copy();
	vector *= scalar;
	return vector;
}
Vec operator*(double scalar, Vec vector) {
	return vector * scalar;
}
Vec Vec::operator/(double divisor) const {
	Vec dividend = copy();
	dividend /= divisor;
	return dividend;
}

Vec Vec::operator+(Vec addend) const {
	ensureSameVecDim(d, addend.d, "add");
	Vec augend = copy();
	augend += addend;
	return augend;
}
Vec Vec::operator-(Vec subtrahend) const {
	ensureSameVecDim(d, subtrahend.d, "subtract");
	Vec minuend = copy();
	minuend -= subtrahend;
	return minuend;
}
Vec Vec::operator%(Vec multiplier) const {
	ensureSameVecDim(d, multiplier.d, "multiply");
	Vec multiplicand = copy();
	multiplicand %= multiplier;
	return multiplicand;
}
double Vec::operator*(Vec multiplier) const {
	if(d == 0 || multiplier.d == 0) return 0;

	Vec hdm = *this % multiplier; // hdm is for Hadamard

	int len = d; // Is cut in half until down to 1.

	FinLin::setArg(FinLin::reduce, 0, hdm.clmem);

	while(len != 1) {
		// Include the last element in the event of an odd length
		if(len % 2 == 1) {
			hdm.data[0] += hdm.data[len - 1]; // Add it to the first component
			FinLin::writeBuffer(hdm.clmem, 0, sizeof(double), hdm.data);
		}

		len /= 2;
		FinLin::setArg(FinLin::reduce, 1, len);

		FinLin::execKernel(FinLin::reduce, 0, len, 0);
	}

	// Only the first element is read. It should equal the result.
	FinLin::readBuffer(hdm.clmem, 0, sizeof(double), hdm.data);

	return hdm.data[0];
}

// Unary operations
Vec Vec::operator-() const {
	return -1 * *this;
}

double Vec::norm() const {
	return sqrt(*this * *this);
}
Vec Vec::normal() const {
	return *this / norm();
}

Vec Vec::sigmoid() const {
	Vec res = copy();
	res.setSigmoid();
	return res;
}
Vec Vec::dsigmoid() const {
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

