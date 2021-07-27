// Copyright (c) 2021 Thomas Kaldahl

#include "finlin.hpp"

// Helper functions
void ensureSameVeciDim(int d1, int d2, const char *operation) {
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
void Veci::createMem() {
	clmem = clCreateBuffer(
		FinLin::context,
		CL_MEM_READ_WRITE,
		d * sizeof(int),
		NULL,
		&FinLin::err
	);
	FinLin::checkErr();
	dirty = true;
}

Veci Veci::copy() const {
	Veci res = Veci(d);
	memcpy(res.data, data, d * sizeof(int));
	return res;
}
bool Veci::update() {
	if(!dirty) return false;
	FinLin::writeBuffer(clmem, 0, d*sizeof(int), data);
	dirty = false;
	return true;
}

// Constructors
Veci::Veci(int dimension, int* components) {
	d = dimension;
	data = components;
	createMem();
}
Veci::Veci(int dimension) {
	d = dimension;
	data = (int*)malloc(d * sizeof(int));
	memset(data, 0, d * sizeof(int));
	createMem();
}
Veci::Veci(Vec vec) {
	d = vec.d;
	data = (int*)malloc(d * sizeof(int));
	for(int i = 0; i < d; i++) {
		data[i] = (int)vec.data[i];
	}
	createMem();
}

// Statics
Veci Veci::randomUniform(int dim, int min, int max) {
	int *components = (int*)malloc(dim * sizeof(int));
	for(int i = 0; i < dim; i++) {
		components[i] = rand() % (max - min) + min;
	}
	return Veci(dim, components);
}

// Accessors
int Veci::dim() const {
	return d;
}

int Veci::comp(int index) const {
	return data[index];
}

char *Veci::string() const {
	const int MAXLEN = 12;
	char *res = (char*)malloc(MAXLEN * d * sizeof(char));
	snprintf(res, 3, "< ");
	char *resp = res + strlen(res);
	for(int i = 0; i < d; i++) {
		snprintf(resp, MAXLEN, "%d, ", data[i]);
		resp += strlen(resp);
	}
	resp -= strlen(", ");
	snprintf(resp, 3, " >");
	return res;
}

// In-place operations
Veci Veci::operator*=(int scalar) {
	update();

	FinLin::setArg(FinLin::scalei, 0, clmem);
	FinLin::setArg(FinLin::scalei, 1, scalar);
	FinLin::execKernel(FinLin::scalei, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

Veci Veci::operator/=(int divisor) {
	update();

	FinLin::setArg(FinLin::dividei, 0, clmem);
	FinLin::setArg(FinLin::dividei, 1, divisor);
	FinLin::execKernel(FinLin::dividei, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

Veci Veci::operator%=(int modulus) {
	update();

	FinLin::setArg(FinLin::modulo, 0, clmem);
	FinLin::setArg(FinLin::modulo, 1, modulus);
	FinLin::execKernel(FinLin::modulo, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

Veci Veci::operator+=(Veci addend) {
	ensureSameVeciDim(d, addend.d, "add");

	update();
	addend.update();

	FinLin::setArg(FinLin::addi, 0, clmem);
	FinLin::setArg(FinLin::addi, 1, addend.clmem);
	FinLin::execKernel(FinLin::addi, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

Veci Veci::operator-=(Veci subtrahend) {
	ensureSameVeciDim(d, subtrahend.d, "subtract");

	update();
	subtrahend.update();

	FinLin::setArg(FinLin::addScaledi, 0, clmem);
	FinLin::setArg(FinLin::addScaledi, 1, subtrahend.clmem);
	FinLin::setArg(FinLin::addScaledi, 2, -1);
	FinLin::execKernel(FinLin::addScaledi, 0, d, 0);

	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

Veci Veci::operator&=(Veci multiplier) {
	ensureSameVeciDim(d, multiplier.d, "multiply");
	update();
	multiplier.update();

	FinLin::setArg(FinLin::hadamardi, 0, clmem);
	FinLin::setArg(FinLin::hadamardi, 1, multiplier.clmem);
	FinLin::execKernel(FinLin::hadamardi, 0, d, 0);
	FinLin::readBuffer(clmem, 0, d*sizeof(int), data);

	return *this;
}

// Binary operations
Veci Veci::operator*(int scalar) const {
	Veci vector = copy();
	vector *= scalar;
	return vector;
}
Veci operator*(int scalar, Veci vector) {
	return vector * scalar;
}
Veci Veci::operator/(int divisor) const {
	Veci dividend = copy();
	dividend /= divisor;
	return dividend;
}
Veci Veci::operator%(int modulus) const {
	Veci dividend = copy();
	dividend /= modulus;
	return dividend;
}
int Veci::operator^(int exponent) const {
	int sqrMag = (*this * *this);
	int res = 1;
	for(int i = 0; i < exponent / 2; i++) {
		res *= sqrMag;
	}
	return res;
}

Veci Veci::operator+(Veci addend) const {
	ensureSameVeciDim(d, addend.d, "add");
	Veci augend = copy();
	augend += addend;
	return augend;
}
Veci Veci::operator-(Veci subtrahend) const {
	ensureSameVeciDim(d, subtrahend.d, "subtract");
	Veci minuend = copy();
	minuend -= subtrahend;
	return minuend;
}
Veci Veci::operator&(Veci multiplier) const {
	ensureSameVeciDim(d, multiplier.d, "multiply");
	Veci multiplicand = copy();
	multiplicand &= multiplier;
	return multiplicand;
}
int Veci::operator*(Veci multiplier) const {
	if(d == 0 || multiplier.d == 0) return 0;

	Veci hdm = *this & multiplier; // hdm is for Hadamard

	return hdm.sum();
}

// Unary operations
int Veci::sum() const {
	Vec mutated = copy();
	mutated.update();

	int len = d; // Is cut in half until down to 1.

	FinLin::setArg(FinLin::reducei, 0, mutated.clmem);

	while(len != 1) {
		// Include the last element in the event of an odd length
		if(len % 2 == 1) {
			mutated.data[0] += mutated.data[len - 1]; // Add it to the first component
			FinLin::writeBuffer(mutated.clmem, 0, sizeof(int), mutated.data);
		}

		len /= 2;
		FinLin::setArg(FinLin::reducei, 1, len);

		FinLin::execKernel(FinLin::reducei, 0, len, 0);
	}

	// Only the first element is read. It should equal the result.
	FinLin::readBuffer(mutated.clmem, 0, sizeof(int), mutated.data);

	return mutated.data[0];
}
Veci Veci::operator~() const {
	Veci negated = copy();
	negated.update();

	FinLin::setArg(FinLin::compNoti, 0, negated.clmem);
	FinLin::execKernel(FinLin::compNoti, 0, d, 0);
	FinLin::readBuffer(negated.clmem, 0, d*sizeof(int), negated.data);

	return negated;
}

Veci Veci::operator-() const {
	return -1 * *this;
}

// Mutators
int Veci::setComp(int index, int value) {
	int prev = data[index];
	data[index] = value;
	dirty = true;
	return prev;
}

