#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Mat m = Mat::randomUniform(4, 4, 1, 2);
	printf("%s\n", m.string());
	Vec *vecs = (Vec*)malloc(4 * sizeof(Vec));
	for(int i = 0; i < 4; i++) {
		vecs[i] = m.colVec(i);
	}
	Mat n = Mat::fromColVecs(4, Vec::gramSchmidt(4, vecs));
	printf("%s\n", n.string());

	printf("%f\n", m.det());
}
