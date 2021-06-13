#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Vec *vecs = (Vec*)malloc(3 * sizeof(Vec));
	vecs[0] = Vec::randomUniform(4, 1, 2);
	vecs[1] = Vec::randomUniform(4, 1, 2);
	vecs[2] = Vec::randomUniform(4, 1, 2);

	printf("%s\n", vecs[0].string());
	printf("%s\n", vecs[1].string());
	printf("%s\n", vecs[2].string());

	Mat m = Mat::fromRowVecs(3, vecs);

	printf("%s\n", m.string());

	vecs = (Vec*)malloc(4 * sizeof(Vec));
	vecs[0] = m.colVec(0);
	vecs[1] = m.colVec(1);
	vecs[2] = m.colVec(2);
	vecs[3] = m.colVec(3);

	printf("%s\n", vecs[0].string());
	printf("%s\n", vecs[1].string());
	printf("%s\n", vecs[2].string());
	printf("%s\n", vecs[3].string());

	m = Mat::fromColVecs(4, vecs);

	printf("%s\n", m.string());

	vecs = (Vec*)malloc(3 * sizeof(Vec));
	vecs[0] = m.rowVec(0);
	vecs[1] = m.rowVec(1);
	vecs[2] = m.rowVec(2);

	printf("%s\n", vecs[0].string());
	printf("%s\n", vecs[1].string());
	printf("%s\n", vecs[2].string());
}
