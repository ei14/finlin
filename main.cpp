#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Vec a = Vec::randomUniform(4, 0, 1);
	Vec b = Vec::randomUniform(4, 0, 1);

	printf("%f\n", a * b);
	printf("%s\n", (Mat::fromRowVec(a) * Mat::fromColVec(b)).string());
}
