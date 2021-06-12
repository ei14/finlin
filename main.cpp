#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Vec a = Vec::randomUniform(4, 1, 2);
	Vec b = Vec::randomUniform(4, 1, 2);

	printf("%s\n", a.string());
	printf("%s\n", b.string());

	a -= b;

	printf("%s\n", a.string());

	Mat c = Mat::randomUniform(4, 3, 1, 2);
	Mat d = Mat::randomUniform(4, 3, 1, 2);

	printf("%s\n", c.string());
	printf("%s\n", d.string());

	c -= d;

	printf("%s\n", c.string());
}
