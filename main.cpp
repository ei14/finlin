#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Vec a = Vec(3, new double[] {1, 2, 3});
	Vec b = Vec(3, new double[] {4, 5, 6});

	printf("%s\n", a.string());
	printf("%s\n", b.string());
	printf("%f\n", 3 * a * b);
	printf("%f\n", a.norm());
	printf("%s\n", a.normal().string());
	printf("%s\n", Vec::randomUniform(3, 10, 20).string());
}
