#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Mat a = Mat::randomUniform(4, 3, 1, 2);
	printf("%s\n", a.string());
	printf("%s\n", a.T().string());
	printf("%s\n", (a.T() * a).string());
}
