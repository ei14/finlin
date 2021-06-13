#include "finlin.hpp"

int main() {
	FinLin::init(2, 0);

	Mat m = Mat::randomUniform(4, 4, -1, 1);
	printf("%s\n", m.string());
	printf("%s\n", m.inv().string());
}
