# FinLin

## Finite linear algebra in C++/OpenCL

### Thomas Kaldahl, 2021

This library provides implementations of vectors, matrices, and operations on
and between them.
It is designed to be run using the GPU to perform parallel computations.

## Usage

### Initialization

FinLin is called by including the header file `finlin.hpp`.

Before use in a program, it must be initialized with the initialization function
`FinLin::init(int platform, int device)`.

OpenCL platform and device numbers are
machine dependent. To determine which platform and device numbers correspond to
your preferred device (e.g. CPU, GPU), try using
[clinfo](https://github.com/Oblomov/clinfo) or a non-Linux equivalent. If in
doubt, `FinLin::init(0, 0)` should get you started.

### Vectors

Vectors can be initialized in a couple ways.

* `Vec(int d)` creates a zero-vector of dimension `d`.
* `Vec(int d, double x)` creates a `d`-dimensional vector populated entirely
  with components equal to `x`.
* `Vec(int d, double *components)` uses `d` doubles at the double array at
  `components` to set the components of the vector.
* `Vec::randomUniform(int d, double min, double max)` creates a
  `d`-dimensional vector with random components from `min` to `max`.

Vectors have a few accessor methods. Given vector `v`,

* `int v.dim()` returns the dimension of `v`.
* `double v.comp(int i)` returns the `i`th component of `v`.
* `char *v.string()` gives a string representation of `v`.
* `double v.norm()` returns the magnitude of `v`.
* `Vec v.normal()` creates a unit vector of the same direction as `v`.

Vectors can be added, negated, and subtracted using standard operations like
`+`, `-`, `+=`, and `-=`. They can also be scaled by doubles with `*`, `/`,
`*=`, and `/=`. The dot product of two vectors is also found with `*`, but this
operation returns a `double` and not a `Vec`. The Hadamard (component-wise)
product between vectors is found with the operators `%` and `%=`.

The `i`th component of a vector `v` can be set to value `x` with
`double v.setComp(int i, double x)`. This method returns the previous value of
the `i`th component of `v`.

There also exists the in-place method `Vec v.normalize()` which preserves the
direction of `v` while scaling it to a unit vector.

For machine learning purposes, the methods `v.setSigmoid()` and
`v.setDsigmoid()` apply an in-place sigmoid and sigmoid-derivative function
on `v` respectively. The exact sigmoid function is
`f(x) = x / (1 + |2x|) + 1/2`. The non in-place methods are `v.sigmoid()` and
`v.dsigmoid()`; they return a new vector with the sigmoid applied, without
modifying `v`.

The static function `Vec *gramSchmidt(int n, Vec *vecs)` performs the
[Gram-Schmidt
Procedure](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
on the `n` vectors located at array `vecs`.

The `v.copy()` method returns a copy of `v`.

Finally, the `bool v.update()` method is a purely technical method which has
no effect on calculation results. It performs the expensive operation of
transferring memory from the CPU to the GPU in anticipation of a GPU
calculation. This needn't be called prior to operations as every operation
automatically calls this method if necessary. Calling this method prior to a
calculation will make the calculation faster, because the step of transferring
memory will have already been performed at calculation time. If nothing is
done between the update call and the operation, there is no net speed advantage
to calling this method explicitly.

### Matrices

Matrices can be initialized in a couple ways.

* `Mat(int d)` creates a `d` by `d` identity matrix.
* `Mat(int d, double x)` creates a `d` by `d` identity matrix, scaled by a
  factor of `x`.
* `Mat(int h, int w)` creates a zero-matrix of height `h` and width `w`.
* `Mat(int h, int w, double *components)` creates a `h` by `w` matrix, drawing
  components from the double array, row after row.
* `Mat::randomUniform(int h, int w, double min, double max)` creates a
  `h` by `w` matrix with random components from `min` to `max`.
* `Mat::fromRowVec(Vec row)` takes a `d`-dimensional vector `row` and creates a
  `1` by `d` matrix from its components.
* `Mat::fromColVec(Vec col)` takes a `d`-dimensional vector `col` and creates a
  `d` by `1` matrix from its components.
* `Mat::fromRowVecs(int n, Vec *rows)` takes `n` `d`-dimensional vectors from
  vector array `rows` and creates a `n` by `d` matrix from their components.
* `Mat::fromColVecs(int n, Vec *cols)` takes `n` `d`-dimensional vectors from
  vector array `cols` and creates a `d` by `n` matrix from their components.

Matrices have a few accessor methods. Given matrix `m`,

* `int m.height()` returns the height of `m`.
* `int m.width()` returns the width of `m`.
* `double m.comp(int r, int c)` returns the component from the `r`th row, `c`th
  column of `m`.
* `char *m.string()` gives a string representation of `m`.
* `double m.det()` returns the determinant of square matrix `m`.
* `double m.trace()` returns the trace of `m`.
* `double m.T()` returns the transpose of `m`.
* `double m.inv()` returns the inverse of square matrix `m`.
* `bool m.invertible()` returns true if `m` is invertible.
* `Vec m.rowVec(int r)` returns the `r`th row of `m` as a vector.
* `Vec m.colVec(int c)` returns the `c`th column of `m` as a vector.

Matrices can be added, negated, and subtracted using standard operations like
`+`, `-`, `+=`, and `-=`. They can also be scaled by doubles with `*`, `/`,
`*=`, and `/=`. Matrices can be multiplied with `*`, but not with `*=`. A matrix
can multiply a vector with `*` as well.

The component in the `r`th row, `c`th column of matrix `m` can be set to value
`x` with
`double m.setComp(int r, int c, double x)`. This method returns the previous
value of the changed component.

The in-place method `m.REFF()` reduces `m` to reduced row echelon form. The
implementation of this is a bit limited at the moment and only supports
certain matrices.

The `m.copy()` method returns a copy of `m`.

Finally, the `bool m.update()` performs similarly to the corresponding vector
method, described above.

## License

This software is licensed under the
[MIT license.](https://opensource.org/licenses/MIT)
