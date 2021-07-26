main: finlin.cpp main.cpp vec.cpp mat.cpp veci.cpp mati.cpp
	g++ -c finlin.cpp
	g++ -c vec.cpp
	g++ -c mat.cpp
	g++ -c veci.cpp
	g++ -c mati.cpp
	ar rvs finlin.a finlin.o vec.o mat.o veci.o mati.o
	g++ main.cpp finlin.a -lOpenCL -o main

run:
	./main

shared: finlin.cpp main.cpp
	g++ -c finlin.cpp -fpic
	ld finlin.o -shared -o libfinlin.so
	g++ main.cpp -lfinlin
