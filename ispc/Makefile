CXX=clang++ -m64
CXXFLAGS=-Iobjs/ -O3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 -g --arch=x86-64 --target=avx2-i32x8

default: permutations

.PHONY: dirs clean
.PRECIOUS: objs/permutations.h

dirs:
	/bin/mkdir -p objs/

clean:
	/bin/rm -rf objs *~ permutations

permutations:
	/bin/mkdir -p objs/
	ispc $(ISPCFLAGS) permutations.ispc -o objs/permutations_ispc.o -h objs/permutations_ispc.h
	clang++ -m64 -g -mavx -Iobjs/ -O3 -Wall -c -o objs/permutations.o permutations.cpp
	clang++ -m64 -g -mavx -lobjs/ -O3 -Wall -c -o objs/task.o tasksys.cpp
	clang++ -m64 -g -mavx -Iobjs/ -O3 -Wall -o permutations objs/permutations.o objs/permutations_ispc.o objs/task.o -lpthread
