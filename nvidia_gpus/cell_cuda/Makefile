##CC = /usr/local/cuda/bin/nvcc
CC = /soft/compilers/cuda/cuda-9.1.85/bin/nvcc

#INCLUDE = /usr/local/cuda/include:../common/inc/
#LIB = /usr/local/cuda/lib:../common/lib/
#LIB = /usr/local/cuda/include:../common/inc/
LIB = /soft/compilers/cuda/cuda-9.1.85/include/
INCLUDE = /soft/compilers/cuda/cuda-9.1.85/nvvm/include/ 
#INCLUDE = ../common/inc/
#LIB = ../common/lib/

LIBRARIES = cudart.lib cutil32.lib

SRC = cell.cu

EXE = cell

release:
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) $(FLAGS) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) 
