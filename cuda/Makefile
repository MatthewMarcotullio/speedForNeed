NVCC = /usr/local/cuda/bin/nvcc 

matrixMult:matrixMult.o
	$(NVCC) matrixMult.o -o matrixMult

matrixMult.o:matrixMult.cu
	$(NVCC) -c matrixMult.cu

clean:
	rm *.o
	rm matrixMult
