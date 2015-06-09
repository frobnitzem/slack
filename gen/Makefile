NVCC=nvcc
CFLAGS=-fPIC
NVFLAGS=--compiler-options '$(CFLAGS)'
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart
#CFLAGS=-g

#kern=test_kern.cu
#NVFLAGS += -DTEST

#kern=kern2.cu
#NVFLAGS += -DTX=96 -DTY=1 -DWX=1 -DWY=96 -DNEST=32

kern=test3.cu

#kern=kern3.cu
#NVFLAGS += -DNAME=tdot_kernel_T8_1_32_1_W1_1_1_32_N8_8

#kern=magma_kern.cu
#NVFLAGS += -DMAGMA

tdot.so: tdot2.c dot_plan.c $(kern) drive_kern.cu
	$(NVCC) $(NVFLAGS) -shared -o $@ $^

tdot2: tdot2.c dot_plan.c $(kern) drive_kern.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

