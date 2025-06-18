NVCC = nvcc
PROJ_DIR = ..
LIB_DIR = ${PROJ_DIR}/../lib

CUB_DIR = ${LIB_DIR}/cub
MGPU_DIR = ${LIB_DIR}/moderngpu
CNMEM_DIR = ${LIB_DIR}/cnmem

LIB_INCLUDE = -I${MGPU_DIR}/src -I${CNMEM_DIR}/include -I${CUB_DIR}  

CNMEM_LIB = -L${CNMEM_DIR}/build -lcnmem
LIBS = ${CNMEM_LIB} 

MGPU_FLAGS = --expt-extended-lambda -use_fast_math -Wno-deprecated-declarations
OPEN_FLAGS=-DOPENMP -Xcompiler -fopenmp -lgomp -lpthread
# To support backtrace, which is useful for debugging
BACKTRACE_FLAGS = -Xcompiler -rdynamic


CFLAGS +=  -O3 -Xptxas=-v -std=c++17 -DUNTRACK_ALLOC_LARGE_VERTEX_NUM -DPTHREAD_LOCK \
					-DCUDA_CONTEXT_PROFILE \
					# -DPROFILE -DCUDA_CONTEXT_PROFILE \
					#-DDEBUG  \
					#-DREUSE_PROFILE
ARCH = -arch=sm_80

.PHONY:all
all: triangle general

triangle: triangle.cu
	$(NVCC) $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ $(LIBS)

triangle-gdb: triangle.cu
	$(NVCC) -g -G $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ $(LIBS)
	
general: general.cu
	$(NVCC) $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ $(LIBS)

general-gdb: general.cu
	$(NVCC) -g -G $(CFLAGS) ${OPEN_FLAGS} ${MGPU_FLAGS} ${ARCH} $(INCLUDE) -o $@ $^ $(LIBS)

clean:
	rm triangle general