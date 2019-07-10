# Makefile

EXE=opencl_d2q9-bgk

CC=icc
CFLAGS= -std=c99 -Wall -O3
LIBS = -lm

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS += -framework OpenCL
else
	LIBS += -lOpenCL
endif

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
	rm -f mpi_d2q9-bgk
	rm -f openmp_d2q9-bgk

mpi:
	mpiicc -std=c99 -Wall mpi_d2q9-bgk.c -o mpi_d2q9-bgk -Ofast -xHOST -O3 -lm

openmp:
	clang -std=c99 -Wall openmp_d2q9-bgk.c -o openmp_d2q9-bgk -fopenmp -lm -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_35 -O3
