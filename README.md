# OpenCL, OpenMP, and MPI parallelisation

As part of the Advanced High Performance Computing unit at the University of Bristol, I parallelised a Lattice Boltzmann code using the OpenCL, OpenMP, and MPI libraries. All three implementations are written in C.

## About

## Usage

### Prerequisits

### OpenCL

``` shell
$ make clean
$ make
$ ./opencl_d2q9-bgk input/input_128x128.params input/obstacles_128x128.dat
```

Job scripts for all implementations are also provided for the PBS Professional job scheduler. They are located in the `jobs/` directory. The program can be compiled as usual and then submitted to the scheduler instead using `qsub opencl_submit`

### OpenMP

``` shell
$ make clean
$ make openmp
$ ./openmp_d2q9-bgk input/input_128x128.params input/obstacles_128x128.dat
```

Or use the job script `qsub openmp_submit`

### MPI

``` shell
$ make clean
$ make mpi
$ mpirun -np 16 mpi_d2q9-bgk input/input_128x128.params input/obstacles_128x128.dat
```

Or use the job script `qsub mpi_submit`