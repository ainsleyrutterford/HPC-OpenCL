# OpenCL, OpenMP, and MPI parallelisation

As part of the Advanced High Performance Computing unit at the University of Bristol, I parallelised a Lattice Boltzmann code using the OpenCL, OpenMP, and MPI libraries. All three implementations are written in C. I achieved the highest grade in the class and was awarded the Crazy prize. A report explaining the optimisations explored is available [here](https://github.com/ainsleyrutterford/HPC-OpenCL/blob/master/report.pdf). 

## About

The OpenCL, OpenMP, and MPI implementations are provided in the `opencl_d2q9-bgk.c`, `openmp_d2q9-bgk.c`, and `mpi_d2q9-bgk.c` files respectively. The kernels for the OpenCL implementation are provided in `kernels.cl`.

The job submission scripts for each implementation are provided in the `jobs/` directory.

## Usage

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
