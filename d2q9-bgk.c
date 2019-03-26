#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"
#define LOCAL_SIZE      16

// struct to hold the parameter values
typedef struct {
  int   nx;            // no. of cells in x-direction
  int   ny;            // no. of cells in y-direction
  int   maxIters;      // no. of iterations
  int   reynolds_dim;  // dimension for Reynolds number
  float density;       // density per link
  float accel;         // density redistribution
  float omega;         // relaxation parameter
  int   tot_cells;     // total unblocked cells
} t_param;

// struct to hold OpenCL objects
typedef struct {
  cl_device_id     device;
  cl_context       context;
  cl_command_queue queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  accelerate_flow_flipped;
  cl_kernel  computation;
  cl_kernel  computation_flipped;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;

  cl_mem velocities;

  size_t local_size;
  int    work_groups;
} t_ocl;

// function prototypes

// load params, allocate memory, load obstacles & initialise fluid particle densities
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

float timestep(const t_param params, t_ocl ocl, float* partial_velocities, bool regular);
int accelerate_flow(const t_param params, t_ocl ocl, bool regular);
float computation(const t_param params, t_ocl ocl, float* partial_velocities, bool regular);
int write_values(const t_param params, float* cells, int* obstacles, float* av_vels);

// finalise, including freeing up allocated memory
int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

// Sum all the densities in the grid.
// The total should remain constant from one timestep to the next.
float total_density(const t_param params, float* cells);

// compute average velocity
float av_velocity(const t_param params, float* cells, int* obstacles, t_ocl ocl);

// calculate Reynolds number
float calc_reynolds(const t_param params, float* cells, int* obstacles, t_ocl ocl);

// utility functions
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

int main(int argc, char* argv[]) {
  char*   paramfile = NULL;    // name of the input parameter file
  char*   obstaclefile = NULL; // name of a the input obstacle file
  t_param params;              // struct to hold parameter values
  t_ocl   ocl;                 // struct to hold OpenCL objects
  float*  cells     = NULL;    // grid containing fluid densities
  float*  tmp_cells = NULL;    // scratch space
  int*    obstacles = NULL;    // grid indicating which cells are blocked
  float*  av_vels   = NULL;    // a record of the av. velocity computed for each timestep
  cl_int  err;
  struct  timeval timstr;      // structure to hold elapsed time
  struct  rusage ru;           // structure to hold CPU time--system and user
  double  tic, toc;            // floating point numbers to calculate elapsed wallclock time
  double  usrtim;              // floating point number to record elapsed user CPU time
  double  systim;              // floating point number to record elapsed system CPU time

  // parse the command line
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  // initialise our data structures and load values from file
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);

  float* partial_velocities = calloc(ocl.work_groups, sizeof(float));
  if (!partial_velocities) {
    printf("Error: could not allocate host memory for partial_velocities\n");
    return EXIT_FAILURE;
  }

  // iterate for maxIters timesteps
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                             sizeof(float) * params.nx * params.ny * NSPEEDS,
                             cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(ocl.queue, ocl.obstacles, CL_TRUE, 0,
                             sizeof(cl_int) * params.nx * params.ny,
                             obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);

  // Enqueue kernel
  size_t accelerate_global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, accelerate_global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  for (int tt = 0; tt < params.maxIters; tt+=2) {
    av_vels[tt]   = timestep(params, ocl, partial_velocities, true);
    av_vels[tt+1] = timestep(params, ocl, partial_velocities, false);
    #ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    #endif
  }

  // Read tmp_cells from device
  err = clEnqueueReadBuffer(ocl.queue, ocl.tmp_cells, CL_TRUE, 0,
                            sizeof(float) * params.nx * params.ny * NSPEEDS,
                            tmp_cells, 0, NULL, NULL);
  checkError(err, "reading tmp_cells data", __LINE__);

  // Read cells from device
  err = clEnqueueReadBuffer(ocl.queue, ocl.cells, CL_TRUE, 0,
                            sizeof(float) * params.nx * params.ny * NSPEEDS,
                            cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // write final values and free memory
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_ocl ocl, float* partial_velocities, bool regular) {
  cl_int err;

  // Enqueue kernel
  size_t computation_global[2] = {params.nx, params.ny};
  size_t local[2] = {ocl.local_size, ocl.local_size};
  err = clEnqueueNDRangeKernel(ocl.queue,
                               (regular) ? ocl.computation : ocl.computation_flipped,
                               2, NULL, computation_global, local, 0, NULL, NULL);
  checkError(err, "enqueueing computation kernel", __LINE__);

  // Read velocities from device
  err = clEnqueueReadBuffer(ocl.queue, ocl.velocities, CL_TRUE, 0,
                            sizeof(float) * ocl.work_groups,
                            partial_velocities, 0, NULL, NULL);
  checkError(err, "reading velocities data", __LINE__);

  // Wait for kernel to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for computation kernel", __LINE__);

  float total_velocity = 0.f;
  for (int i = 0; i < ocl.work_groups; i++) {
    total_velocity += partial_velocities[i];
  }

  return total_velocity / (float) params.tot_cells;
}

float av_velocity(const t_param params, float* cells, int* obstacles, t_ocl ocl) {
  int   tot_cells = 0;   // no. of cells used in calculation
  float tot_u     = 0.f; // accumulated magnitudes of velocity for each cell

  // loop over all non-blocked cells
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      // ignore occupied cells
      if (!obstacles[ii + jj*params.nx]) {
        // local density total
        float local_density = cells[(0 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
                            + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)];

        // compute x velocity component
        float u_x = (cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]
                  - (cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]))
                   / local_density;
        // compute y velocity component
        float u_y = (cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
                  - (cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
                   + cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]))
                   / local_density;
        // accumulate the norm of x- and y- velocity components
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        // increase counter of inspected cells
        ++tot_cells;
      }
    }
  }

return tot_u / (float) tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl) {
  char  message[1024]; // message buffer
  FILE* fp;            // file pointer
  int   xx, yy;        // generic array indices
  int   blocked;       // indicates whether a cell is blocked by an obstacle
  int   retval;        // to hold return value for checking
  char* ocl_src;       // OpenCL kernel source
  long  ocl_size;      // size of OpenCL kernel source

  // open the parameter file
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  // read in the parameter values
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  // and close up the fil/
  fclose(fp);

  // main grid
  *cells_ptr = malloc(sizeof(float) * NSPEEDS * params->ny * params->nx);
  if (*cells_ptr == NULL) die("cannot allocate memory for cell speeds", __LINE__, __FILE__);

  // 'helper' grid, used as scratch space
  *tmp_cells_ptr = malloc(sizeof(float) * NSPEEDS * params->ny * params->nx);
  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cell speeds", __LINE__, __FILE__);

  // the map of obstacles
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  // initialise densities
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density       / 9.f;
  float w2 = params->density       / 36.f;

  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      // centre
      (*cells_ptr)[(0 * params->ny * params->nx) + (ii + jj*params->nx)] = w0;
      // axis directions
      (*cells_ptr)[(1 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(2 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(3 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      (*cells_ptr)[(4 * params->ny * params->nx) + (ii + jj*params->nx)] = w1;
      // diagonals
      (*cells_ptr)[(5 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(6 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(7 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
      (*cells_ptr)[(8 * params->ny * params->nx) + (ii + jj*params->nx)] = w2;
    }
  }

  // first set all cells in obstacle array to zero
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  // open the obstacle data file
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  params->tot_cells = params->nx * params->ny;

  // read-in the blocked cells list
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    // some checks
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    // assign to array
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;

    params->tot_cells--;
  }

  // and close the file
  fclose(fp);

  // allocate space to hold a record of the avarage velocities computed at each timestep
  *av_vels_ptr = (float*) malloc(sizeof(float) * params->maxIters);

  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL) {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*) malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**) &ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t sz;
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->computation = clCreateKernel(ocl->program, "computation", &err);
  checkError(err, "creating computation kernel", __LINE__);

  ocl->accelerate_flow_flipped = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow_flipped kernel", __LINE__);
  ocl->computation_flipped = clCreateKernel(ocl->program, "computation", &err);
  checkError(err, "creating computation_flipped kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                              sizeof(float) * params->nx * params->ny * NSPEEDS, NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                  sizeof(float) * params->nx * params->ny * NSPEEDS, NULL, &err);
  checkError(err, "creating tmp_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                  sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  ocl->local_size = LOCAL_SIZE;
  ocl->work_groups = (params->nx * params->ny) / (ocl->local_size * ocl->local_size);

  ocl->velocities = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * ocl->work_groups, NULL, &err);
  checkError(err, "creating velocities buffer", __LINE__);

  err = clSetKernelArg(ocl->accelerate_flow, 0, sizeof(cl_mem), &ocl->cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow, 1, sizeof(cl_mem), &ocl->obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow, 2, sizeof(cl_int), &params->nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow, 3, sizeof(cl_int), &params->ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow, 4, sizeof(cl_float), &params->density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow, 5, sizeof(cl_float), &params->accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  err = clSetKernelArg(ocl->computation, 0, sizeof(cl_mem), &ocl->cells);
  checkError(err, "setting computation arg 0", __LINE__);
  err = clSetKernelArg(ocl->computation, 1, sizeof(cl_mem), &ocl->tmp_cells);
  checkError(err, "setting computation arg 1", __LINE__);
  err = clSetKernelArg(ocl->computation, 2, sizeof(cl_mem), &ocl->obstacles);
  checkError(err, "setting computation arg 2", __LINE__);
  err = clSetKernelArg(ocl->computation, 3, sizeof(cl_int), &params->nx);
  checkError(err, "setting computation arg 3", __LINE__);
  err = clSetKernelArg(ocl->computation, 4, sizeof(cl_int), &params->ny);
  checkError(err, "setting computation arg 4", __LINE__);
  err = clSetKernelArg(ocl->computation, 5, sizeof(cl_float), &params->omega);
  checkError(err, "setting computation arg 5", __LINE__);
  err = clSetKernelArg(ocl->computation, 6, sizeof(cl_mem), &ocl->velocities);
  checkError(err, "setting computation arg 6", __LINE__);
  err = clSetKernelArg(ocl->computation, 7, sizeof(float) * ocl->local_size * ocl->local_size, NULL);
  checkError(err, "setting computation arg 7", __LINE__);
  err = clSetKernelArg(ocl->computation, 8, sizeof(cl_float), &params->density);
  checkError(err, "setting computation arg 8", __LINE__);
  err = clSetKernelArg(ocl->computation, 9, sizeof(cl_float), &params->accel);
  checkError(err, "setting computation arg 9", __LINE__);

  err = clSetKernelArg(ocl->accelerate_flow_flipped, 0, sizeof(cl_mem), &ocl->tmp_cells);
  checkError(err, "setting accelerate_flow_flipped arg 0", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow_flipped, 1, sizeof(cl_mem), &ocl->obstacles);
  checkError(err, "setting accelerate_flow_flipped arg 1", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow_flipped, 2, sizeof(cl_int), &params->nx);
  checkError(err, "setting accelerate_flow_flipped arg 2", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow_flipped, 3, sizeof(cl_int), &params->ny);
  checkError(err, "setting accelerate_flow_flipped arg 3", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow_flipped, 4, sizeof(cl_float), &params->density);
  checkError(err, "setting accelerate_flow_flipped arg 4", __LINE__);
  err = clSetKernelArg(ocl->accelerate_flow_flipped, 5, sizeof(cl_float), &params->accel);
  checkError(err, "setting accelerate_flow_flipped arg 5", __LINE__);

  err = clSetKernelArg(ocl->computation_flipped, 0, sizeof(cl_mem), &ocl->tmp_cells);
  checkError(err, "setting computation_flipped arg 0", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 1, sizeof(cl_mem), &ocl->cells);
  checkError(err, "setting computation_flipped arg 1", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 2, sizeof(cl_mem), &ocl->obstacles);
  checkError(err, "setting computation_flipped arg 2", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 3, sizeof(cl_int), &params->nx);
  checkError(err, "setting computation_flipped arg 3", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 4, sizeof(cl_int), &params->ny);
  checkError(err, "setting computation_flipped arg 4", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 5, sizeof(cl_float), &params->omega);
  checkError(err, "setting computation_flipped arg 5", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 6, sizeof(cl_mem), &ocl->velocities);
  checkError(err, "setting computation_flipped arg 6", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 7, sizeof(float) * ocl->local_size * ocl->local_size, NULL);
  checkError(err, "setting computation_flipped arg 7", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 8, sizeof(cl_float), &params->density);
  checkError(err, "setting computation_flipped arg 8", __LINE__);
  err = clSetKernelArg(ocl->computation_flipped, 9, sizeof(cl_float), &params->accel);
  checkError(err, "setting computation_flipped arg 9", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl) {
  // free up allocated memory
  free(*cells_ptr);
  free(*tmp_cells_ptr);
  free(*obstacles_ptr);
  free(*av_vels_ptr);
  *cells_ptr     = NULL;
  *tmp_cells_ptr = NULL;
  *obstacles_ptr = NULL;
  *av_vels_ptr   = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.computation);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, float* cells, int* obstacles, t_ocl ocl) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles, ocl) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* cells) {
  float total = 0.f;  // accumulator

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      for (int kk = 0; kk < NSPEEDS; kk++) {
        total += cells[(kk * params.ny * params.nx) + (ii + jj*params.nx)];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float* cells, int* obstacles, float* av_vels) {
  FILE* fp;                     // file pointer
  const float c_sq = 1.f / 3.f; // sq. of speed of sound
  float local_density;          // per grid cell sum of densities
  float pressure;               // fluid pressure in grid cell
  float u_x;                    // x-component of velocity in grid cell
  float u_y;                    // y-component of velocity in grid cell
  float u;                      // norm--root of summed squares--of u_x and u_y

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      // an occupied cell
      if (obstacles[ii + jj*params.nx]) {

        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;

      } else { // no obstacle

        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += cells[(kk * params.ny * params.nx) + (ii + jj*params.nx)];
        }

        // compute x velocity component
        u_x = (cells[(1 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]
            - (cells[(3 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]))
            /  local_density;
        // compute y velocity component
        u_y = (cells[(2 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(5 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(6 * params.ny * params.nx) + (ii + jj*params.nx)]
            - (cells[(4 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(7 * params.ny * params.nx) + (ii + jj*params.nx)]
            +  cells[(8 * params.ny * params.nx) + (ii + jj*params.nx)]))
            /  local_density;
        // compute norm of velocity
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        // compute pressure
        pressure = local_density * c_sq;
      }

      // write to file
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice() {
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++) {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++) {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env) {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices) {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}
