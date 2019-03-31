#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <omp.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define MASTER 0

/* struct to hold the parameter values */
typedef struct {
  int   nx;             /* no. of cells in x-direction */
  int   ny;             /* no. of cells in y-direction */
  int   maxIters;       /* no. of iterations */
  int   reynolds_dim;   /* dimension for Reynolds number */
  float density;        /* density per link */
  float accel;          /* density redistribution */
  float omega;          /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float* speed0;
  float* speed1;
  float* speed2;
  float* speed3;
  float* speed4;
  float* speed5;
  float* speed6;
  float* speed7;
  float* speed8;
} t_speed;

int rank;
int left;
int right;
int size;
int tag = 0;
int local_nrows;
int local_ncols;
int remote_nrows;
MPI_Status status;
int total_cells;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               uint8_t** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, const uint8_t* restrict obstacles, int rank);
int write_values(const t_param params, t_speed cells, uint8_t* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             uint8_t** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed cells, uint8_t* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed cells, uint8_t* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void zero_image(t_speed* image, int cols, int rows);
int calc_nrows_from_rank(int rank, int size, int ny);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
  char*    paramfile    = NULL; /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed  cells;               /* grid containing fluid densities */
  t_speed  tmp_cells;           /* scratch space */
  uint8_t* obstacles = NULL;    /* grid indicating which cells are blocked */
  float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER) printf("Starting...\n");

  // parse the command line
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  if (rank == MASTER) {
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  }

  if (rank == MASTER) printf("Initialised...\n");

  left  = (rank - 1 + size) % size;
  right = (rank + 1)        % size;

  if (rank == MASTER) {
    for (int k = 1; k < size; k++) {
      MPI_Ssend(&params.nx, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.ny, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.maxIters, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.reynolds_dim, 1, MPI_INT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.density, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.accel, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
      MPI_Ssend(&params.omega, 1, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&params.nx, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.ny, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.maxIters, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.reynolds_dim, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.density, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.accel, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&params.omega, 1, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
  }

  local_nrows = calc_nrows_from_rank(rank, size, params.ny);
  local_ncols = params.nx;

  printf("rank %d local_nrows %d local_ncols %d\n", rank, local_nrows, local_ncols);

  uint8_t* local_obstacles = _mm_malloc(sizeof(uint8_t) * (local_ncols * local_nrows), 64);
  float* local_av_vels = _mm_malloc(sizeof(float) * params.maxIters, 64);

  t_speed local_cells;
  t_speed local_tmp_cells;

  local_cells.speed0 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed1 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed2 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed3 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed4 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed5 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed6 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed7 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_cells.speed8 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);

  local_tmp_cells.speed0 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed1 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed2 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed3 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed4 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed5 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed6 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed7 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);
  local_tmp_cells.speed8 = _mm_malloc(sizeof(float) * (local_ncols * (local_nrows + 2)), 64);

  zero_image(&local_cells    , local_ncols, local_nrows + 2);
  zero_image(&local_tmp_cells, local_ncols, local_nrows + 2);

  printf("Scattering...\n");

  for (int x = 0; x < params.nx; x++) {
    if (rank == MASTER) {
      for (int y = 1; y < (local_nrows + 2) - 1; y++) {
        local_cells.speed0[x + y*params.nx] = cells.speed0[x + (y-1)*params.nx];
        local_cells.speed1[x + y*params.nx] = cells.speed1[x + (y-1)*params.nx];
        local_cells.speed2[x + y*params.nx] = cells.speed2[x + (y-1)*params.nx];
        local_cells.speed3[x + y*params.nx] = cells.speed3[x + (y-1)*params.nx];
        local_cells.speed4[x + y*params.nx] = cells.speed4[x + (y-1)*params.nx];
        local_cells.speed5[x + y*params.nx] = cells.speed5[x + (y-1)*params.nx];
        local_cells.speed6[x + y*params.nx] = cells.speed6[x + (y-1)*params.nx];
        local_cells.speed7[x + y*params.nx] = cells.speed7[x + (y-1)*params.nx];
        local_cells.speed8[x + y*params.nx] = cells.speed8[x + (y-1)*params.nx];
        local_obstacles[x + (y-1)*params.nx] = obstacles[x + (y-1)*params.nx];
      }
      for (int k = 1; k < size; k++) {
        remote_nrows = calc_nrows_from_rank(k, size, params.ny);
        float* cells_sendbuf = (float*) malloc(sizeof(float) * remote_nrows * NSPEEDS);
        uint8_t* obstacles_sendbuf = (uint8_t*) malloc(sizeof(uint8_t) * remote_nrows);
        for (int y = 0; y < remote_nrows; y++) {
          cells_sendbuf[y + 0*remote_nrows] = cells.speed0[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 1*remote_nrows] = cells.speed1[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 2*remote_nrows] = cells.speed2[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 3*remote_nrows] = cells.speed3[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 4*remote_nrows] = cells.speed4[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 5*remote_nrows] = cells.speed5[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 6*remote_nrows] = cells.speed6[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 7*remote_nrows] = cells.speed7[x + (local_nrows * k + y) * params.nx];
          cells_sendbuf[y + 8*remote_nrows] = cells.speed8[x + (local_nrows * k + y) * params.nx];
          obstacles_sendbuf[y] = obstacles[x + (local_nrows * k + y) * params.nx];
        }
        MPI_Ssend(cells_sendbuf, remote_nrows * NSPEEDS, MPI_FLOAT, k, tag, MPI_COMM_WORLD);
        MPI_Ssend(obstacles_sendbuf, remote_nrows, MPI_UINT8_T, k, tag, MPI_COMM_WORLD);
      }
    } else {
      float* cells_recvbuf = (float*) malloc(sizeof(float) * local_nrows * NSPEEDS);
      uint8_t* obstacles_recvbuf = (uint8_t*) malloc(sizeof(uint8_t) * local_nrows);
      MPI_Recv(cells_recvbuf, local_nrows * NSPEEDS, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);
      MPI_Recv(obstacles_recvbuf, local_nrows, MPI_UINT8_T, MASTER, tag, MPI_COMM_WORLD, &status);
      for (int y = 1; y < (local_nrows + 2) - 1; y++) {
        local_cells.speed0[x + y*params.nx] = cells_recvbuf[y-1 + 0*remote_nrows];
        local_cells.speed1[x + y*params.nx] = cells_recvbuf[y-1 + 1*remote_nrows];
        local_cells.speed2[x + y*params.nx] = cells_recvbuf[y-1 + 2*remote_nrows];
        local_cells.speed3[x + y*params.nx] = cells_recvbuf[y-1 + 3*remote_nrows];
        local_cells.speed4[x + y*params.nx] = cells_recvbuf[y-1 + 4*remote_nrows];
        local_cells.speed5[x + y*params.nx] = cells_recvbuf[y-1 + 5*remote_nrows];
        local_cells.speed6[x + y*params.nx] = cells_recvbuf[y-1 + 6*remote_nrows];
        local_cells.speed7[x + y*params.nx] = cells_recvbuf[y-1 + 7*remote_nrows];
        local_cells.speed8[x + y*params.nx] = cells_recvbuf[y-1 + 8*remote_nrows];
        local_obstacles[x + (y-1)*params.nx] = obstacles_recvbuf[y-1];
      }
    }
  }

  if (rank == MASTER) printf("Finished scattering...\n");

  if (rank == MASTER) printf("Starting timesteps...\n");

  double local_tic = MPI_Wtime();

  for (int tt = 0; tt < params.maxIters; tt+=2) {
    local_av_vels[tt]   = timestep(params, &local_cells, &local_tmp_cells, local_obstacles, rank);
    local_av_vels[tt+1] = timestep(params, &local_tmp_cells, &local_cells, local_obstacles, rank);
    #ifdef DEBUG
      printf("==timestep: %d==\n", tt);
      printf("av velocity: %.12E\n", av_vels[tt]);
      printf("tot density: %.12E\n", total_density(params, cells));
    #endif
  }

  if (rank == MASTER) {
    for (int i = 0; i < params.maxIters; i++) {
      av_vels[i] += local_av_vels[i];
    }
    for (int k = 1; k < size; k++) {
      float* av_vels_recvbuf = malloc(sizeof(float) * params.maxIters);
      MPI_Recv(av_vels_recvbuf, params.maxIters, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
      for (int i = 0; i < params.maxIters; i++) {
        av_vels[i] += av_vels_recvbuf[i];
      }
    }
    for (int i = 0; i < params.maxIters; i++) {
      av_vels[i] /= total_cells;
    }
  } else {
    MPI_Send(local_av_vels, params.maxIters, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  double local_toc = MPI_Wtime();

  if (rank == MASTER) printf("Timesteps finished...\n");

  double global_tic, global_toc;
  MPI_Reduce(&local_tic, &global_tic, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_toc, &global_toc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == MASTER) printf("Receiving...\n");

  for (int x = 0; x < params.nx; x++) {
    if (rank == MASTER) {
      for (int y = 1; y < (local_nrows + 2) - 1; y++) {
        cells.speed0[x + (y-1)*params.nx] = local_cells.speed0[x + y*params.nx];
        cells.speed1[x + (y-1)*params.nx] = local_cells.speed1[x + y*params.nx];
        cells.speed2[x + (y-1)*params.nx] = local_cells.speed2[x + y*params.nx];
        cells.speed3[x + (y-1)*params.nx] = local_cells.speed3[x + y*params.nx];
        cells.speed4[x + (y-1)*params.nx] = local_cells.speed4[x + y*params.nx];
        cells.speed5[x + (y-1)*params.nx] = local_cells.speed5[x + y*params.nx];
        cells.speed6[x + (y-1)*params.nx] = local_cells.speed6[x + y*params.nx];
        cells.speed7[x + (y-1)*params.nx] = local_cells.speed7[x + y*params.nx];
        cells.speed8[x + (y-1)*params.nx] = local_cells.speed8[x + y*params.nx];
      }
      for (int k = 1; k < size; k++) {
        remote_nrows = calc_nrows_from_rank(k, size, params.ny);
        float* cells_recvbuf = (float*) malloc(sizeof(float) * remote_nrows * NSPEEDS);
        MPI_Recv(cells_recvbuf, remote_nrows * NSPEEDS, MPI_FLOAT, k, tag, MPI_COMM_WORLD, &status);
        for (int y = 0; y < remote_nrows; y++) {
          cells.speed0[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 0*remote_nrows];
          cells.speed1[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 1*remote_nrows];
          cells.speed2[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 2*remote_nrows];
          cells.speed3[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 3*remote_nrows];
          cells.speed4[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 4*remote_nrows];
          cells.speed5[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 5*remote_nrows];
          cells.speed6[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 6*remote_nrows];
          cells.speed7[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 7*remote_nrows];
          cells.speed8[x + (local_nrows * k + y) * params.nx] = cells_recvbuf[y + 8*remote_nrows];
        }
      }
    } else {
      float* cells_sendbuf = (float*) malloc(sizeof(float) * local_nrows * NSPEEDS);
      for (int y = 1; y < (local_nrows + 2) - 1; y++) {
        cells_sendbuf[y-1 + 0*remote_nrows] = local_cells.speed0[x + y*params.nx];
        cells_sendbuf[y-1 + 1*remote_nrows] = local_cells.speed1[x + y*params.nx];
        cells_sendbuf[y-1 + 2*remote_nrows] = local_cells.speed2[x + y*params.nx];
        cells_sendbuf[y-1 + 3*remote_nrows] = local_cells.speed3[x + y*params.nx];
        cells_sendbuf[y-1 + 4*remote_nrows] = local_cells.speed4[x + y*params.nx];
        cells_sendbuf[y-1 + 5*remote_nrows] = local_cells.speed5[x + y*params.nx];
        cells_sendbuf[y-1 + 6*remote_nrows] = local_cells.speed6[x + y*params.nx];
        cells_sendbuf[y-1 + 7*remote_nrows] = local_cells.speed7[x + y*params.nx];
        cells_sendbuf[y-1 + 8*remote_nrows] = local_cells.speed8[x + y*params.nx];
      }
      MPI_Ssend(cells_sendbuf, local_nrows * NSPEEDS, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }
  }

  if (rank == MASTER) printf("Received...\n");

  _mm_free(local_cells.speed0);
  _mm_free(local_cells.speed1);
  _mm_free(local_cells.speed2);
  _mm_free(local_cells.speed3);
  _mm_free(local_cells.speed4);
  _mm_free(local_cells.speed5);
  _mm_free(local_cells.speed6);
  _mm_free(local_cells.speed7);
  _mm_free(local_cells.speed8);
  _mm_free(local_tmp_cells.speed0);
  _mm_free(local_tmp_cells.speed1);
  _mm_free(local_tmp_cells.speed2);
  _mm_free(local_tmp_cells.speed3);
  _mm_free(local_tmp_cells.speed4);
  _mm_free(local_tmp_cells.speed5);
  _mm_free(local_tmp_cells.speed6);
  _mm_free(local_tmp_cells.speed7);
  _mm_free(local_tmp_cells.speed8);

  _mm_free(local_obstacles);
  _mm_free(local_av_vels);

  if (rank == MASTER) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", global_toc - global_tic);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  }

  MPI_Finalize();

}

float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, const uint8_t* restrict obstacles, int rank) {

  float* sendbuf = (float*) malloc(sizeof(float) * local_ncols * NSPEEDS);
  float* recvbuf = (float*) malloc(sizeof(float) * local_ncols * NSPEEDS);

  // send to the left, receive from the right
  for (int x = 0; x < local_ncols; x++) {
    sendbuf[x + 0*local_ncols] = cells->speed0[x + 1*local_ncols];
    sendbuf[x + 1*local_ncols] = cells->speed1[x + 1*local_ncols];
    sendbuf[x + 2*local_ncols] = cells->speed2[x + 1*local_ncols];
    sendbuf[x + 3*local_ncols] = cells->speed3[x + 1*local_ncols];
    sendbuf[x + 4*local_ncols] = cells->speed4[x + 1*local_ncols];
    sendbuf[x + 5*local_ncols] = cells->speed5[x + 1*local_ncols];
    sendbuf[x + 6*local_ncols] = cells->speed6[x + 1*local_ncols];
    sendbuf[x + 7*local_ncols] = cells->speed7[x + 1*local_ncols];
    sendbuf[x + 8*local_ncols] = cells->speed8[x + 1*local_ncols];
  }

  MPI_Sendrecv(sendbuf, local_ncols * NSPEEDS, MPI_FLOAT, left , tag,
               recvbuf, local_ncols * NSPEEDS, MPI_FLOAT, right, tag,
               MPI_COMM_WORLD, &status);

  for (int x = 0; x < local_ncols; x++) {
    cells->speed0[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 0*local_ncols];
    cells->speed1[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 1*local_ncols];
    cells->speed2[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 2*local_ncols];
    cells->speed3[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 3*local_ncols];
    cells->speed4[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 4*local_ncols];
    cells->speed5[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 5*local_ncols];
    cells->speed6[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 6*local_ncols];
    cells->speed7[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 7*local_ncols];
    cells->speed8[x + (local_nrows + 1)*local_ncols] = recvbuf[x + 8*local_ncols];
  }

  // send to the right, receive from the left
  for (int x = 0; x < local_ncols; x++) {
    sendbuf[x + 0*local_ncols] = cells->speed0[x + local_nrows*local_ncols];
    sendbuf[x + 1*local_ncols] = cells->speed1[x + local_nrows*local_ncols];
    sendbuf[x + 2*local_ncols] = cells->speed2[x + local_nrows*local_ncols];
    sendbuf[x + 3*local_ncols] = cells->speed3[x + local_nrows*local_ncols];
    sendbuf[x + 4*local_ncols] = cells->speed4[x + local_nrows*local_ncols];
    sendbuf[x + 5*local_ncols] = cells->speed5[x + local_nrows*local_ncols];
    sendbuf[x + 6*local_ncols] = cells->speed6[x + local_nrows*local_ncols];
    sendbuf[x + 7*local_ncols] = cells->speed7[x + local_nrows*local_ncols];
    sendbuf[x + 8*local_ncols] = cells->speed8[x + local_nrows*local_ncols];
  }

  MPI_Sendrecv(sendbuf, local_ncols * NSPEEDS, MPI_FLOAT, right, tag,
               recvbuf, local_ncols * NSPEEDS, MPI_FLOAT, left , tag,
               MPI_COMM_WORLD, &status);

  for (int x = 0; x < local_ncols; x++) {
    cells->speed0[x] = recvbuf[x + 0*local_ncols];
    cells->speed1[x] = recvbuf[x + 1*local_ncols];
    cells->speed2[x] = recvbuf[x + 2*local_ncols];
    cells->speed3[x] = recvbuf[x + 3*local_ncols];
    cells->speed4[x] = recvbuf[x + 4*local_ncols];
    cells->speed5[x] = recvbuf[x + 5*local_ncols];
    cells->speed6[x] = recvbuf[x + 6*local_ncols];
    cells->speed7[x] = recvbuf[x + 7*local_ncols];
    cells->speed8[x] = recvbuf[x + 8*local_ncols];
  }

  if (rank == size - 1) {

    // accelerate_flow

    // compute weighting factors
    const float init_w1 = params.density * params.accel / 9.f;
    const float init_w2 = params.density * params.accel / 36.f;

    // modify the 2nd row of the grid
    const int jj = local_nrows - 2;

    for (int ii = 0; ii < local_ncols; ii++) {
      // if the cell is not occupied and we don't send a negative density
      if (!obstacles[ii + jj*local_ncols]
      && (cells->speed3[ii + (jj+1)*local_ncols] - init_w1) > 0.f
      && (cells->speed6[ii + (jj+1)*local_ncols] - init_w2) > 0.f
      && (cells->speed7[ii + (jj+1)*local_ncols] - init_w2) > 0.f) {
        // increase 'east-side' densities
        cells->speed1[ii + (jj+1)*local_ncols] += init_w1;
        cells->speed5[ii + (jj+1)*local_ncols] += init_w2;
        cells->speed8[ii + (jj+1)*local_ncols] += init_w2;
        // decrease 'west-side' densities
        cells->speed3[ii + (jj+1)*local_ncols] -= init_w1;
        cells->speed6[ii + (jj+1)*local_ncols] -= init_w2;
        cells->speed7[ii + (jj+1)*local_ncols] -= init_w2;
      }
    }

  }

  const float c_sq = 1.f / 3.f;  // square of speed of sound
  const float w0   = 4.f / 9.f;  // weighting factor
  const float w1   = 1.f / 9.f;  // weighting factor
  const float w2   = 1.f / 36.f; // weighting factor
  const float denominator = 2.f * c_sq * c_sq;

  int   tot_cells = 0; // no. of cells used in calculation
  float tot_u = 0.f;   // accumulated magnitudes of velocity for each cell

  for (int jj = 1; jj < local_nrows + 1; jj++) {
    for (int ii = 0; ii < local_ncols; ii++) {

      const int y_n = jj + 1;
      const int x_e = (ii + 1) % local_ncols;
      const int y_s = (jj - 1);
      const int x_w = (ii == 0) ? (ii + local_ncols - 1) : (ii - 1);

      const float speed0 = cells->speed0[ii + jj*local_ncols];
      const float speed1 = cells->speed1[x_w + jj*local_ncols];
      const float speed2 = cells->speed2[ii + y_s*local_ncols];
      const float speed3 = cells->speed3[x_e + jj*local_ncols];
      const float speed4 = cells->speed4[ii + y_n*local_ncols];
      const float speed5 = cells->speed5[x_w + y_s*local_ncols];
      const float speed6 = cells->speed6[x_e + y_s*local_ncols];
      const float speed7 = cells->speed7[x_e + y_n*local_ncols];
      const float speed8 = cells->speed8[x_w + y_n*local_ncols];

      // compute local density total
      const float local_density = speed0 + speed1 + speed2
                                + speed3 + speed4 + speed5
                                + speed6 + speed7 + speed8;

      // compute x and y velocity components
      const float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) / local_density;
      const float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) / local_density;

      // if the cell contains an obstacle
      if (obstacles[(jj-1)*local_ncols + ii]) {

        tmp_cells->speed0[ii + jj*local_ncols] = speed0;
        tmp_cells->speed1[ii + jj*local_ncols] = speed3;
        tmp_cells->speed2[ii + jj*local_ncols] = speed4;
        tmp_cells->speed3[ii + jj*local_ncols] = speed1;
        tmp_cells->speed4[ii + jj*local_ncols] = speed2;
        tmp_cells->speed5[ii + jj*local_ncols] = speed7;
        tmp_cells->speed6[ii + jj*local_ncols] = speed8;
        tmp_cells->speed7[ii + jj*local_ncols] = speed5;
        tmp_cells->speed8[ii + jj*local_ncols] = speed6;

      } else {

        const float constant = 1.f - (u_x * u_x + u_y * u_y) * 1.5f;

        // directional velocity components
        const float u1 =   u_x;        // east
        const float u2 =         u_y;  // north
        const float u3 = - u_x;        // west
        const float u4 =       - u_y;  // south
        const float u5 =   u_x + u_y;  // north-east
        const float u6 = - u_x + u_y;  // north-west
        const float u7 = - u_x - u_y;  // south-west
        const float u8 =   u_x - u_y;  // south-east

        // relaxation step
        tmp_cells->speed0[ii + jj*local_ncols] = speed0 + params.omega * (w0 * local_density * constant - speed0);
        tmp_cells->speed1[ii + jj*local_ncols] = speed1 + params.omega * (w1 * local_density * (u1 / c_sq + (u1 * u1) / denominator + constant) - speed1);
        tmp_cells->speed2[ii + jj*local_ncols] = speed2 + params.omega * (w1 * local_density * (u2 / c_sq + (u2 * u2) / denominator + constant) - speed2);
        tmp_cells->speed3[ii + jj*local_ncols] = speed3 + params.omega * (w1 * local_density * (u3 / c_sq + (u3 * u3) / denominator + constant) - speed3);
        tmp_cells->speed4[ii + jj*local_ncols] = speed4 + params.omega * (w1 * local_density * (u4 / c_sq + (u4 * u4) / denominator + constant) - speed4);
        tmp_cells->speed5[ii + jj*local_ncols] = speed5 + params.omega * (w2 * local_density * (u5 / c_sq + (u5 * u5) / denominator + constant) - speed5);
        tmp_cells->speed6[ii + jj*local_ncols] = speed6 + params.omega * (w2 * local_density * (u6 / c_sq + (u6 * u6) / denominator + constant) - speed6);
        tmp_cells->speed7[ii + jj*local_ncols] = speed7 + params.omega * (w2 * local_density * (u7 / c_sq + (u7 * u7) / denominator + constant) - speed7);
        tmp_cells->speed8[ii + jj*local_ncols] = speed8 + params.omega * (w2 * local_density * (u8 / c_sq + (u8 * u8) / denominator + constant) - speed8);
      }

      // accumulate the norm of x- and y- velocity components
      tot_u += (obstacles[(jj-1)*local_ncols + ii]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
    }
  }

  return tot_u;
}

float av_velocity(const t_param params, t_speed cells, uint8_t* obstacles) {
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx]) {
        /* local density total */
        float local_density = cells.speed0[ii + jj*params.nx]
                            + cells.speed1[ii + jj*params.nx]
                            + cells.speed2[ii + jj*params.nx]
                            + cells.speed3[ii + jj*params.nx]
                            + cells.speed4[ii + jj*params.nx]
                            + cells.speed5[ii + jj*params.nx]
                            + cells.speed6[ii + jj*params.nx]
                            + cells.speed7[ii + jj*params.nx]
                            + cells.speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (cells.speed1[ii + jj*params.nx]
                   + cells.speed5[ii + jj*params.nx]
                   + cells.speed8[ii + jj*params.nx]
                  - (cells.speed3[ii + jj*params.nx]
                   + cells.speed6[ii + jj*params.nx]
                   + cells.speed7[ii + jj*params.nx]))
                   / local_density;
        /* compute y velocity component */
        float u_y = (cells.speed2[ii + jj*params.nx]
                   + cells.speed5[ii + jj*params.nx]
                   + cells.speed6[ii + jj*params.nx]
                  - (cells.speed4[ii + jj*params.nx]
                   + cells.speed7[ii + jj*params.nx]
                   + cells.speed8[ii + jj*params.nx]))
                   / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float) tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
               uint8_t** obstacles_ptr, float** av_vels_ptr) {
  char  message[1024]; /* message buffer */
  FILE* fp;            /* file pointer */
  int   xx, yy;        /* generic array indices */
  int   blocked;       /* indicates whether a cell is blocked by an obstacle */
  int   retval;        /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
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

  /* and close up the file */
  fclose(fp);

  cells_ptr->speed0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  cells_ptr->speed8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  tmp_cells_ptr->speed0 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed1 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed2 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed3 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed4 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed5 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed6 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed7 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  tmp_cells_ptr->speed8 = _mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int8_t) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density       / 9.f;
  float w2 = params->density       / 36.f;

  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      /* centre */
      cells_ptr->speed0[ii + jj*params->nx] = w0;
      /* axis directions */
      cells_ptr->speed1[ii + jj*params->nx] = w1;
      cells_ptr->speed2[ii + jj*params->nx] = w1;
      cells_ptr->speed3[ii + jj*params->nx] = w1;
      cells_ptr->speed4[ii + jj*params->nx] = w1;
      /* diagonals */
      cells_ptr->speed5[ii + jj*params->nx] = w2;
      cells_ptr->speed6[ii + jj*params->nx] = w2;
      cells_ptr->speed7[ii + jj*params->nx] = w2;
      cells_ptr->speed8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  total_cells = params->nx * params->ny;

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;

    total_cells--;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*) _mm_malloc(sizeof(float) * params->maxIters, 64);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed* cells_ptr, t_speed* tmp_cells_ptr,
             uint8_t** obstacles_ptr, float** av_vels_ptr) {
  // free up allocated memory
  _mm_free(cells_ptr->speed0);
  _mm_free(cells_ptr->speed1);
  _mm_free(cells_ptr->speed2);
  _mm_free(cells_ptr->speed3);
  _mm_free(cells_ptr->speed4);
  _mm_free(cells_ptr->speed5);
  _mm_free(cells_ptr->speed6);
  _mm_free(cells_ptr->speed7);
  _mm_free(cells_ptr->speed8);

  _mm_free(tmp_cells_ptr->speed0);
  _mm_free(tmp_cells_ptr->speed1);
  _mm_free(tmp_cells_ptr->speed2);
  _mm_free(tmp_cells_ptr->speed3);
  _mm_free(tmp_cells_ptr->speed4);
  _mm_free(tmp_cells_ptr->speed5);
  _mm_free(tmp_cells_ptr->speed6);
  _mm_free(tmp_cells_ptr->speed7);
  _mm_free(tmp_cells_ptr->speed8);

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed cells, uint8_t* obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed cells) {
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      total = cells.speed0[ii + jj*params.nx]
            + cells.speed1[ii + jj*params.nx]
            + cells.speed2[ii + jj*params.nx]
            + cells.speed3[ii + jj*params.nx]
            + cells.speed4[ii + jj*params.nx]
            + cells.speed5[ii + jj*params.nx]
            + cells.speed6[ii + jj*params.nx]
            + cells.speed7[ii + jj*params.nx]
            + cells.speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed cells, uint8_t* obstacles, float* av_vels) {
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx]) {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      } else { /* no obstacle */
        local_density = 0.f;

        local_density = cells.speed0[ii + jj*params.nx]
                      + cells.speed1[ii + jj*params.nx]
                      + cells.speed2[ii + jj*params.nx]
                      + cells.speed3[ii + jj*params.nx]
                      + cells.speed4[ii + jj*params.nx]
                      + cells.speed5[ii + jj*params.nx]
                      + cells.speed6[ii + jj*params.nx]
                      + cells.speed7[ii + jj*params.nx]
                      + cells.speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (cells.speed1[ii + jj*params.nx]
                   + cells.speed5[ii + jj*params.nx]
                   + cells.speed8[ii + jj*params.nx]
                  - (cells.speed3[ii + jj*params.nx]
                   + cells.speed6[ii + jj*params.nx]
                   + cells.speed7[ii + jj*params.nx]))
                   / local_density;
        /* compute y velocity component */
        float u_y = (cells.speed2[ii + jj*params.nx]
                   + cells.speed5[ii + jj*params.nx]
                   + cells.speed6[ii + jj*params.nx]
                  - (cells.speed4[ii + jj*params.nx]
                   + cells.speed7[ii + jj*params.nx]
                   + cells.speed8[ii + jj*params.nx]))
                   / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
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

void zero_image(t_speed* image, int cols, int rows) {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            image->speed0[x + y*cols] = 0;
            image->speed1[x + y*cols] = 0;
            image->speed2[x + y*cols] = 0;
            image->speed3[x + y*cols] = 0;
            image->speed4[x + y*cols] = 0;
            image->speed5[x + y*cols] = 0;
            image->speed6[x + y*cols] = 0;
            image->speed7[x + y*cols] = 0;
            image->speed8[x + y*cols] = 0;
        }
    }
}

int calc_nrows_from_rank(int rank, int size, int ny) {
    int nrows;

    nrows = ny / size;                             // integer division
    if ((ny % size) != 0) {                        // if there is a remainder
        if (rank == size - 1) nrows += ny % size;  // add remainder to last rank
    }

    return nrows;
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
