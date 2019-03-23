#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

kernel void computation(global float* cells,
                        global float* tmp_cells,
                        global int*   obstacles,
                        int nx, int ny,
                        float omega,
                        global float* global_velocities,
                        local  float* local_velocities) {

  float c_sq = 1.f / 3.f;  /* square of speed of sound */
  float w0   = 4.f / 9.f;  /* weighting factor */
  float w1   = 1.f / 9.f;  /* weighting factor */
  float w2   = 1.f / 36.f; /* weighting factor */
  float denominator = 2.f * c_sq * c_sq;

  float tot_u = 0.f;   /* accumulated magnitudes of velocity for each cell */

  int local_id_x = get_local_id(0);
  int local_id_y = get_local_id(1);

  int local_size_x = get_local_size(0);
  int local_size_y = get_local_size(1);

  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int y_n = (jj + 1) & (ny - 1);
  int x_e = (ii + 1) & (nx - 1);
  int y_s = (jj + ny - 1) & (ny - 1);
  int x_w = (ii + nx - 1) & (nx - 1);

  float speed0 = cells[(0 * ny * nx) + (ii + jj*nx)];
  float speed1 = cells[(1 * ny * nx) + (x_w + jj*nx)];
  float speed2 = cells[(2 * ny * nx) + (ii + y_s*nx)];
  float speed3 = cells[(3 * ny * nx) + (x_e + jj*nx)];
  float speed4 = cells[(4 * ny * nx) + (ii + y_n*nx)];
  float speed5 = cells[(5 * ny * nx) + (x_w + y_s*nx)];
  float speed6 = cells[(6 * ny * nx) + (x_e + y_s*nx)];
  float speed7 = cells[(7 * ny * nx) + (x_e + y_n*nx)];
  float speed8 = cells[(8 * ny * nx) + (x_w + y_n*nx)];

  /* compute local density total */
  float local_density = speed0 + speed1 + speed2
                      + speed3 + speed4 + speed5
                      + speed6 + speed7 + speed8;

  /* compute x and y velocity components */
  float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) / local_density;
  float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) / local_density;

  /* if the cell contains an obstacle */
  if (obstacles[jj*nx + ii]) {

    tmp_cells[(0 * ny * nx) + (ii + jj* nx)] = speed0;
    tmp_cells[(1 * ny * nx) + (ii + jj* nx)] = speed3;
    tmp_cells[(2 * ny * nx) + (ii + jj* nx)] = speed4;
    tmp_cells[(3 * ny * nx) + (ii + jj* nx)] = speed1;
    tmp_cells[(4 * ny * nx) + (ii + jj* nx)] = speed2;
    tmp_cells[(5 * ny * nx) + (ii + jj* nx)] = speed7;
    tmp_cells[(6 * ny * nx) + (ii + jj* nx)] = speed8;
    tmp_cells[(7 * ny * nx) + (ii + jj* nx)] = speed5;
    tmp_cells[(8 * ny * nx) + (ii + jj* nx)] = speed6;

  } else {

    float const_val = 1.f - (u_x * u_x + u_y * u_y) * 1.5f;

    /* directional velocity components */
    float u1 =   u_x;        /* east */
    float u2 =         u_y;  /* north */
    float u3 = - u_x;        /* west */
    float u4 =       - u_y;  /* south */
    float u5 =   u_x + u_y;  /* north-east */
    float u6 = - u_x + u_y;  /* north-west */
    float u7 = - u_x - u_y;  /* south-west */
    float u8 =   u_x - u_y;  /* south-east */

    /* relaxation step */
    tmp_cells[(0 * ny * nx) + (ii + jj* nx)] = speed0 + omega * (w0 * local_density * const_val - speed0);
    tmp_cells[(1 * ny * nx) + (ii + jj* nx)] = speed1 + omega * (w1 * local_density * (u1 / c_sq + (u1 * u1) / denominator + const_val) - speed1);
    tmp_cells[(2 * ny * nx) + (ii + jj* nx)] = speed2 + omega * (w1 * local_density * (u2 / c_sq + (u2 * u2) / denominator + const_val) - speed2);
    tmp_cells[(3 * ny * nx) + (ii + jj* nx)] = speed3 + omega * (w1 * local_density * (u3 / c_sq + (u3 * u3) / denominator + const_val) - speed3);
    tmp_cells[(4 * ny * nx) + (ii + jj* nx)] = speed4 + omega * (w1 * local_density * (u4 / c_sq + (u4 * u4) / denominator + const_val) - speed4);
    tmp_cells[(5 * ny * nx) + (ii + jj* nx)] = speed5 + omega * (w2 * local_density * (u5 / c_sq + (u5 * u5) / denominator + const_val) - speed5);
    tmp_cells[(6 * ny * nx) + (ii + jj* nx)] = speed6 + omega * (w2 * local_density * (u6 / c_sq + (u6 * u6) / denominator + const_val) - speed6);
    tmp_cells[(7 * ny * nx) + (ii + jj* nx)] = speed7 + omega * (w2 * local_density * (u7 / c_sq + (u7 * u7) / denominator + const_val) - speed7);
    tmp_cells[(8 * ny * nx) + (ii + jj* nx)] = speed8 + omega * (w2 * local_density * (u8 / c_sq + (u8 * u8) / denominator + const_val) - speed8);
  }

  /* accumulate the norm of x- and y- velocity components */
  tot_u = (obstacles[jj*nx + ii]) ? 0 : sqrt((u_x * u_x) + (u_y * u_y));

  local_velocities[local_id_x + local_id_y * local_size_x] = tot_u;

  int group_id_x    = get_group_id(0);
  int group_id_y    = get_group_id(1);

  int work_groups_x = get_global_size(0) / local_size_x;

  uint group_size = local_size_x * local_size_y;

  for (uint stride = group_size / 2; stride > 0; stride /= 2) {

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((local_id_x + local_id_y * local_size_x) < stride) {
      local_velocities[local_id_x + local_id_y * local_size_x] +=
      local_velocities[(local_id_x + local_id_y * local_size_x) + stride];
    }
  }

  if (local_id_x == 0 && local_id_y == 0) {
    global_velocities[group_id_x + group_id_y * work_groups_x] = local_velocities[0];
  }
}

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel) {
  // compute weighting factors
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  // modify the 2nd row of the grid
  int jj = ny - 2;

  // get column index
  int ii = get_global_id(0);

  // if the cell is not occupied and we don't send a negative density
  if (!obstacles[ii + jj* nx]
      && (cells[(3 * ny * nx) + (ii + jj* nx)] - w1) > 0.f
      && (cells[(6 * ny * nx) + (ii + jj* nx)] - w2) > 0.f
      && (cells[(7 * ny * nx) + (ii + jj* nx)] - w2) > 0.f) {
    // increase 'east-side' densities
    cells[(1 * ny * nx) + (ii + jj* nx)] += w1;
    cells[(5 * ny * nx) + (ii + jj* nx)] += w2;
    cells[(8 * ny * nx) + (ii + jj* nx)] += w2;
    // decrease 'west-side' densities
    cells[(3 * ny * nx) + (ii + jj* nx)] -= w1;
    cells[(6 * ny * nx) + (ii + jj* nx)] -= w2;
    cells[(7 * ny * nx) + (ii + jj* nx)] -= w2;
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny) {
  // get column and row indices
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  // determine indices of axis-direction neighbours
  // respecting periodic boundary conditions (wrap around)
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  // propagate densities from neighbouring cells, following
  // appropriate directions of travel and writing into
  // scratch space grid
  tmp_cells[(0 * ny * nx) + (ii + jj* nx)] = cells[(0 * ny * nx) + (ii + jj*nx)];   // central cell, no movement
  tmp_cells[(1 * ny * nx) + (ii + jj* nx)] = cells[(1 * ny * nx) + (x_w + jj*nx)];  // east
  tmp_cells[(2 * ny * nx) + (ii + jj* nx)] = cells[(2 * ny * nx) + (ii + y_s*nx)];  // north
  tmp_cells[(3 * ny * nx) + (ii + jj* nx)] = cells[(3 * ny * nx) + (x_e + jj*nx)];  // west
  tmp_cells[(4 * ny * nx) + (ii + jj* nx)] = cells[(4 * ny * nx) + (ii + y_n*nx)]; // south
  tmp_cells[(5 * ny * nx) + (ii + jj* nx)] = cells[(5 * ny * nx) + (x_w + y_s*nx)]; // north-east
  tmp_cells[(6 * ny * nx) + (ii + jj* nx)] = cells[(6 * ny * nx) + (x_e + y_s*nx)]; // north-west
  tmp_cells[(7 * ny * nx) + (ii + jj* nx)] = cells[(7 * ny * nx) + (x_e + y_n*nx)]; // south-west
  tmp_cells[(8 * ny * nx) + (ii + jj* nx)] = cells[(8 * ny * nx) + (x_w + y_n*nx)]; // south-east
}

kernel void rebound(global float* cells,
                    global float* tmp_cells,
                    global int* obstacles,
                    int nx, int ny) {
  // get column and row indices
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  // if the cell contains an obstacle
  if (obstacles[jj*nx + ii]) {
    // called after propagate, so taking values from scratch space
    // mirroring, and writing into main grid
    cells[(1 * ny * nx) + (ii + jj*nx)] = tmp_cells[(3 * ny * nx) + (ii + jj*nx)];
    cells[(2 * ny * nx) + (ii + jj*nx)] = tmp_cells[(4 * ny * nx) + (ii + jj*nx)];
    cells[(3 * ny * nx) + (ii + jj*nx)] = tmp_cells[(1 * ny * nx) + (ii + jj*nx)];
    cells[(4 * ny * nx) + (ii + jj*nx)] = tmp_cells[(2 * ny * nx) + (ii + jj*nx)];
    cells[(5 * ny * nx) + (ii + jj*nx)] = tmp_cells[(7 * ny * nx) + (ii + jj*nx)];
    cells[(6 * ny * nx) + (ii + jj*nx)] = tmp_cells[(8 * ny * nx) + (ii + jj*nx)];
    cells[(7 * ny * nx) + (ii + jj*nx)] = tmp_cells[(5 * ny * nx) + (ii + jj*nx)];
    cells[(8 * ny * nx) + (ii + jj*nx)] = tmp_cells[(6 * ny * nx) + (ii + jj*nx)];
  }
}

kernel void collision(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      float omega,
                      int nx, int ny) {
  const float c_sq = 1.f / 3.f; // square of speed of sound
  const float w0 = 4.f / 9.f;   // weighting factor
  const float w1 = 1.f / 9.f;   // weighting factor
  const float w2 = 1.f / 36.f;  // weighting factor

  // get column and row indices
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  // don't consider occupied cells
  if (!obstacles[ii + jj*nx]) {
    // compute local density total
    float local_density = 0.f;

    for (int kk = 0; kk < NSPEEDS; kk++) {
      local_density += tmp_cells[(kk * ny * nx) + (ii + jj*nx)];
    }

    // compute x velocity component
    float u_x = (tmp_cells[(1 * ny * nx) + (ii + jj*nx)]
                  + tmp_cells[(5 * ny * nx) + (ii + jj*nx)]
                  + tmp_cells[(8 * ny * nx) + (ii + jj*nx)]
                  - (tmp_cells[(3 * ny * nx) + (ii + jj*nx)]
                     + tmp_cells[(6 * ny * nx) + (ii + jj*nx)]
                     + tmp_cells[(7 * ny * nx) + (ii + jj*nx)]))
                 / local_density;
    // compute y velocity component
    float u_y = (tmp_cells[(2 * ny * nx) + (ii + jj*nx)]
                  + tmp_cells[(5 * ny * nx) + (ii + jj*nx)]
                  + tmp_cells[(6 * ny * nx) + (ii + jj*nx)]
                  - (tmp_cells[(4 * ny * nx) + (ii + jj*nx)]
                     + tmp_cells[(7 * ny * nx) + (ii + jj*nx)]
                     + tmp_cells[(8 * ny * nx) + (ii + jj*nx)]))
                 / local_density;

    // velocity squared
    float u_sq = u_x * u_x + u_y * u_y;

    // directional velocity components
    float u[NSPEEDS];
    u[1] =   u_x;        // east
    u[2] =         u_y;  // north
    u[3] = - u_x;        // west
    u[4] =       - u_y;  // south
    u[5] =   u_x + u_y;  // north-east
    u[6] = - u_x + u_y;  // north-west
    u[7] = - u_x - u_y;  // south-west
    u[8] =   u_x - u_y;  // south-east

    // equilibrium densities
    float d_equ[NSPEEDS];
    // zero velocity density: weight w0
    d_equ[0] = w0 * local_density
               * (1.f - u_sq / (2.f * c_sq));
    // axis speeds: weight w1
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                     + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                     + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                     + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                     + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    // diagonal speeds: weight w2
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                     + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                     + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                     + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                     + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                     - u_sq / (2.f * c_sq));

    // relaxation step
    for (int kk = 0; kk < NSPEEDS; kk++) {
      cells[(kk * ny * nx) + (ii + jj*nx)] = tmp_cells[(kk * ny * nx) + (ii + jj*nx)]
                                              + omega
                                              * (d_equ[kk] - tmp_cells[(kk * ny * nx) + (ii + jj*nx)]);
    }
  }
}

kernel void av_velocity(global float* cells,
                        global int* obstacles,
                        int nx, int ny,
                        local  float* local_velocities,
                        local  int* local_tot_cells,
                        global float* global_velocities,
                        global int* global_tot_cells) {
  int tot_cells = 0; // no. of cells used in calculation
  float tot_u = 0.f; // accumulated magnitudes of velocity for each cell

  int local_id_x = get_local_id(0);
  int local_id_y = get_local_id(1);

  int local_size_x = get_local_size(0);
  int local_size_y = get_local_size(1);

  // get column and row indices
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  // ignore occupied cells
  if (!obstacles[ii + jj*nx]) {
    // local density total
    float local_density = 0.f;

    for (int kk = 0; kk < NSPEEDS; kk++) {
      local_density += cells[(kk * ny * nx) + (ii + jj*nx)];
    }

    // x-component of velocity
    float u_x = (cells[(1 * ny * nx) + (ii + jj*nx)]
                  + cells[(5 * ny * nx) + (ii + jj*nx)]
                  + cells[(8 * ny * nx) + (ii + jj*nx)]
                  - (cells[(3 * ny * nx) + (ii + jj*nx)]
                     + cells[(6 * ny * nx) + (ii + jj*nx)]
                     + cells[(7 * ny * nx) + (ii + jj*nx)]))
                 / local_density;
    // compute y velocity component
    float u_y = (cells[(2 * ny * nx) + (ii + jj*nx)]
                  + cells[(5 * ny * nx) + (ii + jj*nx)]
                  + cells[(6 * ny * nx) + (ii + jj*nx)]
                  - (cells[(4 * ny * nx) + (ii + jj*nx)]
                     + cells[(7 * ny * nx) + (ii + jj*nx)]
                     + cells[(8 * ny * nx) + (ii + jj*nx)]))
                 / local_density;
    // accumulate the norm of x- and y- velocity components
    tot_u = sqrt((u_x * u_x) + (u_y * u_y));
    // increase counter of inspected cells
    tot_cells = 1;
  }

  local_velocities[local_id_x + local_id_y * local_size_x] = tot_u;
  local_tot_cells[local_id_x + local_id_y * local_size_x] = tot_cells;

  int group_id_x    = get_group_id(0);
  int group_id_y    = get_group_id(1);

  int work_groups_x = get_global_size(0) / local_size_x;

  uint group_size = local_size_x * local_size_y;

  for (uint stride = group_size / 2; stride > 0; stride /= 2) {

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((local_id_x + local_id_y * local_size_x) < stride) {
      local_velocities[local_id_x + local_id_y * local_size_x] +=
      local_velocities[(local_id_x + local_id_y * local_size_x) + stride];
      local_tot_cells[local_id_x + local_id_y * local_size_x] +=
      local_tot_cells[(local_id_x + local_id_y * local_size_x) + stride];
    }
  }

  if (local_id_x == 0 && local_id_y == 0) {
    global_velocities[group_id_x + group_id_y * work_groups_x] = local_velocities[0];
    global_tot_cells[group_id_x + group_id_y * work_groups_x] = local_tot_cells[0];
  }
}
