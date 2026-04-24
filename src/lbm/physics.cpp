#include <lbm/physics.hpp>

#include <cassert>
#include <cstdlib>

#include <omp.h>

#include <lbm/communications.hpp>
#include <lbm/config.hpp>
#include <lbm/structures.hpp>

#if DIRECTIONS == 9 && DIMENSIONS == 2
/// Definition of the 9 base vectors used to discretize the directions on each mesh.
const Vector direction_matrix[DIRECTIONS] = {
  // clang-format off
  {+0.0, +0.0},
  {+1.0, +0.0}, {+0.0, +1.0}, {-1.0, +0.0}, {+0.0, -1.0},
  {+1.0, +1.0}, {-1.0, +1.0}, {-1.0, -1.0}, {+1.0, -1.0},
  // clang-format on
};
#else
#error Need to define adapted direction matrix.
#endif

#if DIRECTIONS == 9
/// Weigths used to compensate the differences in lenght of the 9 directional vectors.
const double equil_weight[DIRECTIONS] = {
  // clang-format off
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
  // clang-format on
};

/// Opposite directions for bounce back implementation
const int opposite_of[DIRECTIONS] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
#else
#error Need to define adapted equilibrium distribution function
#endif

double get_vect_norm_2(Vector const a, Vector const b) {
  double res = 0.0;
  for (size_t k = 0; k < DIMENSIONS; k++) {
    res += a[k] * b[k];
  }
  return res;
}

double get_cell_density(const lbm_mesh_cell_t cell) {
  assert(cell.cells != NULL);
  double res = 0.0;
  for (size_t k = 0; k < DIRECTIONS; k++) {
    res += cell[k];
  }
  return res;
}

void get_cell_velocity(Vector v, const lbm_mesh_cell_t cell, double cell_density) {
  assert(v != NULL);
  assert(cell.cells != NULL);

  const double inv_cell_density = 1.0 / cell_density;

  // Loop on all dimensions
  for (size_t d = 0; d < DIMENSIONS; d++) {
    v[d] = 0.0;

    // Sum all directions
    for (size_t k = 0; k < DIRECTIONS; k++) {
      v[d] += cell[k] * direction_matrix[k][d];
    }

    // Normalize
    v[d] *= inv_cell_density;
  }
}

double compute_equilibrium_profile(Vector velocity, double density, int direction) {
  const double v2 = get_vect_norm_2(velocity, velocity);

  // Compute `e_i * v_i / c`
  const double p  = get_vect_norm_2(direction_matrix[direction], velocity);
  const double p2 = p * p;

  // Terms without density and direction weight
  double f_eq = 1.0 + (3.0 * p) + (4.5 * p2) - (1.5 * v2);

  // Multiply everything by the density and direction weight
  f_eq *= equil_weight[direction] * density;

  return f_eq;
}

void compute_cell_collision(lbm_mesh_cell_t cell_out, const lbm_mesh_cell_t cell_in) {
  // Compute macroscopic values
  const double density = get_cell_density(cell_in);
  Vector v;
  get_cell_velocity(v, cell_in, density);

  // Loop on microscopic directions
  for (size_t k = 0; k < DIRECTIONS; k++) {
    // Compute f at equilibrium
    double f_eq = compute_equilibrium_profile(v, density, k);
    // Compute f_out
    cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - f_eq);
  }
}

void compute_bounce_back(lbm_mesh_cell_t cell) {
  double tmp[DIRECTIONS];
  for (size_t k = 0; k < DIRECTIONS; k++) {
    tmp[k] = cell[opposite_of[k]];
  }
  for (size_t k = 0; k < DIRECTIONS; k++) {
    cell[k] = tmp[k];
  }
}

double helper_compute_poiseuille(const size_t i, const size_t size) {
  const double y = (double)(i - 1);
  const double L = (double)(size - 1);
  return 4.0 * INFLOW_MAX_VELOCITY / (L * L) * (L * y - y * y);
}

void compute_inflow_zou_he_poiseuille_distr(const Mesh* mesh, lbm_mesh_cell_t cell, size_t id_y) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  // Set macroscopic fluid info
  // Poiseuille distribution on X and null on Y
  // We just want the norm, so `v = v_x`
  const double v = helper_compute_poiseuille(id_y, mesh->height);

  // Compute rho from U and inner flow on surface
  const double rho = (cell[0] + cell[2] + cell[4] + 2 * (cell[3] + cell[6] + cell[7])) / (1.0 - v);

  // Now compute unknown microscopic values
  cell[1] = cell[3]; // + (2.0/3.0) * density * v_y <--- no velocity on Y so v_y = 0
  cell[5] = cell[7] - (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); // + (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0
  cell[8] = cell[6] + (1.0 / 2.0) * (cell[2] - cell[4])
            + (1.0 / 6.0) * (rho * v); //- (1.0/2.0) * rho * v_y    <--- no velocity on Y so v_y = 0

  // No need to copy already known one as the value will be "loss" in the wall at propagatation time
}

void compute_outflow_zou_he_const_density(lbm_mesh_cell_t cell) {
#if DIRECTIONS != 9
#error Implemented only for 9 directions
#endif

  double const rho = 1.0;
  // Compute macroscopic velocity depending on inner flow going onto the wall
  const double v = -1.0 + (1.0 / rho) * (cell[0] + cell[2] + cell[4] + 2 * (cell[1] + cell[5] + cell[8]));

  // Now can compute unknown microscopic values
  cell[3] = cell[1] - (2.0 / 3.0) * rho * v;
  cell[7] = cell[5]
            + (1.0 / 2.0) * (cell[2] - cell[4])
            // - (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
  cell[6] = cell[8]
            + (1.0 / 2.0) * (cell[4] - cell[2])
            // + (1.0/2.0) * (rho * v_y)    <--- no velocity on Y so v_y = 0
            - (1.0 / 6.0) * (rho * v);
}

void special_cells(Mesh* mesh, lbm_mesh_type_t* mesh_type, const lbm_comm_t* mesh_comm) {
  // Loop on all inner cells
  for (size_t i = 1; i < mesh->width - 1; i++) {
    for (size_t j = 1; j < mesh->height - 1; j++) {
      switch (*(lbm_cell_type_t_get_cell(mesh_type, i, j))) {
      case CELL_FUILD:
        break;
      case CELL_BOUNCE_BACK:
        compute_bounce_back(Mesh_get_cell(mesh, i, j));
        break;
      case CELL_LEFT_IN:
        compute_inflow_zou_he_poiseuille_distr(mesh, Mesh_get_cell(mesh, i, j), j + mesh_comm->y);
        break;
      case CELL_RIGHT_OUT:
        compute_outflow_zou_he_const_density(Mesh_get_cell(mesh, i, j));
        break;
      }
    }
  }
}

void collision(Mesh* mesh_out, const Mesh* mesh_in) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);
  assert(mesh_in->cell_count == mesh_out->cell_count);

  const double relax = RELAX_PARAMETER;
  constexpr double one_ninth        = 1.0 / 9.0;
  constexpr double four_ninths      = 4.0 * one_ninth;
  constexpr double one_thirty_sixth = 1.0 / 36.0;

  const double* in0 = Mesh_get_direction(mesh_in, 0);
  const double* in1 = Mesh_get_direction(mesh_in, 1);
  const double* in2 = Mesh_get_direction(mesh_in, 2);
  const double* in3 = Mesh_get_direction(mesh_in, 3);
  const double* in4 = Mesh_get_direction(mesh_in, 4);
  const double* in5 = Mesh_get_direction(mesh_in, 5);
  const double* in6 = Mesh_get_direction(mesh_in, 6);
  const double* in7 = Mesh_get_direction(mesh_in, 7);
  const double* in8 = Mesh_get_direction(mesh_in, 8);

  double* out0 = Mesh_get_direction(mesh_out, 0);
  double* out1 = Mesh_get_direction(mesh_out, 1);
  double* out2 = Mesh_get_direction(mesh_out, 2);
  double* out3 = Mesh_get_direction(mesh_out, 3);
  double* out4 = Mesh_get_direction(mesh_out, 4);
  double* out5 = Mesh_get_direction(mesh_out, 5);
  double* out6 = Mesh_get_direction(mesh_out, 6);
  double* out7 = Mesh_get_direction(mesh_out, 7);
  double* out8 = Mesh_get_direction(mesh_out, 8);

  const size_t height = mesh_in->height;

  // Loop on all inner cells
  for (size_t i = 1; i < mesh_in->width - 1; i++) {
    const size_t begin = i * height + 1;
    const size_t end   = begin + height - 2;

#pragma omp simd
    for (size_t id = begin; id < end; id++) {
      const double f0 = in0[id];
      const double f1 = in1[id];
      const double f2 = in2[id];
      const double f3 = in3[id];
      const double f4 = in4[id];
      const double f5 = in5[id];
      const double f6 = in6[id];
      const double f7 = in7[id];
      const double f8 = in8[id];

      // Same macroscopic values as compute_cell_collision(), inlined for D2Q9.
      const double density = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
      const double inv_density = 1.0 / density;
      const double vx          = (f1 - f3 + f5 - f6 - f7 + f8) * inv_density;
      const double vy          = (f2 - f4 + f5 + f6 - f7 - f8) * inv_density;
      const double v2          = vx * vx + vy * vy;
      const double eq_base     = 1.0 - 1.5 * v2;
      const double rho_w0      = density * four_ninths;
      const double rho_w1      = density * one_ninth;
      const double rho_w2      = density * one_thirty_sixth;

      double p    = 0.0;
      double p2   = 0.0;
      double f_eq = rho_w0 * eq_base;
      out0[id]    = f0 - relax * (f0 - f_eq);

      p           = vx;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      out1[id]    = f1 - relax * (f1 - f_eq);

      p           = vy;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      out2[id]    = f2 - relax * (f2 - f_eq);

      p           = -vx;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      out3[id]    = f3 - relax * (f3 - f_eq);

      p           = -vy;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      out4[id]    = f4 - relax * (f4 - f_eq);

      p           = vx + vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      out5[id]    = f5 - relax * (f5 - f_eq);

      p           = -vx + vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      out6[id]    = f6 - relax * (f6 - f_eq);

      p           = -vx - vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      out7[id]    = f7 - relax * (f7 - f_eq);

      p           = vx - vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      out8[id]    = f8 - relax * (f8 - f_eq);
    }
  }
}

void propagation(Mesh* mesh_out, const Mesh* mesh_in) {
  assert(mesh_in->width == mesh_out->width);
  assert(mesh_in->height == mesh_out->height);
  assert(mesh_in->cell_count == mesh_out->cell_count);

  const size_t width  = mesh_out->width;
  const size_t height = mesh_out->height;

  // Loop on all directions first: each direction block is contiguous in the SoA layout.
  for (size_t k = 0; k < DIRECTIONS; k++) {
    const double* in = Mesh_get_direction(mesh_in, k);
    double* out      = Mesh_get_direction(mesh_out, k);
    const int dx     = static_cast<int>(direction_matrix[k][0]);
    const int dy     = static_cast<int>(direction_matrix[k][1]);

    const size_t y_begin = dy < 0 ? 1 : 0;
    const size_t y_end   = dy > 0 ? height - 1 : height;
    const size_t out_y_begin = static_cast<size_t>(static_cast<int>(y_begin) + dy);

    for (size_t i = 0; i < width; i++) {
      const int ii = static_cast<int>(i) + dx;
      if (ii < 0 || ii >= static_cast<int>(width)) {
        continue;
      }

      const size_t in_begin  = i * height + y_begin;
      const size_t out_begin = static_cast<size_t>(ii) * height + out_y_begin;
      const size_t count     = y_end - y_begin;

#pragma omp simd
      for (size_t offset = 0; offset < count; offset++) {
        out[out_begin + offset] = in[in_begin + offset];
      }
    }
  }
}
