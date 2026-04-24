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
  assert(cell != NULL);
  double res = 0.0;
  for (size_t k = 0; k < DIRECTIONS; k++) {
    res += cell[k];
  }
  return res;
}

void get_cell_velocity(Vector v, const lbm_mesh_cell_t cell, double cell_density) {
  assert(v != NULL);
  assert(cell != NULL);

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

  const double relax = RELAX_PARAMETER;
  constexpr double one_ninth        = 1.0 / 9.0;
  constexpr double four_ninths      = 4.0 * one_ninth;
  constexpr double one_thirty_sixth = 1.0 / 36.0;

  // Loop on all inner cells
  for (size_t i = 1; i < mesh_in->width - 1; i++) {
    for (size_t j = 1; j < mesh_in->height - 1; j++) {
      lbm_mesh_cell_t cell_out       = Mesh_get_cell(mesh_out, i, j);
      const lbm_mesh_cell_t cell_in  = Mesh_get_cell(mesh_in, i, j);
      const double f0                = cell_in[0];
      const double f1                = cell_in[1];
      const double f2                = cell_in[2];
      const double f3                = cell_in[3];
      const double f4                = cell_in[4];
      const double f5                = cell_in[5];
      const double f6                = cell_in[6];
      const double f7                = cell_in[7];
      const double f8                = cell_in[8];

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
      cell_out[0] = f0 - relax * (f0 - f_eq);

      p           = vx;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[1] = f1 - relax * (f1 - f_eq);

      p           = vy;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[2] = f2 - relax * (f2 - f_eq);

      p           = -vx;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[3] = f3 - relax * (f3 - f_eq);

      p           = -vy;
      p2          = p * p;
      f_eq        = rho_w1 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[4] = f4 - relax * (f4 - f_eq);

      p           = vx + vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[5] = f5 - relax * (f5 - f_eq);

      p           = -vx + vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[6] = f6 - relax * (f6 - f_eq);

      p           = -vx - vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[7] = f7 - relax * (f7 - f_eq);

      p           = vx - vy;
      p2          = p * p;
      f_eq        = rho_w2 * (eq_base + 3.0 * p + 4.5 * p2);
      cell_out[8] = f8 - relax * (f8 - f_eq);
    }
  }
}

void propagation(Mesh* mesh_out, const Mesh* mesh_in) {
  // Loop on all cells
  for (size_t j = 0; j < mesh_out->height; j++) {
    for (size_t i = 0; i < mesh_out->width; i++) {
      // For all direction
      for (size_t k = 0; k < DIRECTIONS; k++) {
        // Compute destination point
        ssize_t ii = (i + direction_matrix[k][0]);
        ssize_t jj = (j + direction_matrix[k][1]);
        // Propagate to neighboor nodes
        if ((ii >= 0 && ii < mesh_out->width) && (jj >= 0 && jj < mesh_out->height)) {
          Mesh_get_cell(mesh_out, ii, jj)[k] = Mesh_get_cell(mesh_in, i, j)[k];
        }
      }
    }
  }
}
