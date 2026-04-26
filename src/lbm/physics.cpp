#include <lbm/physics.hpp>
#include <immintrin.h>

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

alignas(32) const double dir0[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
alignas(32) const double dir1[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};


#else
#error Need to define adapted direction matrix.
#endif

#if DIRECTIONS == 9
/// Weigths used to compensate the differences in lenght of the 9 directional vectors.
alignas(32) const double equil_weight[DIRECTIONS] = {
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

  // Loop on all dimensions
  for (size_t d = 0; d < DIMENSIONS; d++) {
    v[d] = 0.0;

    // Sum all directions
    for (size_t k = 0; k < DIRECTIONS; k++) {
      v[d] += cell[k] * direction_matrix[k][d];
    }

    // Normalize
    v[d] /= cell_density;
  }
}

double compute_equilibrium_profile(Vector velocity, double density, int direction) {
  const double v2 = get_vect_norm_2(velocity, velocity);

  // Compute `e_i * v_i / c`
  const double p  = get_vect_norm_2(direction_matrix[direction], velocity);
  const double p2 = p * p;

  // Terms without density and direction weight
  double f_eq = 1.0 + (3.0 * p) + ((9.0 / 2.0) * p2) - ((3.0 / 2.0) * v2);

  // Multiply everything by the density and direction weight
  f_eq *= equil_weight[direction] * density;

  return f_eq;
}


alignas(32) static const double C_ONE[4]   = {1.0, 1.0, 1.0, 1.0};
alignas(32) static const double C_THREE[4] = {3.0, 3.0, 3.0, 3.0};

inline double reduce_avx2(__m256d v) {
  __m128d low = _mm256_extractf128_pd(v, 0);   // [a0, a1]
  __m128d high = _mm256_extractf128_pd(v, 1);  // [a2, a3]
  
  // Additionner les 2 lanes
  __m128d sum128 = _mm_add_pd(low, high);       // [a0+a2, a1+a3]
  
  // HADD SSE (inverse et additionne)
  __m128d hadd = _mm_hadd_pd(sum128, sum128);   // [a0+a2+a1+a3, a0+a2+a1+a3]
  
  return _mm_cvtsd_f64(hadd);  // Extraire le premier double
}

inline void compute_cell_collision(lbm_mesh_cell_t cell_out, const lbm_mesh_cell_t cell_in) {
  // Compute macroscop  ic values
  Vector v;
  double density,f_eq,p,p2,c1,c2;


  // Load cell_in[k]
  __m256d c_4a = _mm256_loadu_pd(&cell_in[0]); 
  __m256d c_4b = _mm256_loadu_pd(&cell_in[4]); 

  // Load directions[k][0/1]
  __m256d d0_4a = _mm256_load_pd(&dir0[0]);
  __m256d d0_4b = _mm256_load_pd(&dir0[4]);

  __m256d d1_4a = _mm256_load_pd(&dir1[0]);
  __m256d d1_4b = _mm256_load_pd(&dir1[4]);

  // Cell_in[k]*directions[k][0/1]
  __m256d mult0_a = _mm256_mul_pd(c_4a, d0_4a);
  __m256d mult0_b = _mm256_mul_pd(c_4b, d0_4b);
  __m256d mult1_a = _mm256_mul_pd(c_4a, d1_4a);
  __m256d mult1_b = _mm256_mul_pd(c_4b, d1_4b);

  // v[0]/v[1] += Cell_in[k]*directions[k][0/1]
  __m256d add_mult0 = _mm256_add_pd(mult0_a, mult0_b);
  __m256d add_mult1 = _mm256_add_pd(mult1_a, mult1_b);
  __m256d add_c = _mm256_add_pd(c_4a, c_4b);

  double sum_v0 = reduce_avx2(add_mult0);
  double sum_v1 = reduce_avx2(add_mult1);
  double sum_c = reduce_avx2(add_c);

  // add of the last element
  double _v0 = sum_v0 +  cell_in[8] * dir0[8];
  double _v1 = sum_v1 + cell_in[8] * dir1[8];
  density = sum_c + cell_in[8];

  // we use 1/density to avoid 1 divide
  double inv_density = 1.0 / density;

  // Normalize
  v[0] = _v0*inv_density;
  v[1] = _v1*inv_density;


  // we put outside constant calcul
  c1 = 9.0 / 2.0;
  c2 = 3.0 / 2.0;
  
  // Compute v2 norm
  const double v2 = v[0]*v[0] + v[1]*v[1];
  const double c2_v2 = c2*v2;

  double f_eq_t1,f_eq_t2;

  // direction[0]*v[0/1] + direction[1]*v[0/1]
  __m256d v0_4 = _mm256_set1_pd(v[0]);
  __m256d v1_4 = _mm256_set1_pd(v[1]);

  mult0_a = _mm256_mul_pd(v0_4,d0_4a);
  mult0_b = _mm256_mul_pd(v0_4,d0_4b);

  mult1_a = _mm256_mul_pd(v1_4,d1_4a);
  mult1_b = _mm256_mul_pd(v1_4,d1_4b);


  __m256d p_4a = _mm256_add_pd(mult0_a,mult1_a);
  __m256d p_4b = _mm256_add_pd(mult0_b,mult1_b);

  // calcul de p2 = p*p
  __m256d p2_4a = _mm256_mul_pd(p_4a,p_4a);
  __m256d p2_4b = _mm256_mul_pd(p_4b,p_4b);
  

  __m256d _1 = _mm256_load_pd(C_ONE);;
  __m256d _3 = _mm256_load_pd(C_THREE);

  // 1.0 + (3.0 * p)
  __m256d f_eq_t1_a = _mm256_fmadd_pd(_3,p_4a,_1);
  __m256d f_eq_t1_b = _mm256_fmadd_pd(_3,p_4b,_1);

  __m256d _c1 = _mm256_set1_pd(c1);
  __m256d _c2v2 = _mm256_set1_pd(c2_v2);

  // (c1* p2) - (c2 * v2);
  __m256d f_eq_t2_a = _mm256_fmsub_pd(_c1,p2_4a,_c2v2);
  __m256d f_eq_t2_b = _mm256_fmsub_pd(_c1,p2_4b,_c2v2);

  __m256d f_eq_a = _mm256_add_pd(f_eq_t1_a,f_eq_t2_a);
  __m256d f_eq_b = _mm256_add_pd(f_eq_t1_b,f_eq_t2_b);

  // equil_weight[direction]
  __m256d equil_4a = _mm256_load_pd(&equil_weight[0]); 
  __m256d equil_4b = _mm256_load_pd(&equil_weight[4]);

  // equil_weight[direction] * density;
  __m256d _density = _mm256_set1_pd(density);
  __m256d equil_mult_c_a = _mm256_mul_pd(equil_4a,_density);
  __m256d equil_mult_c_b = _mm256_mul_pd(equil_4b,_density);

  __m256d f_eq_final_a = _mm256_mul_pd(f_eq_a,equil_mult_c_a);
  __m256d f_eq_final_b = _mm256_mul_pd(f_eq_b,equil_mult_c_b);

  __m256d relax = _mm256_set1_pd(RELAX_PARAMETER);

  // (cell_in[k] - f_eq)
  __m256d diff_a = _mm256_sub_pd(c_4a, f_eq_final_a);
  __m256d diff_b = _mm256_sub_pd(c_4b, f_eq_final_b);


  __m256d out_a = _mm256_fnmadd_pd(relax, diff_a, c_4a);
  __m256d out_b = _mm256_fnmadd_pd(relax, diff_b, c_4b);


  // cell_out[k] = cell_in[k] - RELAX_PARAMETER * (cell_in[k] - f_eq)
  _mm256_storeu_pd(&cell_out[0], out_a);
  _mm256_storeu_pd(&cell_out[4], out_b);


  double p9 = dir0[8] * v[0] + dir1[8] * v[1];
  double p2_9 = p9 * p9;
  double f_eq_t1_9 = 1.0 + 3.0 * p9;
  double f_eq_t2_9 = c1 * p2_9 - c2_v2;
  double f_eq_9 = (f_eq_t1_9 + f_eq_t2_9) * equil_weight[8] * density;
  cell_out[8] = cell_in[8] - RELAX_PARAMETER * (cell_in[8] - f_eq_9);

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

  // Loop on all inner cells
  for (size_t i = 1; i < mesh_in->width - 1; i++) {
     for (size_t j = 1; j < mesh_in->height - 1; j++) {
      compute_cell_collision(Mesh_get_cell(mesh_out, i, j), Mesh_get_cell(mesh_in, i, j));
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
