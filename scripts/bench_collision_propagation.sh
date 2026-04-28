#!/usr/bin/env bash
set -euo pipefail
# scripts/bench_collision_propagation.sh --threads "1 2 4 8" --repeats 200 --warmup 20
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BENCH_DIR="${BUILD_DIR}/bench"
BENCH_SRC="${BENCH_DIR}/bench_collision_propagation.cpp"
BENCH_EXE="${BENCH_DIR}/bench_collision_propagation"
CONFIG_FILE="${ROOT_DIR}/config.txt"
REPEATS=200
WARMUP=20
THREADS=""
OUTPUT_CSV="${BENCH_DIR}/collision_propagation.csv"
CXX="${CXX:-g++}"
MPICXX="${MPICXX:-mpicxx}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --config FILE       Configuration LBM a utiliser (defaut: config.txt)
  --repeats N         Nombre d'appels mesures par kernel (defaut: ${REPEATS})
  --warmup N          Nombre d'appels de chauffe par kernel (defaut: ${WARMUP})
  --threads LIST      Liste OpenMP, ex: "1 2 4 8" (defaut: puissances de 2 + max)
  --output FILE       CSV de sortie (defaut: build/bench/collision_propagation.csv)
  --help              Affiche cette aide

Exemples:
  $0
  $0 --threads "1 2 4" --repeats 500
  OMP_PROC_BIND=close OMP_PLACES=cores $0 --threads "1 2 4 8"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --repeats|--iterations)
      REPEATS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Option inconnue: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

default_threads() {
  local max_threads="$1"
  local t=1
  local list=""

  while (( t < max_threads )); do
    list+="${t} "
    t=$(( t * 2 ))
  done

  list+="${max_threads}"
  echo "${list}"
}

if [[ -z "${THREADS}" ]]; then
  MAX_THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  THREADS="$(default_threads "${MAX_THREADS}")"
fi

mkdir -p "${BENCH_DIR}"

cat > "${BENCH_SRC}" <<'CPP'
#include <lbm/config.hpp>
#include <lbm/physics.hpp>
#include <lbm/structures.hpp>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct BenchOptions {
  const char* config_file = "config.txt";
  size_t repeats          = 200;
  size_t warmup           = 20;
};

struct BenchResult {
  double seconds;
  double checksum;
};

static void usage(const char* prog) {
  std::fprintf(
    stderr,
    "Usage: %s --config FILE --repeats N --warmup N\n",
    prog
  );
}

static size_t parse_size(const char* value, const char* option_name) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "Valeur invalide pour %s: %s\n", option_name, value);
    std::exit(2);
  }
  return static_cast<size_t>(parsed);
}

static BenchOptions parse_args(int argc, char** argv) {
  BenchOptions options;

  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      options.config_file = argv[++i];
    } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      options.repeats = parse_size(argv[++i], "--repeats");
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      options.warmup = parse_size(argv[++i], "--warmup");
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Option inconnue ou incomplete: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }

  return options;
}

static void fill_mesh(Mesh* mesh) {
  for (size_t id = 0; id < mesh->cell_count; id++) {
    const size_t x = id / mesh->height;
    const size_t y = id % mesh->height;

    for (size_t k = 0; k < DIRECTIONS; k++) {
      const double perturbation =
        1.0 + 0.001 * static_cast<double>((17 * x + 13 * y + 7 * k) % 23);
      Mesh_get_direction(mesh, k)[id] = equil_weight[k] * perturbation;
    }
  }
}

static double checksum_mesh(const Mesh* mesh) {
  double sum = 0.0;
  const size_t stride = std::max<size_t>(1, mesh->cell_count / 4096);

  for (size_t k = 0; k < DIRECTIONS; k++) {
    const double* values = Mesh_get_direction(mesh, k);
    for (size_t id = 0; id < mesh->cell_count; id += stride) {
      sum += values[id] * static_cast<double>(k + 1);
    }
  }

  return sum;
}

static BenchResult run_kernel(
  void (*kernel)(Mesh*, const Mesh*),
  size_t width,
  size_t height,
  size_t repeats,
  size_t warmup
) {
  Mesh in;
  Mesh out;
  Mesh_init(&in, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  Mesh_init(&out, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  fill_mesh(&in);
  fill_mesh(&out);

  for (size_t i = 0; i < warmup; i++) {
    kernel(&out, &in);
    std::swap(in.cells, out.cells);
  }

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < repeats; i++) {
    kernel(&out, &in);
    std::swap(in.cells, out.cells);
  }
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  const double checksum = checksum_mesh(&in);

  Mesh_release(&in);
  Mesh_release(&out);

  return {elapsed.count(), checksum};
}

static void print_result(
  const char* kernel_name,
  const BenchResult result,
  size_t width,
  size_t height,
  size_t repeats,
  size_t warmup
) {
  const double active_cells = static_cast<double>((width - 2) * (height - 2));
  const double lattice_updates = active_cells * static_cast<double>(repeats);
  const double mlups = lattice_updates / (result.seconds * 1.0e6);
  const double ns_per_lup = result.seconds * 1.0e9 / lattice_updates;

  std::printf(
    "%s,%d,%zu,%zu,%zu,%zu,%.9f,%.6f,%.3f,%.17g\n",
    kernel_name,
    omp_get_max_threads(),
    width - 2,
    height - 2,
    repeats,
    warmup,
    result.seconds,
    ns_per_lup,
    mlups,
    result.checksum
  );
}

int main(int argc, char** argv) {
  const BenchOptions options = parse_args(argc, argv);
  load_config(options.config_file);

  const size_t width = static_cast<size_t>(MESH_WIDTH) + 2;
  const size_t height = static_cast<size_t>(MESH_HEIGHT) + 2;

  const BenchResult collision_result =
    run_kernel(collision, width, height, options.repeats, options.warmup);
  print_result("collision", collision_result, width, height, options.repeats, options.warmup);

  const BenchResult propagation_result =
    run_kernel(propagation, width, height, options.repeats, options.warmup);
  print_result("propagation", propagation_result, width, height, options.repeats, options.warmup);

  return 0;
}
CPP

MPI_COMPILE_FLAGS=()
if command -v "${MPICXX}" >/dev/null 2>&1; then
  MPI_SHOWME="$("${MPICXX}" --showme:compile 2>/dev/null || true)"
  if [[ -n "${MPI_SHOWME}" ]]; then
    read -r -a MPI_COMPILE_FLAGS <<< "${MPI_SHOWME}"
  fi
fi

COMPILE_CMD=(
  "${CXX}"
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -DOMPI_SKIP_MPICXX=1 \
  -DMPICH_SKIP_MPICXX=1 \
  -march=native \
  -fopenmp \
  -I"${ROOT_DIR}/include" \
  -I"${BUILD_DIR}/include" \
  "${MPI_COMPILE_FLAGS[@]}" \
  "${BENCH_SRC}" \
  "${ROOT_DIR}/src/lbm/config.cpp" \
  "${ROOT_DIR}/src/lbm/structures.cpp" \
  "${ROOT_DIR}/src/lbm/physics.cpp" \
  -o "${BENCH_EXE}"
)

if ! "${COMPILE_CMD[@]}"; then
  echo "Compilation avec ${CXX} echouee, tentative avec ${MPICXX}." >&2
  "${MPICXX}" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -DOMPI_SKIP_MPICXX=1 \
    -DMPICH_SKIP_MPICXX=1 \
    -march=native \
    -fopenmp \
    -I"${ROOT_DIR}/include" \
    -I"${BUILD_DIR}/include" \
    "${BENCH_SRC}" \
    "${ROOT_DIR}/src/lbm/config.cpp" \
    "${ROOT_DIR}/src/lbm/structures.cpp" \
    "${ROOT_DIR}/src/lbm/physics.cpp" \
    -o "${BENCH_EXE}"
fi

{
  echo "kernel,threads,width,height,repeats,warmup,seconds,ns_per_lup,mlups,checksum"
  for thread_count in ${THREADS}; do
    OMP_NUM_THREADS="${thread_count}" "${BENCH_EXE}" \
      --config "${CONFIG_FILE}" \
      --repeats "${REPEATS}" \
      --warmup "${WARMUP}"
  done
} > "${OUTPUT_CSV}"

echo "Benchmark ecrit dans: ${OUTPUT_CSV}"
echo
if command -v column >/dev/null 2>&1; then
  column -s, -t "${OUTPUT_CSV}"
else
  cat "${OUTPUT_CSV}"
fi
