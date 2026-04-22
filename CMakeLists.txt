cmake_minimum_required(VERSION 3.25...4.2)

project(
  top.lbm
  LANGUAGES CXX
  VERSION 0.1.0
  DESCRIPTION "Hybrid MPI+OpenMP parallel D2Q9 lattice Boltzmann solver for Kármán vortex street flow"
  HOMEPAGE_URL "https://dssgabriel.github.io/CHP203-TOP/project"
)

find_package(MPI 3.0 REQUIRED)
find_package(OpenMP 4.0 REQUIRED)
include(cmake/third-party.cmake)

add_library(top.lbm-lib STATIC)
add_executable(top.lbm-exe)
add_executable(top.display)

add_subdirectory(src/lbm)
add_subdirectory(src/bin)

# Profil de compilation par type de build
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )

  # Debug: confortable pour gdb / assertions / stacks
  string(APPEND CMAKE_CXX_FLAGS_DEBUG
    " -Og -g3 -fno-omit-frame-pointer"
  )

  # RelWithDebInfo: meilleur choix pour MAQAO / perf / hotspot
  string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO
    " -O3 -g -fno-omit-frame-pointer -march=native"
  )

  # Release: bench final
  string(APPEND CMAKE_CXX_FLAGS_RELEASE
    " -O3 -DNDEBUG -march=native"
  )
endif()