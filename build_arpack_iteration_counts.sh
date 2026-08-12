#!/bin/sh
set -eu

conda_rpath=""
if [ -n "${CONDA_PREFIX:-}" ]; then
  conda_pkgconfig="${CONDA_PREFIX}/lib/pkgconfig"
  export PKG_CONFIG_PATH="${conda_pkgconfig}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
  conda_rpath="-Wl,-rpath,${CONDA_PREFIX}/lib"
fi

if ! pkg-config --exists arpack; then
  echo "ARPACK-NG development files were not found by pkg-config." >&2
  echo "For Conda, install them with: conda install -c conda-forge arpack" >&2
  exit 1
fi

${CXX:-c++} -O3 -std=c++17 -Wall -Wextra -pedantic \
  arpack_iteration_counts.cpp \
  $(pkg-config --cflags --libs arpack) \
  ${conda_rpath} \
  -o arpack_iteration_counts

echo "Built ./arpack_iteration_counts"
