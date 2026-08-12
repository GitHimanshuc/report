#include <arpack.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr a_int N = 500;
constexpr a_int NEV = 3;
constexpr a_int NCV = 20;

struct Result {
  std::vector<double> eigenvalues;
  a_int update_iterations;
  long operator_calls;
  double elapsed_ms;
  double max_residual;
};

double diagonal_entry(a_int i) {
  const double value = static_cast<double>(i + 1);
  return value * value;
}

std::vector<double> quadratic_diagonal() {
  std::vector<double> diagonal(N);
  for (a_int i = 0; i < N; ++i) {
    diagonal[i] = diagonal_entry(i);
  }
  return diagonal;
}

std::vector<double> largest_cluster_diagonal(double internal_spacing,
                                             double boundary_gap) {
  std::vector<double> diagonal = quadratic_diagonal();
  const double fourth_largest = diagonal[N - 4];
  diagonal[N - 3] = fourth_largest + boundary_gap;
  diagonal[N - 2] = diagonal[N - 3] + internal_spacing;
  diagonal[N - 1] = diagonal[N - 2] + internal_spacing;
  return diagonal;
}

Result solve(const std::vector<double>& diagonal, const std::string& which,
             bool shift_invert) {
  const a_int ldv = N;
  const a_int lworkl = NCV * (NCV + 8);

  // INFO=1 tells ARPACK to use this deterministic starting vector.
  std::vector<double> resid(N, 1.0);
  std::vector<double> v(ldv * NCV, 0.0);
  std::vector<double> workd(3 * N, 0.0);
  std::vector<double> workl(lworkl, 0.0);
  std::vector<a_int> iparam(11, 0);
  std::vector<a_int> ipntr(11, 0);

  iparam[0] = 1;             // Use ARPACK's exact shifts.
  iparam[2] = 10 * N;        // Maximum Arnoldi update iterations.
  iparam[3] = 1;             // Block size.
  iparam[6] = shift_invert ? 3 : 1;

  a_int ido = 0;
  a_int info = 1;
  long operator_calls = 0;
  const char* bmat = "I";
  const double tol = 0.0;    // ARPACK chooses machine precision.

  const auto start = std::chrono::steady_clock::now();
  while (true) {
    dsaupd_c(&ido, bmat, N, which.c_str(), NEV, tol, resid.data(), NCV,
             v.data(), ldv, iparam.data(), ipntr.data(), workd.data(),
             workl.data(), lworkl, &info);

    if (ido == -1 || ido == 1) {
      const double* x = workd.data() + ipntr[0] - 1;
      double* y = workd.data() + ipntr[1] - 1;
      for (a_int i = 0; i < N; ++i) {
        const double entry = diagonal[i];
        y[i] = shift_invert ? x[i] / entry : entry * x[i];
      }
      ++operator_calls;
    } else if (ido == 2) {
      // B is the identity, so y = B*x = x.
      const double* x = workd.data() + ipntr[0] - 1;
      double* y = workd.data() + ipntr[1] - 1;
      std::copy(x, x + N, y);
    } else if (ido == 99) {
      break;
    } else {
      throw std::runtime_error("Unexpected ARPACK reverse-communication code");
    }
  }
  const auto stop = std::chrono::steady_clock::now();

  if (info != 0) {
    throw std::runtime_error("dsaupd_c failed with info=" +
                             std::to_string(info));
  }

  std::vector<a_int> select(NCV, 0);
  std::vector<double> eigenvalues(NEV, 0.0);
  std::vector<double> eigenvectors(N * NEV, 0.0);
  const a_int rvec = 1;
  const char* howmny = "A";
  const double sigma = 0.0;

  dseupd_c(rvec, howmny, select.data(), eigenvalues.data(),
           eigenvectors.data(), N, sigma, bmat, N, which.c_str(), NEV, tol,
           resid.data(), NCV, v.data(), ldv, iparam.data(), ipntr.data(),
           workd.data(), workl.data(), lworkl, &info);
  if (info != 0) {
    throw std::runtime_error("dseupd_c failed with info=" +
                             std::to_string(info));
  }

  std::vector<std::pair<double, double>> values_and_residuals;
  for (a_int j = 0; j < NEV; ++j) {
    double residual_squared = 0.0;
    for (a_int i = 0; i < N; ++i) {
      const double z = eigenvectors[i + j * N];
      const double residual = diagonal[i] * z - eigenvalues[j] * z;
      residual_squared += residual * residual;
    }
    values_and_residuals.emplace_back(eigenvalues[j],
                                      std::sqrt(residual_squared));
  }
  std::sort(values_and_residuals.begin(), values_and_residuals.end());

  double max_residual = 0.0;
  for (a_int j = 0; j < NEV; ++j) {
    eigenvalues[j] = values_and_residuals[j].first;
    max_residual = std::max(max_residual, values_and_residuals[j].second);
  }

  return {eigenvalues,
          iparam[2],
          operator_calls,
          std::chrono::duration<double, std::milli>(stop - start).count(),
          max_residual};
}

void print_result(const std::string& name, const Result& result,
                  const std::string& operation) {
  std::cout << name << "\n  eigenvalues: [";
  for (a_int i = 0; i < NEV; ++i) {
    if (i != 0) std::cout << ' ';
    std::cout << result.eigenvalues[i];
  }
  std::cout << "]\n  ARPACK update iterations: " << result.update_iterations
            << "\n  " << operation << ": " << result.operator_calls
            << "\n  elapsed time: " << std::fixed << std::setprecision(3)
            << result.elapsed_ms << " ms\n  maximum residual: "
            << std::scientific << result.max_residual << "\n";
}

void print_gap_sweep(const std::string& heading,
                     const std::string& varied_quantity,
                     const std::array<double, 13>& gaps,
                     bool vary_internal_spacing) {
  constexpr double fixed_gap = 1000.0;
  std::cout << '\n'
            << heading << '\n'
            << "  "
            << (vary_internal_spacing ? "boundary gap" : "internal spacing")
            << " fixed at " << std::scientific << std::setprecision(3)
            << fixed_gap << '\n'
            << std::setw(20) << varied_quantity << std::setw(22)
            << "update iterations" << std::setw(26)
            << "matrix-vector products" << std::setw(22)
            << "maximum residual" << std::setw(24)
            << "max eigenvalue error" << '\n';

  for (const double gap : gaps) {
    const double internal_spacing = vary_internal_spacing ? gap : fixed_gap;
    const double boundary_gap = vary_internal_spacing ? fixed_gap : gap;
    const std::vector<double> diagonal =
        largest_cluster_diagonal(internal_spacing, boundary_gap);
    const Result result = solve(diagonal, "LM", false);
    double max_eigenvalue_error = 0.0;
    for (a_int j = 0; j < NEV; ++j) {
      max_eigenvalue_error =
          std::max(max_eigenvalue_error,
                   std::abs(result.eigenvalues[j] - diagonal[N - NEV + j]));
    }
    std::cout << std::scientific << std::setprecision(3) << std::setw(18) << gap
              << std::defaultfloat << std::setw(20)
              << result.update_iterations << std::setw(20)
              << result.operator_calls << std::scientific
              << std::setprecision(3) << std::setw(22) << result.max_residual
              << std::setw(24) << max_eigenvalue_error
              << '\n';
  }
}

}  // namespace

int main() {
  try {
    const std::vector<double> diagonal = quadratic_diagonal();
    print_result("Smallest magnitude (SM)", solve(diagonal, "SM", false),
                 "matrix-vector products");
    print_result("Largest magnitude (LM)", solve(diagonal, "LM", false),
                 "matrix-vector products");
    print_result("Shift-invert (sigma=0)", solve(diagonal, "LM", true),
                 "inverse applications");

    constexpr std::array<double, 13> gaps{
        1.0e3,  1.0e2,  1.0e1,  1.0,     1.0e-1, 1.0e-2, 1.0e-3,
        1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9};
    print_gap_sweep(
        "Largest-magnitude sweep: spacing within the requested top-three "
        "cluster",
        "internal spacing", gaps, true);
    print_gap_sweep(
        "Largest-magnitude sweep: separation of the requested cluster from "
        "the rest",
        "boundary gap", gaps, false);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
