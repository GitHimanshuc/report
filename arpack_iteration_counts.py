#!/usr/bin/env python3
"""Reproduce the diagonal-matrix ARPACK comparison in the paper."""

from time import perf_counter

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigs


N = 500
K = 3
NCV = 20
SEED = 0


def run_regular(matrix, diagonal, v0, which):
    """Run regular ARPACK and count matrix-vector products."""
    operator_calls = 0

    def matvec(vector):
        nonlocal operator_calls
        operator_calls += 1
        return matrix @ vector

    operator = LinearOperator(matrix.shape, matvec=matvec, dtype=matrix.dtype)
    start = perf_counter()
    eigenvalues, eigenvectors = eigs(
        operator, k=K, which=which, ncv=NCV, tol=0.0, v0=v0
    )
    elapsed = perf_counter() - start
    return summarize(eigenvalues, eigenvectors, diagonal, operator_calls, elapsed)


def run_shift_invert(matrix, diagonal, v0):
    """Run shift-invert ARPACK and count inverse-operator applications."""
    inverse_calls = 0
    start = perf_counter()
    lu, pivots = lu_factor(matrix)

    def inverse_matvec(vector):
        nonlocal inverse_calls
        inverse_calls += 1
        return lu_solve((lu, pivots), vector)

    operator = LinearOperator(
        matrix.shape, matvec=lambda vector: matrix @ vector, dtype=matrix.dtype
    )
    inverse_operator = LinearOperator(
        matrix.shape, matvec=inverse_matvec, dtype=matrix.dtype
    )
    eigenvalues, eigenvectors = eigs(
        operator,
        k=K,
        sigma=0.0,
        which="LM",
        ncv=NCV,
        tol=0.0,
        v0=v0,
        OPinv=inverse_operator,
    )
    elapsed = perf_counter() - start
    return summarize(eigenvalues, eigenvectors, diagonal, inverse_calls, elapsed)


def summarize(eigenvalues, eigenvectors, diagonal, calls, elapsed):
    order = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    residuals = [
        np.linalg.norm(diagonal * eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
        for i in range(K)
    ]
    return eigenvalues.real, calls, elapsed, max(residuals)


def print_result(name, result, operation):
    eigenvalues, calls, elapsed, max_residual = result
    print(name)
    print(f"  eigenvalues: {eigenvalues}")
    print(f"  {operation}: {calls}")
    print(f"  elapsed time: {1_000 * elapsed:.1f} ms")
    print(f"  maximum residual: {max_residual:.3e}")


def main():
    diagonal = np.arange(1, N + 1, dtype=np.float64) ** 2
    matrix = np.diag(diagonal)
    v0 = np.random.default_rng(SEED).standard_normal(N)

    smallest = run_regular(matrix, diagonal, v0, "SM")
    largest = run_regular(matrix, diagonal, v0, "LM")
    shifted = run_shift_invert(matrix, diagonal, v0)

    print_result("Smallest magnitude (SM)", smallest, "matrix-vector products")
    print_result("Largest magnitude (LM)", largest, "matrix-vector products")
    print_result("Shift-invert (sigma=0)", shifted, "inverse applications")


if __name__ == "__main__":
    main()
