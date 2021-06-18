from typing import Optional

import numpy as np
import scipy.sparse as sp

from spatial_nets.base import Model


class UnconstrainedModel(Model):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def transform(self, mat: np.ndarray) -> np.ndarray:
        K = self.total_flow_ / mat.sum()
        return K * mat


class ProductionConstrained(Model):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def transform(self, mat: np.ndarray) -> np.ndarray:
        """
        Production constrained model. The probabilities are stored as an attribute.

        Parameters
        ----------
        mat : array_like
            The input matrix.

        Returns
        -------
        np.array
            % Normalised probability matrix with rows that sum to one.

        """
        assert not sp.issparse(mat), "sparse matrices are not supported"
        # We assume that mat is NOT of type  sparse so that sums are flat

        pmat = mat.astype(float)  # to avoid problems with division
        row_sum = mat.sum(axis=1)
        idx = row_sum > 0
        pmat[idx] = pmat[idx] / row_sum[idx, np.newaxis]
        self.probabilities_ = pmat

        return self.target_rows_[:, np.newaxis] * pmat


class AttractionConstrained(Model):
    def transform(self, mat: np.ndarray) -> np.ndarray:
        """
        Normalise a matrix as column-stochastic.

        Parameters
        ----------
        mat : array_like
            The input matrix.

        Returns
        -------
        np.array
            % Normalised probability matrix with columns that sum to one.

        """

        assert not sp.issparse(mat), "sparse matrices are not supported"
        # We assume that mat is NOT of type  sparse so that sums are flat

        pmat = mat.astype(float)  # to avoid problems with division
        col_sum = mat.sum(axis=0)
        idx = col_sum > 0
        pmat[:, idx] = pmat[:, idx] / col_sum[np.newaxis, idx]
        self.probabilities_ = pmat

        return pmat * self.target_cols_[np.newaxis, :]


class DoublyConstrained(Model):
    def __init__(
        self,
        constraint: Optional[str] = None,
        tol: float = 1e-3,
        maxiters: int = 100,
        verbose: bool = False,
    ):
        super().__init__(constraint=constraint)
        self.tol = tol
        self.maxiters = maxiters
        self.verbose = verbose

    def transform(self, mat: np.ndarray) -> np.ndarray:
        prediction, a, b = simple_ipf(
            mat,
            self.target_rows_,
            self.target_cols_,
            self.tol,
            self.maxiters,
            self.verbose,
        )
        self.balancing_factors_ = (a, b)

        # NB: by default we factor the probabilities as production-constrained
        a = a / self.target_rows_
        self.probabilities_ = a[:, np.newaxis] * mat * b[np.newaxis, :]

        return prediction


def simple_ipf(
    mat: np.ndarray,
    target_rows: np.ndarray = None,
    target_cols: np.ndarray = None,
    tol: float = 1e-3,
    maxiters: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """ Iterative proportional fit procedure."""

    if maxiters < 1:
        raise ValueError("invalid number of maximum iterations")

    N, _ = mat.shape

    if target_rows is None:
        target_rows = np.ones(N)
    if target_cols is None:
        target_cols = np.ones(N)

    niter = 0
    b = np.ones(N)

    while niter < maxiters:
        a = target_rows / (mat @ b)
        b = target_cols / (mat.T @ a)

        out = a[:, np.newaxis] * mat * b[np.newaxis, :]
        bool_row = np.allclose(out.sum(axis=1), target_rows, atol=tol)
        # rel_tol instead? this is the problem I address inside CPC, etc.
        bool_col = np.allclose(out.sum(axis=0), target_cols, atol=tol)

        niter += 1

        if bool_row and bool_col:
            break

    if verbose:
        print(f"Nb iters until convergence: {niter}")

    return out, a, b
