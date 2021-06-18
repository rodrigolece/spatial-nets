from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy import stats
from tabulate import tabulate

from spatial_nets.locations import LocationsDataClass, DataNotSet


class Model(ABC):
    @abstractmethod
    def transform(self, mat: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    def __init__(self, constraint: Optional[str] = None):
        """

        Parameters
        ----------
        constraint  : str
            The type of constraint to use.

        """
        if constraint not in [
            None,
            "production",
            "attraction",
            "doubly",
        ]:
            raise ValueError("invalid constraint")

        self.constraint: Optional[str] = constraint

        self.probabilities_: np.ndarray = None
        # TODO: maybe move out balancing factors and put in doubly constrained?
        self.balancing_factors_: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.coef_ = None

    def fit(self, data: LocationsDataClass):
        self.total_flow_ = data.flow_data.sum()
        self.target_rows_ = data.target_rows
        self.target_cols_ = data.target_cols

        return self

    def fit_transform(
        self, data: LocationsDataClass, mat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(data).transform(mat)

    #  def constrained_model(
    #      self, f_mat: np.ndarray, constraint_type: str, maxiters=500, verbose=False
    #  ) -> np.ndarray:
    #      """
    #      Calculate the constrained flux from the affinity matrix f_ij.

    #      Parameters
    #      ----------
    #      f_mat : array_like
    #      constrained : {'unconstrained', 'production', 'attraction', 'doubly'}

    #      Returns
    #      -------
    #      np.array
    #          NxN constrained flow matrix.

    #      """
    #      if self.total_flow_ is None:
    #          raise DataNotSet("the data for the constraints is needed")

    #      assert constraint_type in [
    #          "unconstrained",
    #          "production",
    #          "attraction",
    #          "doubly",
    #      ], f"invalid constraint {constraint_type}"

    #      if constraint_type == "unconstrained":
    #          K = self.total_flow_ / f_mat.sum()
    #          out = K * f_mat

    #      elif constraint_type == "production":
    #          p_mat = self.probability_matrix(f_mat, constraint_type)
    #          out = self.target_rows[:, np.newaxis] * p_mat

    #      elif constraint_type == "attraction":
    #          p_mat = self.probability_matrix(f_mat, constraint_type)
    #          out = p_mat * self.targe_cols[np.newaxis, :]

    #      elif constraint_type == "doubly":
    #          out = simple_ipf(
    #              f_mat,
    #              self.target_rows,
    #              self.targe_cols,
    #              maxiters=maxiters,
    #              verbose=verbose,
    #          )

    #  return out

    #  def draw_multinomial(self, p_mat, constraint_type, seed=0):
    #      """Draw from the constrained model using multinomial distribution."""
    #      assert constraint_type in [
    #          "production",
    #          "attraction",
    #      ], f"invalid constraint {constraint_type}"

    #      rng = np.random.RandomState(seed)

    #      out = sp.lil_matrix((self.N, self.N), dtype=int)

    #      if constraint_type == "production":
    #          for i in range(self.N):
    #              draw = rng.multinomial(self.target_rows[i], p_mat[i])
    #              (j,) = draw.nonzero()
    #              out[i, j] = draw[j]

    #      elif constraint_type == "attraction":
    #          for j in range(self.N):
    #              draw = rng.multinomial(self.targe_cols[j], p_mat[:, j])
    #              (i,) = draw.nonzero()
    #              out[i, j] = draw[i]

    #      return out.tocsr()

    #  def score_draws(self, pmat, constraint_type, nb_repeats=20):
    #      """Calculate goodness-of-fit measures for multiple draws."""
    #      out = np.zeros((nb_repeats, 3))

    #      for k in range(nb_repeats):
    #          T = self.draw_multinomial(pmat, constraint_type, seed=k)
    #          out[k] = (
    #              CPC(T, self.flow_data),
    #              CPL(T, self.flow_data),
    #              RMSE(T, self.flow_data),
    #          )

    #      return out

    # def CPC_from_model(self, model: str,
    #                    constraint_type: str,
    #                    params: ParamDict) -> float:
    #     """
    #     Use a particular model and constraint type to calculate the CPC fit.
    #
    #     Parameters
    #     ----------
    #     model : {'gravity', 'radiation', 'io'}
    #     params : dict
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     Data = self.flow_data
    #     return CPC(Data, self.constrained_model(model, constraint_type, params))
    #

    #  def pvalues_approx(
    #      self, pmat: np.ndarray, constraint_type: str
    #  ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    #      """
    #      Calculate the p-values using the normal approximation.

    #      Parameters
    #      ----------
    #      pmat : np.ndarray
    #      constraint_type : {'production', 'attraction'}

    #      Returns
    #      -------
    #      Tuple[csr_matrix, csr_matrix]
    #          The first matrix stores the p-values for the hypothesis
    #          "the observation is not signifcantly larger than the predicted mean"
    #          and the second matrix for the hypothesis "the observation is not
    #          signifcantly smaller than the predicted mean."

    #      """
    #      if self.flow_data is None:
    #          raise DataNotSet("the data for comparison is needed")

    #      assert constraint_type in [
    #          "production",
    #          "attraction",
    #      ], f"invalid constraint {constraint_type}"

    #      # Entry-wise first and second moments (binomial model)
    #      if constraint_type == "production":
    #          Exp = self.target_rows[:, np.newaxis] * pmat
    #      elif constraint_type == "attraction":
    #          Exp = pmat * self.targe_cols[np.newaxis, :]
    #      Std = np.sqrt(Exp * (1 - pmat))

    #      i, j = self.flow_data.nonzero()
    #      shp = self.flow_data.shape
    #      Z_score = (self.flow_data[i, j] - Exp[i, j]) / Std[i, j]
    #      Z_score = np.asarray(Z_score).flatten()

    #      data_plus = stats.norm.cdf(-Z_score)
    #      data_minus = stats.norm.cdf(Z_score)

    #      plus = sp.csr_matrix((data_plus, (i, j)), shape=shp)
    #      minus = sp.csr_matrix((data_minus, (i, j)), shape=shp)

    #      return (plus, minus)

    #  def pvalues_exact(
    #      self, pmat: np.ndarray, constraint_type: str
    #  ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    #      """
    #      Calculate the p-values using the exact binomial distributions.

    #      Parameters
    #      ----------
    #      pmat : np.ndarray
    #      constraint_type : {'production', 'attraction'}

    #      Returns
    #      -------
    #      Tuple[csr_matrix, csr_matrix]
    #          The first matrix stores the p-values for the hypothesis
    #          "the observation is not signifcantly larger than the predicted mean"
    #          and the second matrix for the hypothesis "the observation is not
    #          signifcantly smaller than the predicted mean."

    #      """
    #      if self.flow_data is None:
    #          raise DataNotSet("the data for comparison is needed")

    #      assert constraint_type in [
    #          "production",
    #          "attraction",
    #      ], f"invalid constraint {constraint_type}"

    #      ii, jj = self.flow_data.nonzero()
    #      shp = self.flow_data.shape
    #      n = len(ii)
    #      data_plus, data_minus = np.zeros(n), np.zeros(n)

    #      # The target N is either the row or the column sum
    #      Ns = (
    #          self.target_rows[ii]
    #          if constraint_type == "production"
    #          else self.targe_cols[jj]
    #      )

    #      for k in range(n):
    #          i, j = ii[k], jj[k]
    #          x, n, p = self.flow_data[i, j], Ns[k], pmat[i, j]
    #          data_plus[k] = stats.binom_test(x, n=n, p=p, alternative="greater")
    #          data_minus[k] = stats.binom_test(x, n=n, p=p, alternative="less")

    #      plus = sp.csr_matrix((data_plus, (ii, jj)), shape=shp)
    #      minus = sp.csr_matrix((data_minus, (ii, jj)), shape=shp)

    #      return (plus, minus)

    #  def significant_edges(
    #      self,
    #      pmat: np.ndarray,
    #      constraint_type: str,
    #      significance: float = 0.01,
    #      exact: bool = False,
    #      verbose: bool = False,
    #  ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    #      """
    #      Calculate the significant edges according to a binomial test (or z-test).

    #      Parameters
    #      ----------
    #      pmat : np.ndarray
    #      constraint_type : {'production', 'attraction'}
    #      significance : float, optional
    #          By default 0.01.
    #      exact : bool, optional
    #          Whether to use the exact calculation as opposed the normal
    #          approximation (z-test). The default is False.
    #      verbose : bool, optional

    #      Returns
    #      -------
    #      Tuple[csr_matrix, csr_matrix]

    #      """
    #      method_name = "pvalues_" + ("exact" if exact else "approx")
    #      method = getattr(self, method_name)
    #      plus, minus = method(pmat, constraint_type)
    #      mask = self.flow_data.astype(bool)

    #      sig_plus = (plus < significance).multiply(mask)
    #      # needed for elementwise multiplication because sparse is matrix type
    #      sig_minus = (minus < significance).multiply(mask)

    #      if verbose:
    #          n, nplus, nminus = mask.nnz, sig_plus.nnz, sig_minus.nnz
    #          nzero = n - nplus - nminus
    #          tab = [
    #              ["Positive (observed larger)", nplus, f"{100*nplus/n:.2f}"],
    #              ["Negative (model larger)", nminus, f"{100*nminus/n:.2f}"],
    #              ["Not-significant:", nzero, f"{100*nzero/n:.2f}"],
    #              ["Total", n, "100.00"],
    #          ]
    #          print(tabulate(tab, headers=["", "Nb", "%"]))

    #      return sig_plus, sig_minus
