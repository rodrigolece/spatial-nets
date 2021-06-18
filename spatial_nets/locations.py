#  import warnings
import pickle
import textwrap
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise

import graph_tool as gt

import spatial_nets.utils as utils

Array = np.ndarray
OptionalFloat = Optional[float]
ParamDict = Dict[str, OptionalFloat]


# __all__ = ["Locations", "CPC", "CPL", "RMSE"]


class LocationsDataClass:
    """
    Locations in 2D space with relative strengths and community attributes.

    % The model represents nodes of a network embedded in 2D space, and provides
    methods for calculating human mobility models.
    The production and attraction attributes measure the relative importance at
    each location (e.g. strenght, population, etc.).

    The community attribute ... TODO

    Attributes
    ----------
    N : int
        The number of locations.
    dmat : np.ndarray
        The NxN matrix of pairwise distances between locations.
    production : np.ndarray
        The vector containing a measure of outgoing strenght at each location.
    attraction : np.ndarray
        The vector containing a measure of incoming strength at each location.

    TODO
    comm_vec : np.ndarray or None, optional
        A vector containing the community (block) assignment of each location
        (default is None). The community methods will be extended in the
        future.

    """

    def __init__(self, data=None, coords=None, **kwargs):
        """
        Initialise the LocationsDataClass object.

        Parameters
        ----------
        data : object to be converted

            Current supported types are:
              a matrix containing OD data
              a tuple containing the production and attraction vectors
              a Graph object

            To be implemented:
              a dict  containing the production and attraction vectors
              a DataFrame containing OD data
              another LocationsDataClass object

        coords : array_like, optional


        """
        self.N: int = None
        self.B: int = None

        self._dmat: np.ndarray = None
        self._production: np.ndarray = None
        self._attraction: np.ndarray = None

        self._flow_data: sp.csr_matrix = None  # TODO: what about other possibilities?
        self.target_rows: np.ndarray = None
        self.target_cols: np.ndarray = None

        #  self._model_params = None
        #  self._affinity_mat: np.ndarray = None
        #  self._balancing_factors: Tuple[np.ndarray, np.ndarray] = None
        #  self._pvalues: Tuple[sp.csr_matrix, sp.csr_matrix] = None
        #  self._significance: float = None

        if isinstance(data, (np.integer, int)):
            if data < 1:
                raise ValueError("invalid nunmber of locations")
            self.N = data

        # For LocationsDataClass or Graph, we can assume the coords are stored
        # inside the object

        elif isinstance(data, LocationsDataClass):
            # TODO: self.copy()
            #  return data.copy()
            pass

        elif isinstance(data, (gt.Graph, gt.GraphView)):
            if (flow := data.ep.get("flow")) is not None:
                self.flow_data = gt.spectral.adjacency(data, weight=flow)
            else:
                raise ValueError("valid inpud needs a `flow` edge property")

            if (pos := data.vp.get("pos")) is not None:
                self.dmat = np.array(list(pos)) * [
                    [1],
                    [-1],
                ]  # due to cairo's coordinate system in the y-direction
            #  elif coords is not None:
            #      self.dmat = coords
            #  TODO: This should be covered by the elif below

        # In any other case, the coordinates should be passed (the coords
        #  argument takes precedence)

        elif coords is not None:

            if isinstance(data, tuple):
                assert len(data) == 2, "two sets of features are needed"
                prod, attrac = data
                self.production = prod
                self.attraction = attrac

            elif isinstance(data, (np.matrix, np.ndarray)):
                # TODO: test that this does not catch sparse matrix
                self.flow_data = data
            elif sp.issparse(data):
                self.flow_data = data

            self.dmat = coords
            # TODO: this should work for Graph object with no `pos` vp

        else:
            raise ValueError("coordinate matrix is needed")

        # TODO: is this good idea?
        for key, val in kwargs.items():
            setattr(self, key, val)

        # TODO
        #  if comm_vec is None:
        #      self.B = 1
        #      self.comm = np.ones(N, dtype=int)
        #  else:
        #      # self.k = len(np.unique(comm_vec))  #TODO:check below works better
        #      self.B = len(set(comm_vec))
        #      self.comm = np.array(comm_vec)

    def __str__(self) -> str:
        s = f"Locations object: N={self.N}"

        if self.B is not None:
            s += f" and B={self.B}"

        if self.flow_data is not None:
            s += "\n -- Data has been set"

        return textwrap.dedent(s)

    def __len__(self) -> int:
        """Returns the number of locations.

        Returns
        -------
        N : int
            The number of locations.

        """
        return self.N

    def copy(self):
        # TODO
        pass

    @property
    def dmat(self):
        return self._dmat

    @dmat.setter
    def dmat(self, location_mat) -> np.ndarray:
        N, m = location_mat.shape

        if (m != N) and (m != 2):
            raise ValueError("invalid shape for coordinate matrix")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid length")
        else:
            self.N = N

        # The location mat has the right shape; now do other tests
        if m == 2:
            self._dmat = pairwise.euclidean_distances(location_mat)

        elif np.any(location_mat < 0):
            raise ValueError("matrix should be non_negative")
        else:  # valid NxN matrix
            self._dmat = location_mat

    @property
    def production(self):
        return self._production

    @production.setter
    def production(self, prod: Array):
        """Should be set first."""
        prod = np.array(prod)
        N = len(prod)

        if not np.all(prod > 0):
            raise ValueError("input should have strictly positive entries")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid length")
        else:
            self.N = N

        self._production = prod

    @property
    def attraction(self):
        return self._attraction

    @attraction.setter
    def attraction(self, attrac: Array):
        """Should be set second."""
        attrac = np.array(attrac)

        if not np.all(attrac > 0):
            raise ValueError("input should have strictly positive entries")
        if len(attrac) != self.N:
            raise ValueError("input has invalid length")

        self._attraction = attrac

    @property
    def flow_data(self):
        """Retrieve the data stored inside the object."""
        return self._flow_data

    @flow_data.setter
    def flow_data(self, flow_mat: Array):
        """Set a copy of the data inside the object."""
        N, m = flow_mat.shape

        if (flow_mat < 0).sum() > 0:  # works for dense and spase matrices
            raise ValueError("data matrix should be non-negative")

        if m != N:
            raise ValueError("invalid shape for matrix")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid shape")

        else:
            self.N = N

        # NB: We remove the diagonal from the data matrix BEFORE calculating
        # the row and column sums
        if sp.issparse(flow_mat):
            self._flow_data = utils.sparsemat_remove_diag(flow_mat)  # creates a copy
        else:
            self._flow_data = flow_mat.copy()
            self._flow_data[np.diag_indices(self.N)] = 0

        # Row and column sums for dense and sparse matrices
        self.production = np.asarray(self._flow_data.sum(axis=1)).flatten()
        self.attraction = np.asarray(self._flow_data.sum(axis=0)).flatten()

        self.target_rows = self.production
        self.target_cols = self.attraction

        return None

    #  @property
    #  def significance(self):
    #      return self._significance

    #  @significance.setter
    #  def significane(self, alpha):
    #      if 0 < alpha < 1:
    #          self._significance = alpha
    #      else:
    #          raise ValueError("significance should lie inside (0, 1)")

    # End of the basic initialisation functions


# def _iterative_proportional_fit(f_mat: Array,
#                                 target_outflow: Array,
#                                 target_inflow: Array,
#                                 rel_tol: float = 1e-3,
#                                 ϵ: float = 1e-6,
#                                 max_iters: int = 1000,
#                                 return_vecs: bool = False,
#                                 safe_mode: bool = False) -> Array:
#     """
#     Apply the IPFP in the case of the doubly constrained model.
#
#     Parameters
#     ----------
#     f_mat : array_like
#         The affinity matrix.
#     target_outflow, target_inflow : array_like
#         Respectively the column-sum and row-sum that the doubly constrained
#         model should have.
#     rel_tol: float, optional
#     ϵ : float, optional
#         The value to use as 'zero' for convergence aspects.
#     max_iters : int, optional
#     return_vecs : bool, optional
#         Return the Ai and Bj normalising vectors (default is False).
#     safe_mode : bool, optional
#         Whether to use assert statements or not (default is False).
#
#     Returns
#     -------
#     np.array
#         The doubly constrained model (or the best approximation to it).
#
#     """
#     N, M = f_mat.shape
#
#     if not isinstance(target_inflow, np.ndarray):
#         target_outflow = np.array(target_outflow)
#     if not isinstance(target_inflow, np.ndarray):
#         target_inflow = np.array(target_inflow)
#
#     if safe_mode:
#         assert N == M, 'matrix is should be square'
#         assert len(target_outflow) == N,\
#             'target_outflow should have same size as matrix'
#         assert len(target_inflow) == N,\
#             'target_inflow should have same size as matrix'
#         assert not np.any(target_outflow < 0),\
#             'target_outflow should be non-negative'
#         assert not np.any(target_inflow < 0),\
#             'target_inflow should be non-negative'
#
#     idx_rows = (target_outflow < ϵ)
#     idx_cols = (target_inflow < ϵ)
#
#     # Verify that the superset of zeros is in target
#     model_outflow = np.asarray(f_mat.sum(axis=1)).flatten()
#     idx_rows_subset = (model_outflow < ϵ)
#     assert np.all(idx_rows[idx_rows_subset]),\
#         'unsatisfiable constraints (zero rows in model and non-zero in target)'
#
#     model_inflow = np.asarray(f_mat.sum(axis=0)).flatten()
#     idx_cols_subset = (model_inflow < ϵ)
#     assert np.all(idx_cols[idx_cols_subset]),\
#         'unsatisfiable constraints (zero cols in model and non-zero in target)'
#
#     # Initialise variables
#     num_iter = 0
#     converged = False
#     outscale_vec_old = np.ones(N)  # vector of A_i (associated to O_i)
#
#     while not converged and num_iter < max_iters:
#         num_iter += 1
#
#         # NB: M @ x is matrix multiplication (no need for x to be a column vec.)
#
#         # Update B_j
#         inscale_vec = np.ones(N)  # vector of B_j (associated to D_j)
#         # Σ_i F_ij * (A_i O_i)
#         denom = f_mat.T @ (outscale_vec_old * target_outflow)
#         np.reciprocal(denom, out=inscale_vec, where=(denom > ϵ))
#         # The where clause takes care of dividing by zero, A_i = 1 in that case
#
#         # Update A_i
#         outscale_vec = np.ones(N)
#         denom = f_mat @ (inscale_vec * target_inflow)  # Σ_j F_ij * (B_j D_j)
#         np.reciprocal(denom, out=outscale_vec, where=(denom > ϵ))
#         # Idem; B_j = 1
#
#         AO_vec = outscale_vec * target_outflow
#         BD_vec = inscale_vec * target_inflow
#         F_rescaled = BD_vec * f_mat * AO_vec[:, np.newaxis]
#
#         # Current error
#         model_outflow = np.asarray(F_rescaled.sum(axis=1)).flatten()
#         model_inflow = np.asarray(F_rescaled.sum(axis=0)).flatten()
#
#         # Note we exclude zero rows or zero columns
#         colsum_error = np.linalg.norm(
#             model_outflow[~idx_rows] / target_outflow[~idx_rows] - 1.0, np.inf)
#         rowsum_error = np.linalg.norm(
#             model_inflow[~idx_cols] / target_inflow[~idx_cols] - 1.0, np.inf)
#         iter_error = np.linalg.norm(outscale_vec - outscale_vec_old)
#
#         # After error, update variable
#         outscale_vec_old = outscale_vec
#
#         if colsum_error < rel_tol and rowsum_error < rel_tol:
#             converged = True
#         elif iter_error < ϵ:
#             warnings.warn(
#                 f'Iterations converged ({num_iter} iters) but constraints cannot be satisfied')
#             # warnings.warn(f'{colsum_error:.3f}, {rowsum_error:.3f}')
#             converged = True
#
#     if not converged:
#         raise FailedToConverge(
#             f'reached maximum number of iterations ({max_iters})')
#
#     return (F_rescaled, outscale_vec, inscale_vec) if return_vecs else F_rescaled


# class FailedToConverge(Exception):
#     """Raised when an iterative method failed to converge."""
#
#     pass


class DataNotSet(Exception):
    """Raised when a method needs to access the data and it has not been set."""

    pass


def save_model(
    filename, locs=None, constraint_type=None, grav_params=None, balancing_factors=None
):
    """
    Save a model storing optionally the locations object, the parameters or
    the balancing factors.

    """
    save_dict = {}

    if locs is not None:
        assert locs.data is not None
        save_dict["locs"] = locs

    if constraint_type is not None:
        assert constraint_type in [
            "unconstrained",
            "production",
            "attraction",
            "doubly",
        ]
        save_dict["constraint_type"] = constraint_type

    if grav_params is not None:
        assert len(grav_params) == 3
        save_dict["grav_params"] = grav_params

    if balancing_factors is not None:
        assert len(balancing_factors) == 2
        save_dict["balancing_factors"] = balancing_factors

    if len(save_dict) > 0:
        with open(filename, "wb") as f:
            pickle.dump(save_dict, f)
    else:
        print("Need to provide arguments")

    return None


def load_gravity_model(filename, locs=None, return_locs=False):
    """
    Load gravity model using the saved parameters and optionally the saved
    balancing factors.

    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
        constraint_type = data.get("constraint_type")

        if locs is not None:
            pass
        elif "locs" in data:
            locs = data["locs"]
        else:
            raise AttributeError("`locs` not provided")

        fmat = locs.gravity_matrix(*data["grav_params"])

    if constraint_type is None:
        out = fmat

    elif constraint_type == "unconstrained":
        K = locs.data.sum() / fmat.sum()
        out = K * fmat

    elif constraint_type == "production":
        pmat = locs.probability_matrix(fmat, "production")
        out = locs.data_out[:, np.newaxis] * pmat

    elif constraint_type == "attraction":
        pmat = locs.probability_matrix(fmat, "attraction")
        out = pmat * locs.data_in[np.newaxis, :]

    elif constraint_type == "doubly" and "balancing_factors" in data:
        a, b = data["balancing_factors"]
        out = a[:, np.newaxis] * fmat * b[np.newaxis, :]

    elif constraint_type == "doubly":
        out = simple_ipf(fmat, locs.data_out, locs.data_in)

    else:
        pass

    return (out, locs) if return_locs else out
