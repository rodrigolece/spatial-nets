import warnings
import numpy as np
import numpy.ma as ma
from scipy import stats, sparse, optimize
from sklearn import linear_model
from sklearn.metrics import pairwise
import textwrap
from tabulate import tabulate
from typing import Dict, Tuple, Optional

import utils

Array = np.array
OptionalFloat = Optional[float]
ParamDict = Dict[str, OptionalFloat]


__all__ = ['Locations', 'binomial_pvalues', 'CPC', 'CPL']


class Locations(object):
    """
    Locations in 2D space with relevance and community attributes.

    The model represents nodes of a network embedded in 2D space, and provides
    methods for calculating human mobility models. The relevance attribute
    captures for example the population at each location (or some other measure
    of intrinsic strenght). The community attribute is currently optional but
    the model will be extended to make full use of it.

    Attributes
    ----------
    N : int
        Number of nodes.
    dmat : np.array
        The matrix of pariwise distances between locations.
    relevance : np.array
        A measure of the relevance or intrinsic strength of the locations.
    in_relevance : np.array
        A second measure of the relevance (to be used as the in-attraction
        whereas relevance is the out-attraction).
    k : int
        The number of communities.
    comm : np.array
        The community of the nodes.
    data: np.array or None
        NxN matrix that hold data observations.

    """

    def __init__(self, N: int,
                 location_mat: Array,
                 relevance_vec: Array,
                 in_relevance_vec: Array = None,
                 comm_vec: Array = None,
                 use_coords: bool = False) -> None:
        """
        Instantiate the Locations object.

        Parameters
        ----------
        N : int
            The number of locations.
        location_mat : array_like
            The NxN matrix of pairwise distances between locations.
            Alternatively, if `use_coords` is set to True this should be the Nx2
            array of coordinates.
        relevance_vec : array_like
            The primary relevance vector of each location. For example, this
            could be the population. If in_relevance_vec is also provided (in
            the setting of a full radiation model, with vectors calculated from
            the empirical data) then this vector is taken to be the out_relevance.
        in_relevance_vec : array_like or None, optional
            Provided to distintiguish two sets of intrinsic measures of relevance
            for each location (default is None). If provided, relevance_vec is
            taken as the out_relevance.
        comm_vec : array_like or None, optional
            A vector containing the community (block) assignment of each location
            (default is None). The community methods will be extended in the
            future.
        use_coords : bool, optional
            If set to True, `location_mat` should correspond to the Nx2 array of
            coordinates and the class uses the pairwise euclidean distance
            between locations.

        """
        assert (location_mat.shape == (N, 2) and use_coords is True) \
            or location_mat.shape == (N, N), \
            'location_mat is not NxN (distances), or Nx2 (coords) with `use_coords=True`'
        assert len(relevance_vec) == N, 'relevance_vec should have length N'

        if use_coords:
            self.dmat = pairwise.euclidean_distances(location_mat)
        else:
            assert np.all(location_mat >= 0.0), \
                'distance_matrix should be non_negative'
            self.dmat = location_mat

        self.N = N
        self.relevance = np.array(relevance_vec)  # to convert list or Series
        self._data = None
        self._datasparseflag = False

        assert np.all(self.relevance >= 0.0), \
            'relevance of locations should be non-negative'

        if in_relevance_vec is None:
            self.in_relevance = self.relevance  # binding to avoid duplicate data
            self._tworelevancesflag = False
        else:
            assert len(in_relevance_vec) == N, \
                'in_relevance_vec should have length N'
            self.in_relevance = np.array(in_relevance_vec)
            self._tworelevancesflag = True
            assert np.all(self.in_relevance >= 0.0), \
                'in_relevance of locations should be non-negative'

        if comm_vec is None:
            self.k = 1
            self.comm = np.ones(N, dtype=int)
        else:
            self.k = len(np.unique(comm_vec))
            self.comm = comm_vec

        return None

    def __str__(self) -> str:
        if self._tworelevancesflag:
            s = f'''
                Locations object (in- and out-relevance provided):
                    N={self.N} and k={self.k}'''
        else:
            s = f'Locations object: N={self.N} and k={self.k}'

        if self.data is not None:
            s = s + '\n -- Data has been set'

        return textwrap.dedent(s)

    @property
    def data(self):
        """Retrieve the data stored inside the object."""
        return self._data

    @data.setter
    def data(self, data_mat: Array):
        """Set a copy of the data inside the object."""
        assert np.sum(data_mat < 0) == 0, f'data matrix should be non-negative'
        assert data_mat.shape == (self.N, self.N), 'data_mat is not NxN'

        outflow = data_mat.sum(axis=1)
        inflow = data_mat.sum(axis=0)

        if sparse.issparse(data_mat):
            self._datasparseflag = True
            self._data = utils.sparsemat_remove_diag(data_mat)  # creates a copy
            outflow = np.asarray(outflow).flatten()
            inflow = np.asarray(inflow).flatten()
        else:
            self._data = data_mat.copy()
            self._data[np.diag_indices(self.N)] = 0

        self.data_out = outflow
        self.data_in = inflow

        return None

    def io_matrix(self, threshold: float = np.inf, **kwargs) -> Array:
        """
        Calculate intervening opportunities matrix.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).

        Returns
        -------
        np.array
            nxn matrix containing the intervening opportunities.

        """
        N = self.N
        S = np.zeros((N, N), dtype=self.in_relevance.dtype.type)

        # We use masked arrays
        m_dmat = ma.masked_greater_equal(self.dmat, threshold)
        idx_mat = ma.argsort(m_dmat, axis=1)  # sort each row

        m_rel = ma.masked_array(self.in_relevance)

        for i, idx in enumerate(idx_mat):
            m_rel.mask = m_dmat[i].mask
            m_rel[i] = ma.masked        # the populations at origin is not used
            m_rel[idx[-1]] = ma.masked  # the furthest population is not used

            rolled_idx = np.roll(idx, 1)
            S[i, idx] = m_rel[rolled_idx].cumsum()

        return S

    def radiation_matrix(self, threshold: float = np.inf,
                         finite_correction: bool = True,
                         **kwargs) -> Array:
        """
        Calculate the radiation matrix.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).
        finite_correction :
            Wether to normalise each row using the term $1/(1 - m_i / M)$.

        Returns
        -------
        np.array
            nxn matrix of flows predicted by the radiation model.

        """
        S = self.io_matrix(threshold=threshold)
        m = self.relevance[:, np.newaxis]  # column vector
        n = self.in_relevance[:, np.newaxis]  # column vector

        num = m * n.T
        den = (m + S) * (m + n.T + S)
        with np.errstate(invalid='ignore'):
            # 0/0 inside arrays is invalid floating point, not divide error;
            # to cover the case m_i + S_ij = 0 (can occur when m_i is zero)
            out = num / den
            out[np.isnan(out)] = 0.0

        out[np.diag_indices_from(out)] = 0.0

        if finite_correction:
            M = np.sum(m)
            norm = (1 - m / M)  # m is col vec and so is norm
            out /= norm  # this is right division, and therefore row scaling

        return out

    def gravity_matrix(self, γ: float,
                       α: float = 1.0, β: float = 1.0, **kwargs) -> Array:
        """
        Calculate the gravity matrix.

        Parameters
        ----------
        γ : float
            Decay exponent.

        Returns
        -------
        np.array
            nxn matrix of flows predicted by the gravity model.

        """
        m = self.relevance[:, np.newaxis]  # column vector
        n = self.in_relevance[:, np.newaxis]  # column vector

        with np.errstate(divide='ignore', invalid='raise'):
            # 0**(-γ) raises divide error, 0 * ∞ raises invalid error in multiply
            dmat_power = self.dmat ** (-γ)
            dmat_power[np.diag_indices(self.N)] = 0.0
            # note this doesn't fix problems outside diagonal, this is on purpose

            out = (m ** α) * (n.T ** β) * dmat_power

        return out

    def probability_matrix(self, f_mat: Array, constraint_type: str) -> Array:
        """
        Normalise the affinity matrix f_ij to get a probability.

        Parameters
        ----------
        f_mat : array_like
            A matrix that constains the flows predicted by one of the models.
        constraint_type  : str
            The type of constraint to use. Only 'production' or 'attraction'
            make sense in this context.

        Returns
        -------
        np.array
            nxn normalised probability matrix with either rows or colums that
            sum to one.

        """
        assert constraint_type in ['production', 'attraction'], \
            f'invalid constraint {constraint_type}'

        assert not sparse.issparse(f_mat), \
            'sparse matrices are not supported'

        p_mat = f_mat.astype(float)  # to avoid problems with division

        if constraint_type == 'production':
            # We assume that f_mat is NOT of type  sparse so that sums are flat
            row_sum = f_mat.sum(axis=1)
            idx = (row_sum > 0)
            p_mat[idx] /= row_sum[idx, np.newaxis]

        elif constraint_type == 'attraction':
            col_sum = f_mat.sum(axis=0)
            idx = (col_sum > 0)
            p_mat[:, idx] /= col_sum[np.newaxis, idx]

        return p_mat

    def constrained_flux(self, f_mat: Array,
                         constraint_type: str) -> Array:
        """
        Calculate the constrained flux from the affinity matrix f_ij.

        Parameters
        ----------
        f_mat : array_like
        constrained : {'production', 'attraction', 'unconstrained', 'doubly'}

        Returns
        -------
        np.array
            nxn constrained flow matrix.

        """
        if self.data is None:
            raise DataNotSet('the data for the constraints is needed')

        assert constraint_type in ['production', 'attraction', 'unconstrained',
                                   'doubly'], \
            f'invalid constraint {constraint_type}'

        if constraint_type == 'unconstrained':
            ϕ = self.data.sum() * f_mat / f_mat.sum()

        elif constraint_type == 'production':
            p_mat = self.probability_matrix(f_mat, constraint_type)
            ϕ = p_mat * self.data_out[:, np.newaxis]

        elif constraint_type == 'attraction':
            p_mat = self.probability_matrix(f_mat, constraint_type)
            ϕ = self.data_in[np.newaxis, :] * p_mat

        elif constraint_type == 'doubly':
            ϕ = _iterative_proportional_fit(f_mat, self.data_out, self.data_in)

        return ϕ

    def constrained_model(self, model: str,
                          constraint_type: str,
                          params: ParamDict,
                          rounded: bool = False,
                          sparse: bool = False) -> Array:
        """
        Calculate the prediction of a model and apply the constraint type.

        Parameters
        ----------
        model : {'gravity', 'radiation', 'io'}
        params : dict
        rounded : bool, optional
            Whether to round the output of the model or not (default is False).
        sparse : bool, optional
            Whether to return the output model as a sparse matrix (default is
            False). If set to True, the output is rounded irrespective of the
            kwarg `rounded` which is ignored.

        Returns
        -------
        np.array

        """
        method_name = f'{model}_matrix'
        method = getattr(self, method_name, None)
        if method:
            try:
                f_mat = method(**params)
            except Exception as e:
                message = f'Failed to apply model {model} with parameter {params}'
                raise ValueError(message) from e
        else:
            raise NotImplementedError(f'model {model} not implemented')

        try:
            ϕ = self.constrained_flux(f_mat, constraint_type)
        except Exception as e:
            message = f'Failed to apply constraint type {model}'
            raise ValueError(message) from e

        if rounded or sparse:
            ϕ = np.round(ϕ).astype(int)
        if sparse:
            ϕ = sparse.csr_matrix(ϕ)

        return ϕ

    def CPC_from_model(self, model: str,
                       constraint_type: str,
                       params: ParamDict) -> float:
        """
        Use a particular model and constraint type to calculate the CPC fit.

        Parameters
        ----------
        model : {'gravity', 'radiation', 'io'}
        params : dict

        Returns
        -------
        float

        """
        Data = self.data
        return CPC(Data, self.constrained_model(model, constraint_type, params))

    def gravity_calibrate_gamma(self, constraint_type: str = 'production',
                                bounds: Tuple[float, float] = (1e-3, 3)) -> float:
        """
        Calibrate the gravity power law parameter by maximising the CPC metric.

        Parameters
        ----------
        constraint_type : str, optional
        bounds : tuple, optional
            Default is (1e-3, 3).

        Returns
        -------
        float

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        res = optimize.minimize_scalar(
            lambda x: -self.CPC_from_model('gravity', constraint_type, dict(γ=x)),
            method='Bounded',
            bounds=bounds
        )
        st = res.status
        assert st == 0, f'optimization exit status is non-zero: {st}'

        return res.x

    def gravity_calibrate_all(self, verbose: bool = False) -> Tuple[float, float, float]:
        """
        Calibrate the all of the gravity parameters using linear least squares.

        Parameters
        ----------
        verbose: bool, optional
            The defalt is False


        Returns
        -------
        α, β, γ : float

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        N = self.N

        # The observations
        i, j = self.data.nonzero()
        y = np.asarray(self.data[i, j]).flatten()

        # The model data
        m = self.data_out[:, np.newaxis]  # col. vec.
        n = self.data_in[np.newaxis, :]   # row vec.

        m_repeated = np.repeat(m, N, axis=1)
        n_repeated = np.repeat(n, N, axis=0)

        X = np.vstack((m_repeated[i, j], n_repeated[i, j], self.dmat[i, j])).T
        logX = np.log(X)

        if verbose:
            ρ, _ = stats.pearsonr(logX[:, 0], logX[:, 1])
            print(f'Correlation between log O and D columns, ρ : {ρ:.3f}\n')

        reg = linear_model.LinearRegression()
        reg.fit(logX, np.log(y))

        α, β, γ = reg.coef_
        γ = -γ
        # k = np.exp(reg.intercept_)  # in practice works worse than unconstrained

        if γ < 0.5:
            warnings.warn(f'γ = {γ:.3f}')

        return α, β, γ

    def significant_binomial(self, model: str,
                             significance: float = 0.01,
                             return_negative: bool = False,
                             verbose: bool = False) -> Tuple[float, float]:
        """
        Calculate the significant entries according to a production-constrained
        binomial model.

        Parameters
        ----------
        model : str
            The model to use, either 'gravity' or 'radiation'.
        significance : float, optional
            By default 0.01.
        return_negative : bool, optional
            Whether to output the entries for which the model is significanly
            larger than the observations (negative edges). The defualt is False,
            and the method returns the entries for which the observations are
            significantly larger (positive edges).

        Returns
        -------
        positive_edges : (np.array, np.array)
            A tuple of positive edges. Alternatively if return_negative is set
            to True, return a second tuple.

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        assert model in ['gravity', 'radiation'], f'invalid model {model}'

        if model == 'gravity':
            α, β, γ = self.gravity_calibrate_all(verbose=False)
            f_mat = self.gravity_matrix(γ, α, β)
            p_mat = self.probability_matrix(f_mat, 'production')
        elif model == 'radiation':
            p_mat = self.radiation_matrix(finite_correction=True)

        # Entry-wise first and second moments (binomial model)
        Data = self.data
        Exp = p_mat * self.data_out[:, np.newaxis]
        Std = np.sqrt(Exp * (1 - p_mat))

        i, j = (Data > 0).multiply(np.round(Exp) >= 1.0).nonzero()

        Z_score = np.asarray((Data[i, j] - Exp[i, j]) / Std[i, j]).squeeze()

        idx_plus = stats.norm.cdf(-Z_score) < significance
        idx_minus = stats.norm.cdf(Z_score) < significance
        # idx_zero = ~np.bitwise_or(idx_plus, idx_minus)

        if verbose:
            n, nplus, nminus = len(i), np.sum(idx_plus), np.sum(idx_minus)
            nzero = n - nplus - nminus
            tab = [['Positive (observed larger)', nplus, f'{100*nplus/n:.2f}'],
                   ['Negative (model larger)', nminus, f'{100*nminus/n:.2f}'],
                   ['Not-significant:', nzero, f'{100*nzero/n:.2f}'],
                   ['Total', n, '100.00']]
            print(tabulate(tab, headers=['', 'Nb', '%']))

        plus = i[idx_plus], j[idx_plus]
        minus = i[idx_minus], j[idx_minus]

        return (plus, minus) if return_negative else plus


def binomial_pvalues(N : int, p : float, x : int) -> Tuple[float, float]:
    """
    Calculate the common part of commuters between two models.

    Parameters
    ----------
    N, p : int, float
        The parameters of the binomial distribution.
    x : int
        The observation.

    Returns
    -------
    float, float

    """
    N, x = int(round(N)), int(round(x))
    B = stats.binom(N, p)

    # Test below seems to be significantly slower
    # if x < N*p:
    #     probas = B.pmf(range(x+1)).cumsum()
    #     out =  1 - probas[x-1], probas[x]
    # else:
    #     probas = B.pmf(range(N, x-1, -1)).cumsum()
    #     out = probas[-1], 1 - probas[-2]

    probas = B.pmf(range(x+1)).cumsum()
    out =  1 - round(probas[x-1], 6), round(probas[x], 6)
    # due to numerical errors I can get cumsum larger than 1 and this is why
    # we need to use rounding

    return out


def CPC(F1: Array, F2: Array) -> float:
    """
    Calculate the common part of commuters between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    denom_mat = F1 + F2
    num_mat = denom_mat - np.abs(F1 - F2)  # this implements 2*min(F1, F2)

    return np.sum(num_mat) / np.sum(denom_mat)


def CPL(F1: Array, F2: Array) -> float:
    """
    Calculate the common part of links between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    bool_F1 = F1 > 0
    bool_F2 = F2 > 0

    # sparse implements np.matrix and * is mat-mat multplication, not elementwise
    if sparse.issparse(F1):
        prod = bool_F1.multiply(bool_F2)
    elif sparse.issparse(F2):
        prod = bool_F2.multiply(bool_F1)
    else:
        prod = bool_F1 * bool_F2

    return 2 * np.sum(prod) / (np.sum(bool_F1) + np.sum(bool_F2))


def _iterative_proportional_fit(f_mat: Array,
                                target_outflow: Array,
                                target_inflow: Array,
                                rel_tol: float = 1e-3,
                                ϵ: float = 1e-6,
                                max_iters: int = 1000,
                                return_vecs: bool = False,
                                safe_mode: bool = False) -> Array:
    """
    Apply the IPFP in the case of the doubly constrained model.

    Parameters
    ----------
    f_mat : array_like
        The affinity matrix.
    target_outflow, target_inflow : array_like
        Respectively the column-sum and row-sum that the doubly constrained
        model should have.
    rel_tol: float, optional
    ϵ : float, optional
        The value to use as 'zero' for convergence aspects.
    max_iters : int, optional
    return_vecs : bool, optional
        Return the Ai and Bj normalising vectors (default is False).
    safe_mode : bool, optional
        Whether to use assert statements or not (default is False).

    Returns
    -------
    np.array
        The doubly constrained model (or the best approximation to it).

    """
    N, M = f_mat.shape

    if not isinstance(target_inflow, np.ndarray):
        target_outflow = np.array(target_outflow)
    if not isinstance(target_inflow, np.ndarray):
        target_inflow = np.array(target_inflow)

    if safe_mode:
        assert N == M, 'matrix is should be square'
        assert len(target_outflow) == N,\
            'target_outflow should have same size as matrix'
        assert len(target_inflow) == N,\
            'target_inflow should have same size as matrix'
        assert np.all(target_outflow >= 0.0),\
            'target_outflow should be non-negative'
        assert np.all(target_inflow >= 0.0),\
            'target_inflow should be non-negative'

    idx_rows = (target_outflow < ϵ)
    idx_cols = (target_inflow < ϵ)

    # Verify that the superset of zeros is in target
    model_outflow = np.asarray(f_mat.sum(axis=1)).flatten()
    idx_rows_subset = (model_outflow < ϵ)
    assert np.all(idx_rows[idx_rows_subset]),\
        'unsatisfiable constraints (zero rows in model and non-zero in target)'

    model_inflow = np.asarray(f_mat.sum(axis=0)).flatten()
    idx_cols_subset = (model_inflow < ϵ)
    assert np.all(idx_cols[idx_cols_subset]),\
        'unsatisfiable constraints (zero cols in model and non-zero in target)'

    # Initialise variables
    num_iter = 0
    converged = False
    outscale_vec_old = np.ones(N)  # vector of A_i (associated to O_i)

    while converged is False and num_iter < max_iters:
        num_iter += 1

        # NB: M @ x is matrix multiplication (no need for x to be a column vec.)

        # Update B_j
        inscale_vec = np.ones(N)  # vector of B_j (associated to D_j)
        # Σ_i F_ij * (A_i O_i)
        denom = f_mat.T @ (outscale_vec_old * target_outflow)
        np.reciprocal(denom, out=inscale_vec, where=(denom > ϵ))
        # The where clause takes care of dividing by zero, A_i = 1 in that case

        # Update A_i
        outscale_vec = np.ones(N)
        denom = f_mat @ (inscale_vec * target_inflow)  # Σ_j F_ij * (B_j D_j)
        np.reciprocal(denom, out=outscale_vec, where=(denom > ϵ))
        # Idem; B_j = 1

        AO_vec = outscale_vec * target_outflow
        BD_vec = inscale_vec * target_inflow
        F_rescaled = BD_vec * f_mat * AO_vec[:, np.newaxis]

        # Current error
        model_outflow = np.asarray(F_rescaled.sum(axis=1)).flatten()
        model_inflow = np.asarray(F_rescaled.sum(axis=0)).flatten()

        # Note we exclude zero rows or zero columns
        colsum_error = np.linalg.norm(
            model_outflow[~idx_rows] / target_outflow[~idx_rows] - 1.0, np.inf)
        rowsum_error = np.linalg.norm(
            model_inflow[~idx_cols] / target_inflow[~idx_cols] - 1.0, np.inf)
        iter_error = np.linalg.norm(outscale_vec - outscale_vec_old)

        # After error, update variable
        outscale_vec_old = outscale_vec

        if colsum_error < rel_tol and rowsum_error < rel_tol:
            converged = True
        elif iter_error < ϵ:
            warnings.warn(
                f'Iterations converged ({num_iter} iters) but constraints cannot be satisfied')
            # warnings.warn(f'{colsum_error:.3f}, {rowsum_error:.3f}')
            converged = True

    if converged is False:
        raise FailedToConverge(
            f'reached maximum number of iterations ({max_iters})')

    return (F_rescaled, outscale_vec, inscale_vec) if return_vecs else F_rescaled


class FailedToConverge(Exception):
    """Raised when a iterative method failed to converge."""

    pass


class DataNotSet(Exception):
    """Raised when a method needs to access the data and it has not been set."""

    pass
