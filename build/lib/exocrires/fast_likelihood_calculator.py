"""
exocrires.fast_likelihood_calculator
====================================

Fast, mathematically identical re-implementation of
``exocrires.spectra_processor.processor_likelihood_map.likelihood_calculator`` /
``likelihood_map`` for the case where the noise covariance ``Sigma_0`` is
diagonal (which is what ``covariance_calculator`` always returns: a per-pixel
variance vector, not a full matrix).

Why this exists
---------------
The original ``likelihood_calculator`` materialises the diagonal covariance as a
dense ``(Nd x Nd)`` array with ``np.diagflat`` and inverts it with
``np.linalg.inv``.  That is ``O(Nd^3)`` work and ``O(Nd^2)`` memory *per call*.
With the CRIRES+ K2166 setting ``Nd ~ 6144``, so a single call allocates ~300 MB
and takes ~45 s; a ``(35 rows x 40 velocity grids)`` map therefore needs ~17 h
per order.  For a diagonal ``Sigma_0`` every one of those operations collapses to
an element-wise one::

    Sigma_0^-1 x        ->  x / sigma^2
    M^T Sigma_0^-1 M    ->  (M.T * 1/sigma^2) @ M          (Nc x Nc)
    chi_0^2             ->  sum(residual^2 / sigma^2)
    sum(log Sigma_0)    ->  sum(log sigma^2)

which brings the same map down to a few seconds (~250-600x measured).

The log-likelihood evaluated is the ``log_sigma_0=True`` branch of the original,
i.e. the marginalised likelihood of Ruffio et al. (2019) eq. 15 / Landman et al.
(2024) eq. 15::

    lnL = ln(prior) - 0.5 * (sum(log sigma^2) + ln det(M^T Sigma_0^-1 M))
          + ((Nd - Nc + gamma - 1) / 2) * ln(1 / chi_0^2)

BUGFIX (2026-09-22): the determinant term above was previously written with a
"-" (i.e. the formula read "... - ln det(...)"), which flips its sign relative
to both cited references and to this module's own ``log_sigma_0=False``-style
derivation. Found and confirmed against Landman et al. (2024) on 2026-09-05,
fixed here and in ``processor_likelihood_map.likelihood_calculator`` on
2026-09-22. The determinant depends on the model design matrix (hence on the
trial velocity), so this is not a constant offset in general; it was validated
on 2026-09-05 to have a negligible effect on *self-referenced* (peak-relative)
significance figures for one survey grid, but it does change raw/absolute
log-likelihood values -- see ``exocrires.spectra_retriever`` for a case
(un-normalised model amplitude) where the sign of this term matters directly.

Agreement with the original
---------------------------
Verified against ``processor_likelihood_map.likelihood_calculator`` /
``likelihood_map`` over random and realistically ill-scaled inputs
(tiny variances ~1e-9, Nc = 1, 2, 5), and over full maps with NaN masking:

    * scalar calculator .......... relative difference <= 1.1e-15
    * map, matrix_components=2 ... max |difference| 4.5e-13 (rel 1.5e-16)
    * map, single component + ``skip_velocity_grid_single_component``
                                   max |difference| 4.5e-13

These relative-agreement figures still hold after the 2026-09-22 sign fix
(both implementations were changed identically, so they still match each
other to the same precision). The specific reference number below, however,
was measured *before* that fix and is now stale as an absolute value:

End-to-end on the real H1800_10_1.51_6.6/0308 K2166 dataset (7 orders,
Nd = 6144, 35 spatial rows, 20 velocity grids), the summed planet-vs-no-planet
map agreed with the previously saved reference map to 7.3e-12 everywhere and
reproduced dlnL = 1741.67 at the planet row for v = 10 km/s -- pre-fix value,
not re-verified post-fix.

Deliberate differences from the original
----------------------------------------
1.  ``matrix_components=1`` **without** ``skip_velocity_grid_single_component``.
    The original has a variable-shadowing bug in that branch: the inner
    ``for v in tqdm(...)`` loop variable is shadowed by the list comprehension
    ``[model_matrix[v][i][...] for v in range(model_matrix.shape[0])]``, so the
    design matrix is built from *every* velocity grid at once (Nc = n_vgrids
    instead of 1) and the identical value is then written to every velocity bin.
    This module builds the intended single-component matrix for grid ``v``.
    The two therefore disagree in this branch only, and this module is correct.
    The branch is not used by the production run scripts, which always pass
    either ``matrix_components=2`` or ``matrix_components=1`` together with
    ``skip_velocity_grid_single_component=True``.

2.  ``ln det`` is obtained from the eigenvalues of the (symmetric positive
    definite) normal matrix rather than from ``np.log(np.linalg.det(...))``.
    The two agree to within float rounding, but ``det`` overflows to ``inf``
    for large ``Nc`` combined with small variances -- ``Nc=10`` at
    ``sigma^2 = 1e-12`` already gives ``det ~ 1e156``, one order of magnitude
    from silently returning ``inf``.

3.  ``log_sigma_0=False`` is not reproduced bit-for-bit.  That branch of the
    original evaluates ``np.linalg.det(Sigma_0)``, i.e. the product of ``Nd``
    variances, which underflows to exactly ``0.0`` for any realistic ``Nd`` and
    makes the result ``-inf``.  Passing ``log_sigma_0=False`` here emits a
    ``RuntimeWarning`` and uses the numerically stable sum-of-logs instead.

4.  Unfilled cells are ``NaN`` rather than ``0.0``.  The original pre-fills the
    map with zeros, so a row it never computes is indistinguishable from a
    genuine ``lnL = 0``.  ``np.nansum`` over the result -- which the production
    run scripts already use -- treats the two identically.

5.  Failures are localised and detected.  A rank-deficient model matrix leaves
    ``M^T Sigma_0^-1 M`` numerically singular; the original then returns a
    meaningless value built on a ``det`` that is pure rounding noise.  Here the
    condition number is checked explicitly, the affected cell becomes ``NaN``,
    and the rest of the map still completes.

6.  Inputs are validated (variance positivity, shape agreement, and
    ``n_vgrids`` vs ``model_matrix.shape[0]``).  The original silently leaves
    zeros -- indistinguishable from a genuine ``lnL = 0`` -- in the output rows
    it never fills.

Preserved quirk
---------------
``skip_velocity_grid_single_component`` is tested with ``is not None``, exactly
as in the original.  Passing ``False`` therefore *enables* the skip.  Pass
``None`` (or omit it) to disable.

Typical use
-----------
>>> from exocrires.spectra_processor import processor_likelihood_map
>>> P = processor_likelihood_map()
>>> Map = P.fast_likelihood_map(observation_matrix=obs, model_matrix=mm,
...                             n_vgrids=len(grids), sigma_matrix=var[order],
...                             prior=1, matrix_components=2)
>>> Map_noplanet = P.fast_likelihood_map(observation_matrix=obs, model_matrix=mm0,
...                                      n_vgrids=len(grids), sigma_matrix=var[order],
...                                      prior=1, matrix_components=1,
...                                      skip_velocity_grid_single_component=True)
>>> delta_lnL = Map - Map_noplanet
"""

import warnings

import numpy as np
from scipy.linalg import solve, LinAlgError

__all__ = ['fast_likelihood_calculator', 'fast_likelihood_map']

#: Reject the fit when cond(M^T Sigma_0^-1 M) exceeds this, i.e. when the
#: normal matrix has lost all significant digits to rounding.
_MAX_CONDITION = 1.0 / np.finfo(float).eps


def _log_likelihood_core(d, M, inv_var, log_det_sigma, prior_psi, gamma):
    """Evaluate lnL from pre-computed per-row quantities.

    Split out so that ``fast_likelihood_map`` can hoist ``1/sigma^2`` and
    ``sum(log sigma^2)`` -- which depend on the spatial row only, not on the
    velocity grid -- out of the velocity loop.

    Parameters
    ----------
    d : (Nd,) ndarray            masked data vector
    M : (Nd, Nc) ndarray         masked design matrix
    inv_var : (Nd,) ndarray      1 / sigma^2
    log_det_sigma : float        sum(log sigma^2)
    prior_psi : float            prior P(psi)
    gamma : float                noise-scaling hyper-parameter

    Returns
    -------
    float : the log-likelihood, or NaN if the normal matrix is numerically
            singular or chi_0^2 is non-positive.
    """
    MT_Sigma0_inv_M = (M.T * inv_var) @ M
    MT_Sigma0_inv_d = (d * inv_var) @ M

    # The normal matrix is symmetric positive definite whenever M has full
    # column rank, so one symmetric eigendecomposition yields both the
    # conditioning check and ln det -- cheaper than cond() + slogdet(), and the
    # check is necessary: a rank-deficient design matrix (e.g. a planet
    # template that has gone degenerate with the stellar one) does *not* make
    # scipy's solve raise. It emits a LinAlgWarning and returns garbage built
    # on a determinant that is pure rounding noise.
    eigenvalues = np.linalg.eigvalsh(MT_Sigma0_inv_M)
    if (not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0
            or eigenvalues[-1] / eigenvalues[0] > _MAX_CONDITION):
        return np.nan
    log_determinant = np.sum(np.log(eigenvalues))

    try:
        c_hat = solve(MT_Sigma0_inv_M, MT_Sigma0_inv_d)
    except (LinAlgError, ValueError):
        return np.nan

    residual = d - M @ c_hat
    chi_0_squared = np.sum(residual ** 2 * inv_var)
    if not np.isfinite(chi_0_squared) or chi_0_squared <= 0:
        return np.nan

    Nd = d.shape[0]
    Nc = M.shape[1]

    # BUGFIX (2026-09-22): was "- log_determinant" (i.e. "+0.5*log_determinant"
    # after distributing -0.5), which flipped the sign of the |M^T Sigma_0^-1 M|
    # term relative to Ruffio et al. (2019) eq. 15 / Landman et al. (2024) eq. 15 --
    # both |Sigma_0| and |M^T Sigma_0^-1 M| belong in the denominator of L, i.e.
    # both subtracted here. This mirrors the same fix applied to
    # processor_likelihood_map.likelihood_calculator's log_sigma_0=True branch in
    # spectra_processor.py; the two must agree since this module is a fast
    # drop-in replacement for that one. See that file for the fix history.
    return (np.log(prior_psi)
            - 0.5 * (log_det_sigma + log_determinant)
            + ((Nd - Nc + gamma - 1) / 2) * np.log(1 / chi_0_squared))


def fast_likelihood_calculator(d=None, M=None, Sigma_0_flat=None,
                               log_sigma_0=True, prior_psi=None, gamma=2):
    """Diagonal-covariance equivalent of ``likelihood_calculator``.

    Parameters
    ----------
    d : (Nd,) ndarray
        Data vector.
    M : (Nd, Nc) ndarray
        Model / design matrix.  The last column is conventionally the planet
        component.
    Sigma_0_flat : (Nd,) ndarray
        Per-pixel *variance* (the diagonal of ``Sigma_0``), as returned by
        ``processor_likelihood_map.covariance_calculator``.  Must be strictly
        positive and finite.
    log_sigma_0 : bool, optional
        Accepted for signature compatibility with ``likelihood_calculator``.
        ``False`` warns and falls back to the stable sum-of-logs; see the module
        docstring, point 3.
    prior_psi : float
        Prior P(psi).  Must be positive.
    gamma : float, optional
        Noise-scaling hyper-parameter (default 2).  Note that ``likelihood_map``
        passes ``gamma=matrix_components``; ``fast_likelihood_map`` reproduces
        that.

    Returns
    -------
    float : log-likelihood.
    """
    d = np.asarray(d, dtype=float)
    M = np.asarray(M, dtype=float)
    Sigma_0_flat = np.asarray(Sigma_0_flat, dtype=float)

    if d.ndim != 1:
        raise ValueError(f'd must be 1-D, got shape {d.shape}')
    if M.ndim != 2:
        raise ValueError(f'M must be 2-D (Nd, Nc), got shape {M.shape}')
    if M.shape[0] != d.shape[0]:
        raise ValueError(f'M has {M.shape[0]} rows but d has {d.shape[0]} '
                         'elements')
    if Sigma_0_flat.shape != d.shape:
        raise ValueError(f'Sigma_0_flat has shape {Sigma_0_flat.shape} but d '
                         f'has shape {d.shape}')
    if prior_psi is None or not np.isfinite(prior_psi) or prior_psi <= 0:
        raise ValueError(f'prior_psi must be a positive finite number, got '
                         f'{prior_psi!r}')
    if not np.all(np.isfinite(Sigma_0_flat)) or np.any(Sigma_0_flat <= 0):
        raise ValueError('Sigma_0_flat must be strictly positive and finite; '
                         'mask non-finite / zero-variance pixels out first')

    if not log_sigma_0:
        warnings.warn(
            'log_sigma_0=False reproduces a branch of the original '
            'likelihood_calculator that evaluates det(Sigma_0) directly; that '
            'product underflows to 0 for realistic Nd and yields -inf. Using '
            'the stable sum(log sigma^2) instead.',
            RuntimeWarning, stacklevel=2)

    inv_var = 1.0 / Sigma_0_flat
    log_det_sigma = np.sum(np.log(Sigma_0_flat))

    return _log_likelihood_core(d, M, inv_var, log_det_sigma, prior_psi, gamma)


def fast_likelihood_map(observation_matrix, model_matrix, n_vgrids,
                        sigma_matrix, prior, matrix_components=2,
                        log_sigma=True,
                        skip_velocity_grid_single_component=None,
                        mask_nonfinite_model=False):
    """Diagonal-covariance equivalent of ``likelihood_map``.

    Parameters
    ----------
    observation_matrix : (n_rows, Nd) ndarray
        Observed data, one row per spatial pixel.  Non-finite entries are
        masked out.
    model_matrix : ndarray
        ``(n_v, matrix_components, n_rows, Nd)`` when ``matrix_components >= 2``,
        or ``(n_v, n_rows, Nd)`` when ``matrix_components == 1``.  The velocity
        grid is always the first axis.
    n_vgrids : int
        Number of velocity grids; must equal ``model_matrix.shape[0]`` unless
        ``skip_velocity_grid_single_component`` is set, in which case the single
        value computed at grid 0 is broadcast across all ``n_vgrids`` columns.
    sigma_matrix : (n_rows, Nd) ndarray
        Per-pixel *variance*, from ``covariance_calculator``.  Non-finite
        entries are masked out; the surviving entries must be strictly positive.
    prior : float
        Prior P(psi).
    matrix_components : int, optional
        Number of model components (default 2: stellar + planet).  Also passed
        through as ``gamma``, matching the original.
    log_sigma : bool, optional
        Signature compatibility; see ``fast_likelihood_calculator``.
    skip_velocity_grid_single_component : optional
        For a stellar-only model that does not depend on velocity: compute once
        at grid 0 and broadcast.  Tested with ``is not None`` to match the
        original, so ``False`` *enables* it -- pass ``None`` to disable.
    mask_nonfinite_model : bool, optional
        If True, also drop pixels where any model component is non-finite.
        Default False, matching the original, which propagates model NaNs into
        the result.  The production run scripts ``np.nan_to_num`` the model
        beforehand instead.

    Returns
    -------
    (n_rows, n_vgrids) ndarray of log-likelihoods.
    """
    observation_matrix = np.asarray(observation_matrix)
    sigma_matrix = np.asarray(sigma_matrix)

    if not isinstance(matrix_components, (int, np.integer)) or matrix_components < 1:
        raise ValueError(
            'matrix_components is the number of components in your model and '
            f'must be an integer >= 1 in EXOCRIRES; got {matrix_components!r}. '
            'The last component should be the planet model.')

    n_rows = observation_matrix.shape[0]
    if sigma_matrix.shape[0] != n_rows:
        raise ValueError(f'sigma_matrix has {sigma_matrix.shape[0]} rows but '
                         f'observation_matrix has {n_rows}')

    single_fixed = skip_velocity_grid_single_component is not None
    if single_fixed and matrix_components != 1:
        raise ValueError('skip_velocity_grid_single_component only applies to '
                         f'matrix_components=1, got {matrix_components}')
    if not single_fixed and model_matrix.shape[0] != n_vgrids:
        raise ValueError(
            f'model_matrix.shape[0] is {model_matrix.shape[0]} but n_vgrids is '
            f'{n_vgrids}; they must match, otherwise the output would be '
            'silently left as zeros (indistinguishable from lnL = 0) or would '
            'overrun.')

    log_likelihood_map = np.full((n_rows, n_vgrids), np.nan)

    for i in range(n_rows):
        obs_row = np.asarray(observation_matrix[i], dtype=float).ravel()
        variance_flat = np.asarray(sigma_matrix[i], dtype=float).ravel()
        if variance_flat.shape != obs_row.shape:
            raise ValueError(
                f'row {i}: sigma_matrix row flattens to {variance_flat.shape} '
                f'but observation_matrix row flattens to {obs_row.shape}')

        mask = np.isfinite(obs_row) & np.isfinite(variance_flat)
        if mask_nonfinite_model:
            if matrix_components == 1:
                mask &= np.isfinite(
                    np.asarray(model_matrix[:, i], dtype=float)).all(axis=0)
            else:
                mask &= np.isfinite(
                    np.asarray(model_matrix[:, :matrix_components, i],
                               dtype=float)).all(axis=(0, 1))

        d = obs_row[mask]
        var_m = variance_flat[mask]

        if d.size <= matrix_components:
            # under-determined: fewer surviving pixels than free coefficients
            continue
        if np.any(var_m <= 0):
            raise ValueError(f'row {i}: sigma_matrix contains non-positive '
                             'variances among the finite pixels')

        # sigma depends on the spatial row only, so hoist these out of the
        # velocity loop -- this is where most of the remaining time would go.
        inv_var = 1.0 / var_m
        log_det_sigma = np.sum(np.log(var_m))

        if single_fixed:
            # stellar-only model, fixed at velocity-grid index 0
            value = _log_likelihood_core(
                d, np.asarray(model_matrix[0][i], dtype=float).ravel()[mask][:, None],
                inv_var, log_det_sigma, prior, matrix_components)
            log_likelihood_map[i] = value
        else:
            for v in range(n_vgrids):
                if matrix_components == 1:
                    Mmat = np.asarray(
                        model_matrix[v][i], dtype=float).ravel()[mask][:, None]
                else:
                    Mmat = np.stack(
                        [np.asarray(model_matrix[v][comp, i],
                                    dtype=float).ravel()[mask]
                         for comp in range(matrix_components)], axis=1)
                log_likelihood_map[i, v] = _log_likelihood_core(
                    d, Mmat, inv_var, log_det_sigma, prior, matrix_components)

    return log_likelihood_map
