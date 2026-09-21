"""
exocrires.spectra_retriever
===========================
Atmospheric retrieval for EXOCRIRES, with petitRADTRANS-3 as the forward model.

What this changes relative to the previous EXOCRIRES study
----------------------------------------------------------
The likelihood machinery is unchanged.  The published EXOCRIRES detectability
runs (``EXOCRIRES_likelihood_fast.ipynb``) took a *fixed grid* of pre-computed
theoretical planet spectra (``wispit_1b_spectra.npz`` and the ``upload_*.dat``
templates derived from it), pushed them through
:mod:`exocrires.input_spectra_preparation`, separated them into star/planet
components with :class:`~exocrires.spectra_processor.Spectra_fitting`, and
handed the resulting model matrix to
:meth:`~exocrires.spectra_processor.processor_likelihood_map.fast_likelihood_map`.

This module keeps every one of those steps and replaces only the *first* one:
the planet spectrum is now generated on the fly by petitRADTRANS-3 from a point
in parameter space, so the whole chain becomes a function of the free
parameters and can be sampled with MultiNest::

    theta  (prior cube)
      |
      |  pRT_ForwardModel                       <- NEW (was: fixed .npz / .dat)
      v
    P-T profile (Piette+2020) + chemistry (free or equilibrium)
      |
      |  Radtrans.calculate_flux  (lbl)
      v
    rest-frame emission spectrum
      |
      |  input_spectra_preparation.Spectra_Preparator   <- SAME module as before
      |    limb darkening -> rotational broadening -> instrument LSF -> RV shift
      v
    detector-sampled planet template
      |
      |  Spectra_fitting.fit_planet / fit_star           <- SAME "spectra separation"
      v
    model matrix  (n_v, [star, planet], n_ord, 1, n_pix)
      |
      |  processor_likelihood_map.fast_likelihood_map    <- SAME likelihood
      v
    delta ln L  =  lnL(star + planet)  -  lnL(star only)
      |
      |  pymultinest
      v
    posterior + ln Z

The reference implementation for the pRT-3 half is
``Recipe_DH_Tau_B/tasting_guidebook/Guidebook_GAStronomy_Piette_v4.2.5.py``
(the DH Tau B retrieval).  The P-T parameterisation, the free/equilibrium
chemistry, the species list and the MultiNest wiring below are ported from it;
what is *not* ported is its data model (CRIRES+ 1-D extracted spectra with a GP
covariance), because EXOCRIRES works on 2-D ABBA stacks where the star and the
planet share every pixel and must be separated first.

How the likelihood is assembled across orders - and how that differs from the guidebook
--------------------------------------------------------------------------------------
This module sums a **separate marginal likelihood per spectral order**::

    ln L = sum_over_orders  fast_likelihood_map(obs[order], model[order], ...)

which is exactly what ``EXOCRIRES_likelihood_fast.ipynb`` does
(``delta_sum = np.nansum(delta_lnL, axis=0)`` over ``order_inputs(order, ...)``).
Each order therefore gets **its own** marginalised amplitude for each model
component **and its own noise scale**; orders are treated as statistically
independent blocks.  Within one order all three detectors share a single design
matrix, so there is one planet amplitude per order, not per detector.

The Piette guidebook (``LogLikelihood.__call__``) does something different: it
concatenates every order and detector into one vector and evaluates the Ruffio
formula **once**, globally --

    ln L = -0.5*(N_d - N_phi)*ln(2 pi) + loggamma(0.5*(N_d - N_phi + alpha - 1))
           - 0.5*(logdet_cov + logdet_MT_C_M + (N_d - N_phi + alpha - 1)*ln(chi2_0))

with a single global ``chi2_0``, a single ``logdet_MT_C_M``, one global noise
scale ``s2``, and ``N_phi = n_orders * n_dets`` degrees of freedom subtracted to
account for the per-chip scalings that ``flux_scaling_rolling`` applied to the
model beforehand.  Its per-chip freedom is in the *pre-scaling*, not in the
marginalisation.

Neither is "the" right answer - they are different models:

* **Per-order sum (here, and the EXOCRIRES study).**  Robust to order-to-order
  systematics: a badly calibrated order cannot drag the others, because it
  carries its own amplitude and its own noise scale.  The cost is that
  information about the *relative* flux between orders - a real constraint on
  the spectral slope, hence on temperature - is discarded.
* **Global evaluation (guidebook).**  Keeps the inter-order flux ratios, so the
  continuum shape informs the fit, at the price of being sensitive to any
  chip-level calibration error.

Because the brief for this module was to keep the EXOCRIRES likelihood
untouched and change only the source of the model spectrum, the per-order sum is
what is implemented, and it is verified to reproduce the notebook exactly.  If
you ever want the guidebook's global assembly instead, it is *not* a matter of
changing the loop: it needs a single ``fast_likelihood_map`` call on
order-concatenated ``obs``/``model``/``variance`` arrays, which changes the
statistical model and is not equivalent to summing.

Two consequences of the EXOCRIRES likelihood worth understanding
----------------------------------------------------------------
1. **Amplitude is marginalised - but only with the model normalisation on.**
   ``fast_likelihood_map`` evaluates the Ruffio et al. (2019) eq. 15 marginal
   likelihood, which integrates over the linear coefficients of the design
   matrix (and over the noise scale).  That marginalisation uses an *improper
   flat* prior, so the result is **not** scale-free: ln L keeps a
   ``-0.5 * ln det(M^T Sigma^-1 M)`` term, and scaling the planet column by k
   adds exactly ``-ln(k)`` (a genuine Occam penalty against larger amplitude).
   With ``normalize_model='per_order_median'`` (the default) every template is
   placed on the same per-order scale, the term becomes a constant, and the
   overall flux level really is unconstrained - so planet radius, distance and
   flux calibration must not be sampled, and the guidebook's ``log_R`` /
   absolute-flux machinery is deliberately absent. ``log_g`` is retained
   because it changes the *shape*.
   With ``normalize_model=None`` the ``-ln(k)`` term survives and now
   penalises brighter (hotter) models; see
   :meth:`SpectraRetriever._normalize_template` for the measured size of that
   effect. NOTE (2026-09-22): before this date, ``fast_likelihood_map`` /
   ``likelihood_calculator`` had a sign bug that made this term
   ``+0.5 * ln det(...)`` / ``+ln(k)`` instead, i.e. it *rewarded* brighter
   models under ``normalize_model=None``. The bug was fixed in
   ``fast_likelihood_calculator.py`` / ``spectra_processor.py`` on 2026-09-22
   (found and confirmed against Landman et al. 2024 eq. 15 on 2026-09-05).
   Point 1's conclusion for the *default* ``normalize_model='per_order_median'``
   path is unaffected either way, since the term is a sampled-parameter-
   independent constant in both the old and new sign -- only the un-normalised
   (``normalize_model=None``) direction of the bias flips.

2. **Sampling on delta ln L is equivalent to sampling on ln L.**  The star-only
   term uses model-matrix component 0, which does not depend on any sampled
   parameter, so it is a constant offset.  Subtracting it leaves the posterior
   identical while turning the evidence into a *log Bayes factor* for
   "planet present" vs "star only" - which is the quantity the EXOCRIRES
   detectability work reports.  Set ``objective='lnL'`` to sample the raw
   likelihood instead.

Validation
----------
``build_model_matrix`` was checked against a literal transcription of the
notebook's CELL 5 inner loop on the validation dataset (WISPIT 1b): the star and
planet components agree **exactly** (``np.array_equal``), so hoisting
``fit_star`` out of the velocity loop and restricting the order set changed no
number.  Feeding the original injected template (``wispit_1b_spectra.npz``)
through this module's own path recovers the planet at +21 km/s on a 1 km/s grid
against an injected 19 km/s.

Note that the *saved* reference products in the day folder
(``model_matrix_combine_*.npy``, ``delta_log_likelihood_noplanet_fast.npy``)
were built on 2026-06-10 from an **older** ``stack_sub_badpix.npy`` than the one
now on disk (2026-06-14): their stellar component differs from what the current
stack produces by a median factor of ~0.37.  They are therefore not a
bit-reproducible target, and the ~14% amplitude difference against them is
expected.  Rebuild them if an exact comparison is ever needed.

Testing mode
------------
``testing=True`` (the default) is the configuration this module was developed
and validated in: a single spectral order, coarse line-by-line opacity
sampling, and a small live-point count, so one likelihood call costs seconds
rather than minutes and a full sampling run finishes in minutes.  It exercises
the identical code path as a production run - only the wavelength span, the
opacity sampling and the sampler settings differ.  Switch to
``testing=False`` (and raise ``n_live_points``) for science runs.

Environment
-----------
Needs petitRADTRANS-3 and exocrires importable together.  On this machine that
is the ``prt3`` conda environment (Python 3.13); ``exocrires`` is symlinked into
it.  exocrires itself targets numpy 1.x, so a ``np.trapz`` compatibility alias
is installed at import time (numpy 2.0 renamed it ``np.trapezoid``).

Scope: this module is system-agnostic
------------------------------------
Like the other EXOCRIRES stages, this is a *method* module: it knows how to turn
a 2-D ABBA stack plus a prior into a posterior, and nothing about which object
is being observed.  It therefore contains **no target table, no default prior
ranges and no injected-truth values**.  Everything specific to a system lives in
that project's driver:

    exocrires.spectra_retriever          run_spectra_retriever.py
    ---------------------------          -----------------------------------
    SpectraRetriever, pRT_ForwardModel   which folders/days/stacks exist
    Parameters, pt_profile               separations, v_sys, vsini
    make_free_params (encoder)           the actual prior dict + ranges
    build_retriever                      injected truth, log g, truth .npz

Consequently priors reach this module only as a dict the driver writes out -
``{name: (prior_spec, label)}``, encoded by :func:`make_free_params` - and
:class:`SpectraRetriever` requires ``prep_T``/``prep_G``/``prep_Z`` rather than
assuming a template.  To analyse a new system, write a new ``run_*.py`` with its
own target table and priors and call :func:`build_retriever` - no edit to this
file should be needed.  ``run_spectra_retriever.py`` (WISPIT 1 b/1 c) is the
worked example.

References
----------
Ruffio, J.-B. et al. (2019). AJ 158, 200.  Marginal likelihood, eq. 15.
Piette, A. & Madhusudhan, N. (2020). MNRAS 497, 5136.  P-T parameterisation.
Xuan, J. et al. (2024). ApJ 970, 71.  P-T node placement.
Molliere, P. et al. (2019). A&A 627, A67.  petitRADTRANS.
"""

from __future__ import annotations

import os
import time
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

# ---------------------------------------------------------------------------
# numpy 2 compatibility.
#
# exocrires targets numpy 1.x and calls np.trapz in spectra_processor
# (multiplicative_factor_fit, correct_alpha_star_planet_pixel) and analysis.
# numpy 2.0 renamed it to np.trapezoid and removed the old name.  petitRADTRANS
# 3.2.0 requires numpy 2.x, so the two can only share a process if the alias is
# restored.  Same function, same signature - this is a rename, not a shim with
# behaviour of its own.
# ---------------------------------------------------------------------------
if not hasattr(np, 'trapz'):            # pragma: no cover - numpy-version dependent
    np.trapz = np.trapezoid

from exocrires import info
from exocrires import input_spectra_preparation
from exocrires.spectra_processor import Spectra_fitting, processor_likelihood_map


# ============================================================
# 1. CONSTANTS
# ============================================================

# Explicit pRT3 line-list names.  Without these, pRT prompts interactively to
# disambiguate species that have more than one line list on disk, which
# deadlocks a non-interactive or MPI run.  Ported from the guidebook.
PRT3_SPECIES_OVERRIDES = {
    '12C-16O':      '12C-16O__HITEMP',
    '12C-1H4':      '12C-1H4__MM',
    'CO2_main_iso': '12C-16O2__HITEMP',
    'FeH_main_iso': '56Fe-1H',
    'HCN':          '1H-12C-14N',
    'H2S':          '1H2-32S',
    'NH3':          '14N-1H3',
}

# Full equilibrium-chemistry species set (guidebook v4.2.5).  Every one of these
# is present in the local strw_backup line-list bundle.
EQ_SPECIES_PRT3 = [
    '1H2-16O',            # H2O  - POKAZATEL
    '12C-16O__HITEMP',    # 12CO - HITEMP
    '13C-16O',            # 13CO - HITEMP
    '12C-1H4__MM',        # CH4  - MM/ExoMol
    '14N-1H3',            # NH3  - CoYuTe
    '1H2-32S',            # H2S  - AYT2
    '1H-12C-14N',         # HCN  - Harris
    '12C-16O2__HITEMP',   # CO2  - HITEMP
    '56Fe-1H',            # FeH  - MoLLIST
    '1H-19F',             # HF   - Coxon-Hajig
    '23Na',               # Na   - native equilibrium-table species
    '40Ca',               # Ca   - solar-scaled by [Z/H]
]
EQ_LABEL_NAMES = ['H2O', '12CO', '13CO', 'CH4', 'NH3', 'H2S', 'HCN', 'CO2',
                  'FeH', 'HF', 'Na', 'Ca']

# The three molecules the EXOCRIRES detectability study reports individually.
# Minimal K-band set: what actually shapes a 1900-2500 nm L-/T-type spectrum.
KBAND_SPECIES_MINIMAL = ['1H2-16O', '12C-16O__HITEMP', '12C-1H4__MM']

CO_SOLAR       = 0.5495    # (C/O)_solar, Lodders 2020
LOG_CH_SOLAR   = -3.54     # log10(n_C/n_H)_solar, Asplund+2021
LOG_F_H_SOLAR  = -7.6      # log10(n_F/n_H)_solar, Maiorca+2014
LOG_CA_H_SOLAR = -5.73     # log10(n_Ca/n_H)_solar, Lodders 2020 (A(Ca)=6.27)

# ---- P-T node sets ----------------------------------------------------------
#
# A node set is (log10 pressure nodes deep -> shallow, index of the anchor).
# The profile is built from the anchor outwards by non-negative increments dT_i,
# so it is monotonic by construction, then interpolated in log P.
#
# 'piette8'  - Piette+2020 / Xuan+2024, the original 8-node set.  Anchor at
#              0.2 bar.  Kept as the DEFAULT so earlier runs reproduce exactly.
#
# 'l24_4knot' - 4 knots after Landman+2024 (A&A 682, A48), who use 1 mbar /
#              0.1 / 1 / 10 bar with a third-order spline for beta Pic b with
#              CRIRES+.  Two deliberate differences here:
#                * 5 bar replaces their 10 bar, because that is where this
#                  parameterisation stops and the measured K-band emission
#                  contribution function carries exactly 0.000 of the flux
#                  below it (measured 2026-09-21 on the WISPIT 1b stack).
#                * the anchor is the 1 bar knot, not a shallow one.  The
#                  contribution function peaks at 0.79 bar, so this anchors the
#                  profile at the best-measured pressure instead of at 0.2 bar,
#                  which sits below the 5th percentile of the emission and made
#                  every increment inherit the anchor's uncertainty.
#              Four fewer free parameters than 'piette8', and with only one knot
#              between 0.1 and 5 bar the sampler cannot build the unphysical
#              1 bar kink the 8-node set produced (nabla = 0.14 then 0.59
#              across adjacent layers, against a smooth truth at 0.21-0.26).
PT_NODE_SETS = {
    'piette8':   (np.array([0.7, 0.0, -0.3, -0.7, -1.0, -1.5, -3.0, -5.0]), 3),
    'l24_4knot': (np.array([0.7, 0.0, -1.0, -3.0]), 1),
}

# Back-compatible module-level names: the default set.
LOG_P_NODES = PT_NODE_SETS['piette8'][0]
ANCHOR_IDX  = PT_NODE_SETS['piette8'][1]


def pt_nodes(node_set='piette8'):
    """(log10 P nodes deep->shallow, anchor index) for a named node set."""
    if node_set not in PT_NODE_SETS:
        raise ValueError('unknown P-T node set %r; known: %s'
                         % (node_set, sorted(PT_NODE_SETS)))
    return PT_NODE_SETS[node_set]


def n_dT(node_set='piette8'):
    """How many dT_i parameters a node set needs (one fewer than its nodes)."""
    return len(PT_NODE_SETS[node_set][0]) - 1


def pt_profile(params, pressure, node_set='piette8'):
    """Node-based P-T profile for a parameter dict.

    ``T_anchor`` is the temperature at the anchor node; ``dT_i`` are the
    non-negative temperature increments between adjacent nodes, numbered deep
    -> shallow, so the profile decreases monotonically outwards.  Interpolation
    is PCHIP in log P: a cubic Hermite spline that, unlike an unconstrained
    B-spline, cannot overshoot between knots, so monotonicity at the nodes
    guarantees monotonicity everywhere.  (Landman+2024 get the same property
    from an explicit no-inversion prior on a B-spline.)  Outside the node range
    the profile is isothermal at the nearest node.

    Module-level twin of :meth:`pRT_ForwardModel.make_pt`, so posterior samples
    can be turned into temperature profiles without constructing a Radtrans
    (which would load opacities per sample).  ``make_pt`` delegates here, so the
    two can never drift apart.
    """
    log_p_nodes, anchor_idx = pt_nodes(node_set)
    n = len(log_p_nodes)

    pressure = np.asarray(pressure, dtype=float)
    T_anchor = float(params['T_anchor'])
    dT = np.array([float(params['dT_%d' % (i + 1)]) for i in range(n - 1)])

    T_nodes = np.empty(n)
    T_nodes[anchor_idx] = T_anchor
    for j in range(anchor_idx - 1, -1, -1):
        T_nodes[j] = T_nodes[j + 1] + dT[j]
    for j in range(anchor_idx + 1, n):
        T_nodes[j] = T_nodes[j - 1] - dT[j - 1]

    log_P_asc = log_p_nodes[::-1]
    T_asc = T_nodes[::-1]
    temperature = PchipInterpolator(log_P_asc, T_asc)(np.log10(pressure))
    log_P_atm = np.log10(pressure)
    temperature = np.where(log_P_atm < log_P_asc[0], T_asc[0], temperature)
    temperature = np.where(log_P_atm > log_P_asc[-1], T_asc[-1], temperature)
    return np.clip(temperature, 1.0, 30000.0)


_SPECIES_INFO_CACHE = None


def _get_species_info():
    """species_info.csv as a DataFrame, read once per process.

    The file ships next to this module (copied from the Recipe_DH_Tau_B
    guidebook).  It maps friendly names to pRT names, molecular masses and
    C/O/H/N/S atom counts, and is what makes free chemistry self-describing.
    """
    global _SPECIES_INFO_CACHE
    if _SPECIES_INFO_CACHE is None:
        csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'species_info.csv')
        if not os.path.exists(csv):
            raise FileNotFoundError(
                'species_info.csv not found next to spectra_retriever.py (%s). '
                'Copy it from Recipe_DH_Tau_B/tasting_guidebook/.' % csv)
        _SPECIES_INFO_CACHE = pd.read_csv(csv, index_col=0)
    return _SPECIES_INFO_CACHE


# ============================================================
# 2. PRIORS
# ============================================================

class Parameters:
    """Prior specification and MultiNest prior transform.

    Ported unchanged from the guidebook so that a prior dict written for one
    can be used with the other.

    Parameters
    ----------
    free_params : dict
        ``{key: (prior_spec, mathtext_label)}`` where ``prior_spec`` is either
        ``[lo, hi]`` for a uniform prior or
        ``{'type': 'gaussian', 'mu': mu, 'sigma': sigma}`` for a Gaussian one.
    constant_params : dict
        Fixed values merged into ``self.params`` and visible to the forward
        model exactly like sampled ones.

    Notes
    -----
    The Piette P-T parameterisation needs ``dT_i >= 0`` to stay monotonic; that
    is enforced purely by the prior's lower bound, so ``__call__`` needs no
    ordering logic.
    """

    def __init__(self, free_params, constant_params):
        # Validate/normalise here too, so a hand-written dict gets the same
        # checks whether or not the caller went through make_free_params().
        free_params = make_free_params(free_params)
        self.params = {}
        self.param_priors = {}
        self.param_mathtext = {}
        for key, (prior, mathtext) in free_params.items():
            self.param_priors[key] = prior
            self.param_mathtext[key] = mathtext

        self.param_keys = np.array(list(self.param_priors.keys()))
        self.n_params = len(self.param_keys)
        self.ndim = self.n_params
        self.free_params = free_params
        self.constant_params = constant_params
        self.params.update(constant_params)

    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x * (bounds[1] - bounds[0]) + bounds[0]

    @staticmethod
    def gaussian_prior(mu, sigma):
        return lambda x: norm.ppf(x, loc=mu, scale=sigma)

    def __call__(self, cube, ndim=None, nparams=None):
        """Unit hypercube -> physical parameter space, in place."""
        if ndim is None and nparams is None:
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        for i, key in enumerate(self.param_keys):
            prior = self.param_priors[key]
            if isinstance(prior, dict) and prior.get('type') == 'gaussian':
                cube[i] = self.gaussian_prior(prior['mu'], prior['sigma'])(cube[i])
            else:
                cube[i] = self.uniform_prior(prior)(cube[i])
            self.params[key] = cube[i]

        return self.cube_copy


# ============================================================
# 3. FORWARD MODEL
# ============================================================

class pRT_ForwardModel:
    """petitRADTRANS-3 emission spectrum for one point in parameter space.

    This is the guidebook's ``pRT_spectrum`` reduced to what EXOCRIRES needs:
    P-T profile, chemistry, radiative transfer.  Everything downstream of the
    rest-frame spectrum (broadening, RV, detector sampling) is deliberately
    *not* here - that belongs to :mod:`exocrires.input_spectra_preparation`,
    which :class:`SpectraRetriever` drives, so that the model spectrum and the
    original theoretical templates travel through byte-identical code.

    Absolute-flux scaling is also absent, on purpose: see this module's
    docstring on amplitude marginalisation.

    Parameters
    ----------
    radtrans : petitRADTRANS.radtrans.Radtrans
        Pre-built and reused across likelihood calls (construction loads
        opacities and is far more expensive than one evaluation).
    params : dict
        Physical parameter values, as produced by :class:`Parameters`.
    species : list of str
        pRT line-species names, matching those the ``radtrans`` was built with.
    pressure : ndarray
        Pressure grid in bar, ascending (shallowest first), same grid the
        ``radtrans`` was built with.
    eq_chem : PreCalculatedEquilibriumChemistryTable, optional
        Required only for ``chemistry='equilibrium'``.
    """

    def __init__(self, radtrans, params, species, pressure, eq_chem=None,
                 pt_node_set='piette8'):
        self.radtrans = radtrans
        self.params = params
        self.species = list(species)
        self.pt_node_set = pt_node_set
        self.pressure = np.asarray(pressure, dtype=float)
        self.n_atm_layers = self.pressure.size
        self.eq_chem = eq_chem

        self.temperature = self.make_pt()

        if 'log_g' in params:
            self.gravity = 10 ** float(params['log_g'])
        else:
            raise KeyError("forward model needs 'log_g' (surface gravity, cgs). "
                           "Absolute-flux log_M/log_R scaling is not supported "
                           "here because the likelihood marginalises amplitude.")

        chem = params.get('chemistry', 'free')
        if chem == 'free':
            self.mass_fractions, self.CO, self.C_H = \
                self.free_chemistry(self.species, params)
        elif chem == 'equilibrium':
            if self.eq_chem is None:
                raise ValueError("chemistry='equilibrium' needs eq_chem")
            self.mass_fractions, self.CO, self.C_H = \
                self.equilibrium_chemistry(params)
        else:
            raise ValueError("chemistry must be 'free' or 'equilibrium', got %r" % chem)

        self.MMW = self.mass_fractions['MMW']
        self._prt_wl = None
        self._prt_flux = None

    # ---- P-T profile ----

    def make_pt(self):
        """Node-based P-T profile on the module's pressure grid.

        ``T_anchor`` fixes the temperature at the anchor node; the non-negative
        increments ``dT_i`` build the remaining nodes outward from it, and a
        monotonic PCHIP spline maps them onto the layer grid.  Non-negativity -
        hence monotonicity - comes from the priors, not from code here.  Which
        nodes, and which of them is the anchor, is set by ``pt_node_set``
        (see :data:`PT_NODE_SETS`).
        """
        return pt_profile(self.params, self.pressure, self.pt_node_set)

    # ---- chemistry ----

    def read_species_info(self, species, info_key):
        df = _get_species_info()
        if info_key == 'COH':
            return list(df.loc[species, ['C', 'O', 'H']])
        return df.loc[species, info_key]

    def free_chemistry(self, line_species, params):
        """Vertically constant log-VMR parameters -> pRT mass fractions.

        Also returns C/O and [C/H] derived from the sampled abundances, which
        are useful derived quantities to carry into the posterior.
        """
        species_info = _get_species_info()
        line_species = set(line_species)
        VMR_He = 0.15
        VMR_wo_H2 = VMR_He
        mass_frac = {}
        C = O = H = 0.0

        for sp in species_info.index:
            prt_name = self.read_species_info(sp, 'pRT_name')
            prt_name = PRT3_SPECIES_OVERRIDES.get(prt_name, prt_name)
            mass = self.read_species_info(sp, 'mass')
            COH = self.read_species_info(sp, 'COH')

            if sp in ('H2', 'He'):
                continue
            if prt_name in line_species:
                key = 'log_%s' % sp
                if key not in params:
                    raise KeyError('free chemistry needs %r for species %s' % (key, prt_name))
                VMR_i = 10 ** float(params[key]) * np.ones(self.n_atm_layers)
                mass_frac[prt_name] = mass * VMR_i
                VMR_wo_H2 += VMR_i
                C += COH[0] * VMR_i
                O += COH[1] * VMR_i
                H += COH[2] * VMR_i

        mass_frac['He'] = self.read_species_info('He', 'mass') * VMR_He
        mass_frac['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)
        H += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)

        MMW = sum(mass_frac.values()) * np.ones(self.n_atm_layers)
        for k in list(mass_frac):
            mass_frac[k] = mass_frac[k] / MMW
        mass_frac['MMW'] = MMW

        with np.errstate(divide='ignore', invalid='ignore'):
            CO = float(np.nanmean(C / O))
            C_H = float(np.nanmean(np.log10(C / H))) - LOG_CH_SOLAR
        return mass_frac, CO, C_H

    def equilibrium_chemistry(self, params):
        """pRT3 pre-calculated equilibrium table -> mass fractions.

        ``C_H`` and ``C/O`` are the free parameters; ``[Z/H]`` is derived as
        ``[C/H] - log10((C/O)/(C/O)_solar)``.  Species the table does not carry
        are injected afterwards: ``13CO`` scaled off ``12CO`` by the sampled
        isotopologue ratio, ``HF`` from the ``[F/H]`` parameterisation, and
        ``Ca`` scaled from solar by the derived metallicity.
        """
        from petitRADTRANS.chemistry.utils import simplify_species_list

        C_H = float(params['C_H'])
        CO = float(params['C/O'])
        ZH = C_H - np.log10(CO / CO_SOLAR)

        log_Pquench = params.get('log_Pquench', None)
        carbon_pressure_quench = 10 ** float(log_Pquench) if log_Pquench is not None else None

        mf_eq, MMW, _ = self.eq_chem.interpolate_mass_fractions(
            co_ratios=CO * np.ones(self.n_atm_layers),
            log10_metallicities=ZH * np.ones(self.n_atm_layers),
            temperatures=self.temperature,
            pressures=self.pressure,
            carbon_pressure_quench=carbon_pressure_quench,
            full=True,
        )

        mass_frac = {}
        simple_names = simplify_species_list(self.species)
        ratio_12_13 = 10 ** float(params.get('log_12CO_13CO', np.log10(70.0)))

        for prt_name, simple in zip(self.species, simple_names):
            if prt_name == '13C-16O':
                mass_frac['13C-16O'] = mf_eq['CO'] * (29.002355 / 28.009999) / ratio_12_13
            elif prt_name == '40Ca':
                pass                      # injected below from ZH
            elif simple in mf_eq:
                mass_frac[prt_name] = mf_eq[simple]
            else:
                warnings.warn('%s (%s) not in the equilibrium table, skipping.'
                              % (prt_name, simple))

        mass_frac['H2'] = mf_eq['H2']
        mass_frac['He'] = mf_eq['He']
        mass_frac['MMW'] = MMW

        _mean_MMW = float(np.mean(MMW))
        _vmr_h2 = float(np.mean(mf_eq['H2'])) * _mean_MMW / 2.016

        if '1H-19F' in self.species and 'F_H' in params:
            log_vmr_hf = (float(params['F_H']) + LOG_F_H_SOLAR
                          + np.log10(2.0 * max(_vmr_h2, 1e-10)))
            mass_frac['1H-19F'] = (20.006243 * 10 ** log_vmr_hf
                                   * np.ones(self.n_atm_layers) / _mean_MMW)

        if '40Ca' in self.species:
            log_vmr_ca = (LOG_CA_H_SOLAR + ZH
                          + np.log10(2.0 * max(_vmr_h2, 1e-10)))
            mass_frac['40Ca'] = (40.078 * 10 ** log_vmr_ca
                                 * np.ones(self.n_atm_layers) / _mean_MMW)

        return mass_frac, CO, C_H

    # ---- radiative transfer ----

    def make_prt_flux(self):
        """Run pRT3 and cache the rest-frame spectrum.

        Returns
        -------
        wl : ndarray
            Wavelength in **metres** (the unit
            :mod:`exocrires.input_spectra_preparation` expects).
        flux : ndarray
            Flux in W m^-2 um^-1, converted from pRT3's native
            erg s^-1 cm^-2 cm^-1.  The absolute scale is irrelevant to the
            likelihood (it is marginalised) but the conversion is kept so that
            plotted spectra are in the same units as the .npz templates.
        """
        if self._prt_wl is not None:
            return self._prt_wl, self._prt_flux

        wl, flux, _ = self.radtrans.calculate_flux(
            temperatures=self.temperature,
            mass_fractions=self.mass_fractions,
            reference_gravity=self.gravity,
            mean_molar_masses=self.MMW,
            frequencies_to_wavelengths=True,
        )
        wl = wl * 1e-2            # cm -> m
        flux = flux * 1e-7        # erg s^-1 cm^-2 cm^-1 -> W m^-2 um^-1

        self._prt_wl = wl
        self._prt_flux = flux
        return wl, flux


# ============================================================
# 4. RETRIEVER
# ============================================================

class SpectraRetriever:
    """Drive a pRT-3 retrieval against an EXOCRIRES 2-D ABBA stack.

    The observation-side products (stack, variance, transmission, stellar
    model) are loaded once; the stellar component of the model matrix - which
    depends only on the data, never on the sampled parameters - is built once;
    and each likelihood call then costs one pRT evaluation, one preparation
    pass, one ``fit_planet`` per chip and one ``fast_likelihood_map``.

    Parameters
    ----------
    path_root : str
        Simulation root, e.g. ``/Users/richard/Simulation_WISPIT_1``.
    folder_name, day : str
        Target folder and day folder, e.g. ``'WISPIT_1b'``, ``'260609'``.
    model_name : str
        Planet model folder, e.g. ``'1650-4.0--0.4'`` (only used to locate the
        stack that was produced with it; the retrieval itself no longer uses
        that model's spectrum).
    star_dat : str
        Path (relative to ``path_root``) of the prepared stellar spectrum used
        to derive the telluric/instrument transmission.
    band : str
        CRIRES+ setting, default ``'K2166'``.
    Sep_p : int
        Companion separation in pixels; sets the extraction geometry exactly as
        in the 2dspectra/CC/likelihood notebooks.
    species : list of str, optional
        pRT line species.  Defaults to :data:`KBAND_SPECIES_MINIMAL` in testing
        mode and :data:`EQ_SPECIES_PRT3` otherwise.
    testing : bool
        Testing mode: coarse opacity sampling, a reduced species list and small
        sampler defaults.  It no longer restricts the order set - pass ``order``
        for a single-order smoke test.
    order : int, optional
        Fit exactly this one order.  Convenience for smoke tests; equivalent to
        ``orders=[order]``.  Order 2 carries the strongest planet signal on the
        WISPIT 1b stack (delta lnL ~ 60 at v_sys, against ~9 for order 5), so
        it is the sensible single order to validate against; order 5 is
        noise-dominated and the reference notebook's own map peaks at -6 km/s
        there.
    orders : sequence of int, optional
        Explicit order indices to fit.  Highest precedence.
    wavelength_range : (float, float), optional
        Fit every order overlapping this range, in **nm**.  A convenient way to
        say "everything redward of the deep tellurics" without knowing indices.
    drop_leading_orders : int
        How many leading orders to drop when nothing else is specified
        (default 2).  On K2166 those are 1921-1961 nm and 1990-2031 nm, the two
        bluest orders, which sit in deep telluric absorption and carry little
        planet signal - the same two the EXOCRIRES notebooks skip with
        ``range(2, n_ord)``.  The default therefore fits the remaining 5 orders
        (2064-2472 nm), which is the normal production configuration.
    lbl_opacity_sampling : int
        pRT line-by-line downsampling factor.  The native lbl grid is R=1e6, so
        the model resolution is 1e6/sampling.  Keep this comfortably above the
        instrument's R=1e5, because
        :meth:`Spectra_Preparator.gaussian_broaden` convolves to R without
        deconvolving the input resolution - at sampling=10 the model would
        already sit at R=1e5 and be broadened twice.  Default 4 (R=2.5e5).
    n_atm_layers : int
        Number of pressure layers, log-spaced over ``pressure_range``.
    prep_T, prep_G, prep_Z : float
        Temperature, log g and metallicity identifying the synthetic spectrum
        :class:`input_spectra_preparation.Spectra_Preparator` builds the
        transmission from.  **Required** - they describe the system, not the
        method, so there is no sensible default here.
    spectra_input_folder : str
        Folder (under ``path_root``) holding the prepared input spectra.
    v_sys : float, optional
        System velocity in km/s.  Carried for the driver's convenience.
    log_g_true, truth_npz, target_name : optional
        Injected log g, the reference model .npz and a label for this dataset.
        Purely descriptive - used by drivers for reporting and by
        :meth:`check_truth_consistency`.
    objective : {'delta_lnL', 'lnL'}
        What MultiNest samples.  See the module docstring.
    """

    def __init__(self,
                 path_root,
                 folder_name,
                 day,
                 model_name,
                 star_dat,
                 band='K2166',
                 Sep_p=25,
                 species=None,
                 testing=True,
                 order=None,
                 orders=None,
                 wavelength_range=None,
                 drop_leading_orders=2,
                 lbl_opacity_sampling=4,
                 n_atm_layers=50,
                 pressure_range=(-5.0, 1.0),
                 window=300,
                 polyorder=2,
                 vsini=20.0,
                 prep_T=None,
                 prep_G=None,
                 prep_Z=None,
                 spectra_input_folder='/spectra_input',
                 v_sys=None,
                 log_g_true=None,
                 truth_npz=None,
                 target_name=None,
                 normalize_model='per_order_median',
                 t_min=150.0,
                 t_max=6000.0,
                 pt_node_set='piette8',
                 ld_coeff=None,
                 instrument_R=1e5,
                 objective='delta_lnL',
                 output_dir=None,
                 prefix='retrieval_'):

        # The preparator's (T, G, Z) select the synthetic template the
        # transmission is derived against; they describe the OBSERVED SYSTEM,
        # not the method, so this module refuses to guess them.  Pass them from
        # the project's target table (see the module docstring).  Checked first,
        # before any path is touched, so a misconfigured target fails fast.
        if prep_T is None or prep_G is None or prep_Z is None:
            raise ValueError(
                'prep_T, prep_G and prep_Z are required - they identify the '
                'synthetic spectrum used to build the transmission and are '
                'properties of the system being analysed, not of this module. '
                'Typically they are the model that produced the stack, e.g. '
                "model_name '1650-4.0--0.4' -> prep_T=1650, prep_G=4.0, "
                'prep_Z=-0.4.')
        self.prep_T, self.prep_G, self.prep_Z = prep_T, prep_G, prep_Z

        self.path_root = path_root
        self.folder_name = folder_name
        self.day = day
        self.model_name = model_name
        self.band = band
        self.Sep_p = int(Sep_p)
        self.testing = bool(testing)
        self.lbl_opacity_sampling = int(lbl_opacity_sampling)
        self.n_atm_layers = int(n_atm_layers)
        self.window = int(window)
        self.polyorder = int(polyorder)
        self.vsini = float(vsini)
        if normalize_model not in ('per_order_median', 'global_median', None):
            raise ValueError("normalize_model must be 'per_order_median', "
                             "'global_median' or None")
        self.normalize_model = normalize_model
        # Physical guard on the P-T profile.  The dT priors are independent
        # uniforms, so nothing stops their sum from exceeding T_anchor and
        # driving the upper atmosphere below zero; make_pt would then silently
        # clamp at 1 K and hand pRT a nonsense profile.  Samples outside
        # [t_min, t_max] are rejected instead.
        if pt_node_set not in PT_NODE_SETS:
            raise ValueError('pt_node_set must be one of %s, got %r'
                             % (sorted(PT_NODE_SETS), pt_node_set))
        self.pt_node_set = pt_node_set
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self._n_rejected_pt = 0
        self.instrument_R = float(instrument_R)
        self.objective = objective
        self.prefix = prefix

        if objective not in ('delta_lnL', 'lnL'):
            raise ValueError("objective must be 'delta_lnL' or 'lnL'")

        self.path_a = '%s/%s/%s' % (path_root, folder_name, day)
        self.path_file = '%s/planet/%s/%s' % (self.path_a, model_name, band)
        self.star_file = '%s/%s' % (path_root, star_dat)
        self.output_dir = output_dir or ('%s/retrieval' % self.path_file)
        os.makedirs(self.output_dir, exist_ok=True)

        self.lambda_range = info.lambda_range
        self.band_idx = self.lambda_range['key'].index(band)
        self.wave_range = np.array(self.lambda_range['wave'][self.band_idx])  # (n_ord,2) nm

        # Same extraction geometry as every other stage of the pipeline.
        self.central_pix = max(17, self.Sep_p + 5)
        self.target_pix = int(self.central_pix + self.Sep_p)

        self.pressure = np.logspace(pressure_range[0], pressure_range[1],
                                    self.n_atm_layers)

        self._load_observation()

        # Order selection.  n_ord comes from the stack itself, so the default
        # adapts to whatever (orders, rows, px) the input spectra actually have.
        self.drop_leading_orders = int(drop_leading_orders)
        self.orders = self._select_orders(orders, wavelength_range, order)

        if species is None:
            species = KBAND_SPECIES_MINIMAL if self.testing else EQ_SPECIES_PRT3
        self.species = [PRT3_SPECIES_OVERRIDES.get(s, s) for s in species]

        self.ld_coeff = ld_coeff
        self.prep = input_spectra_preparation.Spectra_Preparator(
            folder_name=spectra_input_folder, T=prep_T, G=prep_G, Z=prep_Z,
            path=path_root)

        # Optional descriptors of the dataset.  Nothing in this class computes
        # with them; they travel with the retriever so drivers and plotting code
        # can report injected truth without a second lookup table.
        self.v_sys = v_sys
        self.log_g_true = log_g_true
        self.truth_npz = truth_npz
        self.target_name = target_name

        self.radtrans = None
        self.eq_chem = None
        self._star_component = None
        self._lnL_star_only = None
        self._n_calls = 0
        self._t_prt = 0.0
        self._t_prep = 0.0
        self._t_like = 0.0

    def _select_orders(self, orders, wavelength_range, order):
        """Resolve which spectral orders to fit.

        Precedence: explicit ``orders`` > ``wavelength_range`` > single
        ``order`` > every order past ``drop_leading_orders``.  The fallback is
        driven by ``self.n_ord``, read from the input stack, so it follows the
        shape of whatever data was loaded.
        """
        available = list(range(self.n_ord))

        if orders is not None:
            chosen = [int(o) for o in orders]
        elif wavelength_range is not None:
            lo, hi = float(wavelength_range[0]), float(wavelength_range[1])
            if lo > hi:
                lo, hi = hi, lo
            # Any order whose span overlaps the requested window.
            chosen = [o for o in available
                      if (self.wave_range[o][1] >= lo
                          and self.wave_range[o][0] <= hi)]
            if not chosen:
                raise ValueError(
                    'no order overlaps %.1f-%.1f nm; this band covers %.1f-%.1f nm'
                    % (lo, hi, self.wave_range[:, 0].min(),
                       self.wave_range[:, 1].max()))
        elif order is not None:
            chosen = [int(order)]
        else:
            chosen = available[self.drop_leading_orders:]

        bad = [o for o in chosen if o not in available]
        if bad:
            raise ValueError('order(s) %s outside the stack, which has %d '
                             'order(s): %s' % (bad, self.n_ord, available))
        if not chosen:
            raise ValueError('empty order selection')
        return sorted(set(chosen))

    def check_truth_consistency(self, npz_path):
        """Warn if a reference .npz post-dates the stack it is compared against.

        A model .npz can be regenerated long after the ABBA stack was built from
        it, in which case its stored P-T profile and spectrum describe a
        *different* planet than the one in the data.  That happened on this
        project: wispit_1b_spectra.npz (2026-09-07) versus the 260609 stack
        (2026-06-14).  Returns True when the npz is at least as old as the
        stack, i.e. safe to treat as ground truth.
        """
        stack = '%s/stack_sub_badpix.npy' % self.path_file
        if not (os.path.exists(npz_path) and os.path.exists(stack)):
            return None
        t_npz, t_stack = os.path.getmtime(npz_path), os.path.getmtime(stack)
        if t_npz > t_stack:
            warnings.warn(
                '%s was modified AFTER %s - it may describe a different model '
                'than the one in this stack, so its P-T profile is not ground '
                'truth here.' % (os.path.basename(npz_path),
                                 os.path.basename(stack)))
            return False
        return True

    def describe_orders(self):
        """One line per fitted order: index, wavelength span, valid pixels."""
        lines = []
        for o in self.orders:
            row = self.stack_sub_badpix[o, self.target_pix]
            lines.append('  order %d : %8.2f - %8.2f nm   %5d valid px'
                         % (o, self.wave_range[o][0], self.wave_range[o][1],
                            int(np.isfinite(row).sum())))
        return '\n'.join(lines)

    # ---- observation side ----

    def _load_observation(self):
        """Load the ABBA stack, variance, transmission and stellar model."""
        need = ['ABBA_stack.npy', 'stack_sub_badpix.npy', 'fit_matrix.npy']
        for f in need:
            p = '%s/%s' % (self.path_file, f)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    '%s missing - run the 2dspectra / CC stages first.' % p)

        self.stack = np.load('%s/ABBA_stack.npy' % self.path_file)
        self.stack_sub_badpix = np.load('%s/stack_sub_badpix.npy' % self.path_file)
        self.fit_matrix = np.load('%s/fit_matrix.npy' % self.path_file)
        self.transmission_tot = np.load('%s/sky/%s/transmission.npy'
                                        % (self.path_a, self.band))
        self.transmission_atmos = np.load('%s/sky/%s/transmission_atmos.npy'
                                          % (self.path_a, self.band))

        self.n_ord, self.n_rows, self.n_pix = self.stack_sub_badpix.shape
        if self.target_pix >= self.n_rows:
            raise ValueError('target_pix=%d but the stack has %d rows - Sep_p '
                             'disagrees with how the stack was extracted'
                             % (self.target_pix, self.n_rows))

        var_path = '%s/variance.npy' % self.path_file
        if os.path.exists(var_path):
            self.var = np.load(var_path)
        else:
            series = np.load('%s/ABBA_series.npy' % self.path_file)
            self.var = processor_likelihood_map().covariance_calculator(
                self.fit_matrix, series, series.shape[1])[0]
            np.save(var_path, self.var)
            del series

        # Master (stellar) spectrum: the 3 rows centred on the star, per chip.
        self.master_matrix = np.zeros((self.n_ord, self.n_pix))
        for i in range(self.n_ord):
            for x in range(3):
                sl = slice(x * 2046, (x + 1) * 2046)
                self.master_matrix[i, sl] = np.trapz(
                    self.stack_sub_badpix[i, self.central_pix - 1:self.central_pix + 2, sl],
                    axis=0)

        # Transmission = observed master / PHOENIX stellar model, i.e. the
        # telluric+instrument response the planet's light also passes through.
        phoenix = np.loadtxt(self.star_file)
        swave, sspec = phoenix[:, 0] * 1e9, phoenix[:, 1]      # m -> nm
        model_matrix_star = np.zeros((self.n_ord, self.n_pix))
        for i in range(self.n_ord):
            m = (swave >= self.wave_range[i][0]) & (swave <= self.wave_range[i][1])
            model_matrix_star[i][0:2046 * 3] = np.interp(
                self.transmission_atmos[i, 0:2046 * 3, 0] * 1e9, swave[m], sspec[m])

        self.transmission_matrix = np.zeros_like(self.master_matrix)
        for i in range(self.n_ord):
            sl = slice(0, 3 * 2046)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.transmission_matrix[i, sl] = (self.master_matrix[i, sl]
                                                   / model_matrix_star[i, sl])

        self.fitter = Spectra_fitting(self.stack, self.transmission_tot[:, :, 1],
                                      window_length=self.window,
                                      polyorder=self.polyorder)
        self.fitter.transmission_matrix = self.transmission_matrix

    def _chip_selection(self, order, x):
        """Finite-pixel mask for one (order, detector) chip, and its master."""
        sl = slice(x * 2046, (x + 1) * 2046)
        master_int = np.trapz(
            self.stack_sub_badpix[order, self.central_pix - 1:self.central_pix + 2, sl],
            axis=0)
        sel = (np.isfinite(master_int)
               & np.isfinite(self.stack_sub_badpix[order, self.target_pix][sl]))
        return sl, master_int, sel

    def build_star_component(self):
        """Stellar component of the model matrix - built once.

        ``fit_star`` depends only on the master spectrum and the observation at
        the planet row, never on the planet template, so this is invariant
        across the whole retrieval.  The original notebook rebuilt it inside
        the velocity loop; hoisting it out is the single biggest saving in the
        likelihood path.
        """
        if self._star_component is not None:
            return self._star_component

        star = np.zeros((self.n_ord, self.n_pix))
        for order in self.orders:
            for x in range(3):
                sl, master_int, sel = self._chip_selection(order, x)
                if sel.sum() <= self.window:
                    continue
                star_c = self.fitter.fit_star(
                    1, master_int[sel],
                    observation=self.stack_sub_badpix[order, self.target_pix, sl][sel],
                    window_length=self.window, polyorder=self.polyorder)
                star[order][sl][sel] = star_c[1]
        self._star_component = star
        return star

    # ---- forward model side ----

    def get_atmosphere(self, redo=False):
        """Build (or reuse) the Radtrans object covering the fitted orders."""
        if self.radtrans is not None and not redo:
            return self.radtrans

        from petitRADTRANS.radtrans import Radtrans

        wl_pad = 7.0    # nm, room for the RV shift at the edges
        lo = min(self.wave_range[o][0] for o in self.orders) - wl_pad
        hi = max(self.wave_range[o][1] for o in self.orders) + wl_pad
        boundaries = np.array([lo, hi]) * 1e-3        # nm -> um

        print('Building Radtrans: %.1f-%.1f nm, %d species, lbl sampling %d'
              % (lo, hi, len(self.species), self.lbl_opacity_sampling))
        self.radtrans = Radtrans(
            line_species=self.species,
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=['H2--H2', 'H2--He'],
            wavelength_boundaries=boundaries,
            line_opacity_mode='lbl',
            line_by_line_opacity_sampling=self.lbl_opacity_sampling,
            pressures=self.pressure,
        )
        return self.radtrans

    def get_eq_chem(self):
        """Load the pRT equilibrium-chemistry table (equilibrium mode only)."""
        if self.eq_chem is None:
            from petitRADTRANS.chemistry.pre_calculated_chemistry import (
                PreCalculatedEquilibriumChemistryTable)
            self.eq_chem = PreCalculatedEquilibriumChemistryTable()
            self.eq_chem.load()
        return self.eq_chem

    def prepare_template(self, wl_m, flux, rv, ld_coeff=None, vsini=None):
        """Rest-frame pRT spectrum -> detector-sampled planet template.

        This is the step the user-facing contract calls "pass through the same
        spectra_preparation module": it uses
        :meth:`Spectra_Preparator.apply_broadening` (rotational + instrumental)
        and :meth:`Spectra_Preparator.radial_velocity_shift`, the identical
        calls the notebook made on the pre-computed .npz spectra.

        Returns
        -------
        ndarray, shape (n_ord, n_pix)
            The template interpolated onto the transmission wavelength grid,
            zero outside the 6138 valid pixels - the layout ``fit_planet``
            expects.
        """
        if ld_coeff is None:
            ld_coeff = self.ld_coeff
        if ld_coeff is None:
            raise RuntimeError('limb-darkening coefficient not set; call '
                               'compute_ld_coeff() or pass ld_coeff=')
        if vsini is None:
            vsini = self.vsini

        lam_range = self.wave_range          # (n_ord, 2), nm
        wl_grid, flux_b = self.prep.apply_broadening(
            wave=wl_m, flux=flux, lam_range=lam_range, ld_coeff=float(ld_coeff),
            R=self.instrument_R, rotation=float(vsini),
            step_size='2x', interpolation='spline')

        shifted = self.prep.radial_velocity_shift(wave=wl_grid, flux=flux_b,
                                                  velocity=float(rv))

        trans = self.transmission_tot
        template = np.zeros((trans.shape[0], self.n_pix))
        for i in self.orders:
            wl = trans[i, :, 0][0:6138]
            m = (shifted[:, 0] >= np.min(wl)) & (shifted[:, 0] <= np.max(wl))
            if m.sum() < 2:
                continue
            template[i, 0:6138] = np.interp(wl, shifted[m][:, 0], shifted[m][:, 1])

        return self._normalize_template(template)

    def _normalize_template(self, template):
        """Put the model template on a parameter-independent flux scale.

        WHY THIS IS NECESSARY.  ``fast_likelihood_map`` marginalises the model
        amplitude with an *improper flat* prior, so its log-likelihood carries a
        ``-0.5 * ln det(M^T Sigma^-1 M)`` term.  Scaling the planet column by k
        therefore adds ``-ln(k)`` to ln L (an Occam penalty against amplitude).

        NOTE (2026-09-22): the sign above was flipped on this date in
        ``fast_likelihood_calculator.py`` / ``spectra_processor.py`` (a genuine
        bug, found and confirmed against Landman et al. 2024 eq. 15 on
        2026-09-05: the code previously computed ``+0.5 * ln det(...)`` /
        ``+ln(k)``, rewarding rather than penalising larger amplitude). The
        measurement below ("+6.9078 = ln(1e3)") and the "reward for hotter,
        brighter atmospheres" description are from *before* that fix and are
        now the mirror image: today the un-normalised term would instead
        *penalise* brighter/hotter models by the same magnitude. Either sign is
        still a spurious, parameter-dependent artifact with no physical
        content, so normalising it away here remains equally necessary; only
        the direction of the bias it would otherwise cause has flipped.

        For the original study that was harmless: every template on the velocity
        grid came from one .npz at one fixed scale, so this term was a constant
        offset that cancelled everywhere.  In a *retrieval* it is not harmless -
        pRT's absolute flux rises steeply with temperature (a factor ~29 across
        the T_anchor prior on this setup), so the term becomes a spurious,
        temperature-dependent artifact worth up to ~3.4 nat per order, ~17 nat
        over five orders, against a real signal of ~155. Left uncorrected it
        biases the posterior in whichever direction the current sign implies.

        Dividing each order's template by its own median removes it.  That costs
        nothing in information: the per-order likelihood assembly already gives
        every order its own free amplitude, so inter-order flux ratios were
        never constrained in the first place.  This is the EXOCRIRES analogue of
        the guidebook's ``_apply_normalization``, applied where this pipeline
        actually needs it.

        ``None`` disables it and reproduces the original notebook's unnormalised
        templates exactly - correct for a fixed-template velocity scan, wrong
        for a retrieval.
        """
        if self.normalize_model is None:
            return template

        if self.normalize_model == 'global_median':
            seg = template[self.orders, 0:6138]
            good = np.isfinite(seg) & (seg != 0)
            if good.any():
                med = np.median(np.abs(seg[good]))
                if med > 0:
                    template = template / med
            return template

        for i in self.orders:                       # 'per_order_median'
            seg = template[i, 0:6138]
            good = np.isfinite(seg) & (seg != 0)
            if not good.any():
                continue
            med = np.median(np.abs(seg[good]))
            if med > 0:
                template[i] = template[i] / med
        return template

    def compute_ld_coeff(self, wl_m, flux):
        """Limb-darkening coefficient, computed once from a reference spectrum.

        ``compute_limb_darkening(Mode='custom')`` builds synthetic stellar
        models through ``exotic_ld`` and is far too slow for the likelihood
        path; it also depends only weakly on the atmospheric parameters, so the
        pipeline's own convention of deriving it once from the model spectrum
        is preserved here - just hoisted out of the loop.
        """
        throughput = {'K2166': 'SPHERE_IRDIS_B_Ks.dat',
                      'L3377': 'Paranal_ERIS_Lp.dat',
                      'L3302': 'Paranal_ERIS_Lp.dat',
                      'M4368': 'Paranal_ERIS_Mp.dat'}[self.band]
        tp = '%s/throughput/%s' % (self.path_root, throughput)
        wave_range = self.lambda_range['wave'][self.band_idx:self.band_idx + 1]
        # np.ravel(...)[0] rather than float(...[0]): compute_limb_darkening
        # returns shape (n_bands, 1), and float() on a size-1 *array* is an
        # error under numpy 2 (it was merely deprecated under numpy 1.x, which
        # is why the original notebook's float(...[0]) worked there).
        coeffs = self.prep.compute_limb_darkening(
            wavelength_range=wave_range, throughput_files=[tp],
            wave=wl_m * 1e10, flux=flux, Mode='custom')
        self.ld_coeff = float(np.ravel(coeffs)[0])
        return self.ld_coeff

    # ---- model matrix + likelihood ----

    def build_model_matrix(self, template):
        """Star + planet components -> the (1, 2, n_ord, 1, n_pix) model matrix.

        The planet half is the "spectra separation" step:
        :meth:`Spectra_fitting.fit_planet` removes the fraction of the planet
        signal that leaks into the stellar master spectrum, returning
        ``transmission * spec_planet - alpha_planet * master``.
        """
        star = self.build_star_component()
        mm = np.zeros((1, 2, self.n_ord, 1, self.n_pix))

        for order in self.orders:
            for x in range(3):
                sl, master_int, sel = self._chip_selection(order, x)
                if sel.sum() <= self.window:
                    continue
                planet_c = self.fitter.fit_planet(
                    1, master_int[sel],
                    self.transmission_matrix[order, sl][sel],
                    spec_planet=template[order, sl][sel],
                    window_length=self.window, polyorder=self.polyorder)
                mm[0][0][order, 0][sl][sel] = star[order][sl][sel]
                mm[0][1][order, 0][sl][sel] = planet_c[1]
        return mm

    @staticmethod
    def _clean_variance(v):
        v = np.array(v, dtype=float)
        v[~np.isfinite(v) | (v <= 0)] = np.nan
        return v

    def _order_inputs(self, model_matrix, order, components):
        obs = np.nan_to_num(self.stack_sub_badpix[order], nan=0.0, posinf=0.0,
                            neginf=0.0)[self.target_pix:self.target_pix + 1, :]
        mdl = np.nan_to_num(model_matrix[:, components, order, :, :],
                            nan=0.0, posinf=0.0, neginf=0.0)
        variance = self.var[order][self.target_pix:self.target_pix + 1]
        return obs, mdl, variance

    def lnL_star_only(self):
        """Star-only marginal likelihood, summed over orders - computed once.

        Independent of every sampled parameter, so it is the constant that
        turns ``lnL`` into ``delta_lnL``.
        """
        if self._lnL_star_only is not None:
            return self._lnL_star_only
        self._lnL_star_only_per_order = {}

        wl = processor_likelihood_map()
        mm = np.zeros((1, 2, self.n_ord, 1, self.n_pix))
        star = self.build_star_component()
        for order in self.orders:
            mm[0][0][order, 0] = star[order]

        total = 0.0
        for order in self.orders:
            obs, mdl0, variance = self._order_inputs(mm, order, [0])
            mdl0 = mdl0[:, 0]                       # (1, 1, Nd)
            val = float(np.sum(wl.fast_likelihood_map(
                observation_matrix=obs, model_matrix=mdl0, n_vgrids=1,
                sigma_matrix=self._clean_variance(variance), prior=1,
                matrix_components=1,
                skip_velocity_grid_single_component=True)))
            self._lnL_star_only_per_order[order] = val
            total += val
        self._lnL_star_only = total
        return total

    def per_order_delta(self, params):
        """delta ln L contributed by each fitted order, as a dict.

        The retrieval's objective is the sum of these.  Useful for spotting an
        order that contributes nothing (or, worse, fits noise) before trusting a
        posterior - on the WISPIT 1b stack order 2 dominates and the rest each
        contribute ~8-13.
        """
        self.lnL_star_only()          # populates the per-order star-only cache
        _, parts = self.log_likelihood(params, return_parts=True)
        return {o: parts['per_order'][o] - self._lnL_star_only_per_order[o]
                for o in self.orders}

    def log_likelihood(self, params, return_parts=False):
        """Full chain: parameters -> delta ln L.

        Parameters
        ----------
        params : dict
            Physical parameters, including ``rv`` and ``log_g``.
        return_parts : bool
            Also return the per-order breakdown and timings.
        """
        # Cheap physical rejection before paying for a pRT evaluation.
        _T = pt_profile(params, self.pressure, self.pt_node_set)
        if _T.min() < self.t_min or _T.max() > self.t_max:
            self._n_rejected_pt += 1
            if return_parts:
                return -np.inf, {'per_order': {}, 'rejected': 'pt_out_of_range'}
            return -np.inf

        t0 = time.time()
        fm = pRT_ForwardModel(self.get_atmosphere(), params, self.species,
                              self.pressure,
                              eq_chem=self.get_eq_chem()
                              if params.get('chemistry') == 'equilibrium' else None,
                              pt_node_set=self.pt_node_set)
        wl_m, flux = fm.make_prt_flux()
        t1 = time.time()

        # vsini and epsilon are honoured as *sampled* parameters when present,
        # falling back to the construction-time values otherwise.  Without this
        # a prior on 'vsini' would be silently ignored.
        template = self.prepare_template(
            wl_m, flux, rv=params['rv'],
            ld_coeff=params.get('epsilon', None),
            vsini=params.get('vsini', None))
        t2 = time.time()

        mm = self.build_model_matrix(template)

        wl_map = processor_likelihood_map()
        per_order = {}
        total = 0.0
        for order in self.orders:
            obs, mdl, variance = self._order_inputs(mm, order, [0, 1])
            val = float(np.sum(wl_map.fast_likelihood_map(
                observation_matrix=obs, model_matrix=mdl, n_vgrids=1,
                sigma_matrix=self._clean_variance(variance), prior=1,
                matrix_components=2)))
            per_order[order] = val
            total += val
        t3 = time.time()

        self._n_calls += 1
        self._t_prt += t1 - t0
        self._t_prep += t2 - t1
        self._t_like += t3 - t2

        if self.objective == 'delta_lnL':
            total = total - self.lnL_star_only()

        if return_parts:
            return total, {'per_order': per_order, 'template': template,
                           'model_matrix': mm, 'forward_model': fm,
                           't_prt': t1 - t0, 't_prep': t2 - t1, 't_like': t3 - t2}
        return total

    def best_fit_components(self, params):
        """Per-order decomposition of the fit at one point in parameter space.

        Returns, for each fitted order, the observed planet-row spectrum, the
        two model columns, and the maximum-likelihood amplitudes obtained from
        the same generalised least squares the marginal likelihood integrates
        over::

            a = (M^T W M)^-1 M^T W d ,   W = diag(1 / variance)

        Everything the diagnostic plots need.  Keys per order: ``wave`` (nm),
        ``data``, ``star``, ``planet`` (the two model columns), ``amp``
        (2-vector), ``mask``, ``model`` (a[0]*star + a[1]*planet).
        """
        fm = pRT_ForwardModel(self.get_atmosphere(), params, self.species,
                              self.pressure,
                              eq_chem=self.get_eq_chem()
                              if params.get('chemistry') == 'equilibrium' else None,
                              pt_node_set=self.pt_node_set)
        wl_m, flux = fm.make_prt_flux()
        template = self.prepare_template(
            wl_m, flux, rv=params['rv'],
            ld_coeff=params.get('epsilon', None),
            vsini=params.get('vsini', None))
        mm = self.build_model_matrix(template)

        out = {}
        for order in self.orders:
            obs, mdl, var = self._order_inputs(mm, order, [0, 1])
            d = obs[0]
            M = mdl[0, :, 0, :].T                  # (n_pix, 2)
            v = self._clean_variance(var)[0]
            good = (np.isfinite(d) & np.isfinite(v) & (v > 0)
                    & np.any(M != 0, axis=1))
            if good.sum() < 10:
                continue
            W = 1.0 / v[good]
            A = (M[good].T * W) @ M[good]
            b = (M[good].T * W) @ d[good]
            try:
                amp = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                amp = np.linalg.lstsq(A, b, rcond=None)[0]
            wave = self.transmission_tot[order, :, 0] * 1e9      # m -> nm
            out[order] = {'wave': wave, 'data': d, 'star': M[:, 0],
                          'planet': M[:, 1], 'amp': amp, 'mask': good,
                          'model': amp[0] * M[:, 0] + amp[1] * M[:, 1],
                          'variance': v}
        return out

    # ---- MultiNest ----

    def PMN_lnL(self, cube=None, ndim=None, nparams=None):
        """MultiNest log-likelihood callback.

        Any failure inside the forward model (an unphysical corner of the
        prior, a chemistry-table edge, a degenerate design matrix) returns a
        large negative value rather than raising, so one bad live point cannot
        abort a run.
        """
        try:
            lnL = self.log_likelihood(self.parameters.params)
        except Exception as exc:                        # noqa: BLE001
            if os.environ.get('EXOCRIRES_RETRIEVER_DEBUG'):
                import traceback
                traceback.print_exc()
            return -1e99
        if not np.isfinite(lnL):
            return -1e99
        return float(min(lnL, 1e10))

    def PMN_run(self, parameters, n_live_points=None, evidence_tolerance=0.5,
                resume=False, mpi=True, verbose=True,
                const_efficiency_mode=None, sampling_efficiency=0.5, **kwargs):
        """Launch MultiNest.

        Parameters
        ----------
        parameters : Parameters
            Prior object; its ``__call__`` is the prior transform.
        n_live_points : int
            Defaults to 50 in testing mode, 400 otherwise.
        const_efficiency_mode : bool, optional
            Defaults to True in testing mode and False otherwise.  Constant-
            efficiency mode makes a high-dimensional run finish quickly (it is
            what the guidebook uses), but MultiNest's own documentation warns
            that the **evidence is unreliable** in that mode unless
            ``sampling_efficiency`` is small (~0.05) and the live-point count is
            generous.  The diagnostic is that MultiNest's two evidence
            estimates - plain nested sampling and importance nested sampling -
            drift apart; in the 13-parameter testing run they differ by ~6.6 nat
            (58.6 vs 65.2).  Posterior *parameter* estimates are far more robust
            than the evidence here, so a testing run is still a valid smoke
            test.  For any evidence you intend to quote, set this False and
            raise ``n_live_points``.
        sampling_efficiency : float
            MultiNest's target sampling efficiency.
        """
        import pymultinest

        self.parameters = parameters
        if n_live_points is None:
            n_live_points = 50 if self.testing else 400
        if const_efficiency_mode is None:
            const_efficiency_mode = bool(self.testing)

        # Force the one-off setup before sampling starts, so its cost and any
        # failure surface here rather than inside the first live point.
        self.get_atmosphere()
        self.build_star_component()
        self.lnL_star_only()

        print('Sampling %d parameters, %d live points, %d order(s), objective=%s'
              % (parameters.n_params, n_live_points, len(self.orders), self.objective))

        t0 = time.time()
        pymultinest.run(
            LogLikelihood=self.PMN_lnL,
            Prior=parameters,
            n_dims=parameters.n_params,
            outputfiles_basename='%s/%s' % (self.output_dir, self.prefix),
            verbose=verbose,
            const_efficiency_mode=const_efficiency_mode,
            sampling_efficiency=sampling_efficiency,
            n_live_points=n_live_points,
            resume=resume,
            evidence_tolerance=evidence_tolerance,
            n_iter_before_update=10,
            use_MPI=mpi,
            **kwargs)
        dt = time.time() - t0
        print('MultiNest finished in %.1f s | %d likelihood calls '
              '(pRT %.1f s, prep %.1f s, likelihood %.1f s)'
              % (dt, self._n_calls, self._t_prt, self._t_prep, self._t_like))
        if self._n_rejected_pt:
            print('  %d sample(s) rejected for leaving T in [%.0f, %.0f] K'
                  % (self._n_rejected_pt, self.t_min, self.t_max))

    def PMN_analyse(self, parameters=None):
        """Load MultiNest output; return (stats, equal-weighted posterior)."""
        import pymultinest
        parameters = parameters or self.parameters
        analyzer = pymultinest.Analyzer(
            n_params=parameters.n_params,
            outputfiles_basename='%s/%s' % (self.output_dir, self.prefix))
        stats = analyzer.get_stats()
        posterior = analyzer.get_equal_weighted_posterior()[:, :-1]
        return stats, posterior

    def summary(self, parameters=None):
        """Human-readable posterior summary; also written as summary.json."""
        stats, posterior = self.PMN_analyse(parameters)
        parameters = parameters or self.parameters
        # pymultinest's 'global evidence' IS the importance-sampled value;
        # the independent estimate to compare it against is the plain nested
        # sampling one.  Comparing 'global evidence' to
        # 'nested importance sampling global log-evidence' compares a number
        # with itself and can never warn.
        ins = stats.get('nested importance sampling global log-evidence')
        ns = stats.get('nested sampling global log-evidence')
        out = {'log_Z': stats['global evidence'],
               'log_Z_err': stats['global evidence error'],
               'log_Z_importance': ins,
               'log_Z_nested_sampling': ns,
               'objective': self.objective,
               'n_live_calls': self._n_calls,
               'params': {}}
        print('\nln Z = %.3f +- %.3f  (%s)'
              % (out['log_Z'], out['log_Z_err'],
                 'log Bayes factor vs star-only'
                 if self.objective == 'delta_lnL' else 'absolute'))
        if (ns is not None and ins is not None and np.isfinite(ns)
                and np.isfinite(ins) and abs(ns - ins) > 1.0):
            print('  WARNING: the two evidence estimates disagree by %.2f nat '
                  '(nested sampling %.3f vs importance sampling %.3f).\n'
                  '           ln Z is NOT converged - raise n_live_points and/or '
                  'tighten evidence_tolerance\n'
                  '           before quoting it.  Parameter posteriors are far '
                  'more robust than ln Z here.' % (abs(ns - ins), ns, ins))
        for i, key in enumerate(parameters.param_keys):
            m = stats['marginals'][i]
            lo, hi = m['1sigma']
            out['params'][key] = {'median': m['median'], 'lo': lo, 'hi': hi}
            print('  %-14s = %10.4f  (+%.4f / -%.4f)'
                  % (key, m['median'], hi - m['median'], m['median'] - lo))
        with open('%s/%ssummary.json' % (self.output_dir, self.prefix), 'w') as f:
            json.dump(out, f, indent=2)
        return out

    # ---- diagnostics ----

    def velocity_scan(self, params, velocities):
        """delta ln L as a function of RV at fixed atmospheric parameters.

        The retrieval analogue of the notebook's velocity map: useful to check
        that a forward model actually peaks at the system velocity before
        spending time on a full sampling run.
        """
        out = np.zeros(len(velocities))
        p = dict(params)
        for k, v in enumerate(velocities):
            p['rv'] = float(v)
            out[k] = self.log_likelihood(p)
        return out


# ============================================================
# 5. PRIOR FACTORIES  (generic - no system-specific numbers)
# ============================================================
#
# These builders assemble the parameter NAMES the forward model understands and
# attach whatever prior the caller supplies.  They deliberately carry no default
# ranges: a prior encodes what you believe about a particular object, so it
# belongs in that project's run_*.py, not in a module shared by every system.
#
# A prior spec is either a ``(lo, hi)`` uniform or the dict returned by
# :func:`gaussian`.  ``Parameters`` maps both onto the unit cube.


def uniform(lo, hi):
    """Uniform prior spec.  Provided for symmetry with :func:`gaussian`."""
    return [float(lo), float(hi)]


def gaussian(mu, sigma):
    """Gaussian prior spec, for use anywhere a ``(lo, hi)`` uniform is accepted.

    ``Parameters.__call__`` maps this through ``norm.ppf``.  The natural use is
    a parameter constrained by information outside the spectrum - surface
    gravity being the standard case, since mass and radius come from
    evolutionary models rather than from the line shapes.  Where log g is
    strongly degenerate with metallicity, a uniform prior lets that degeneracy
    run to the prior edge; a Gaussian anchors it instead.
    """
    return {'type': 'gaussian', 'mu': float(mu), 'sigma': float(sigma)}


def _as_prior(spec):
    """Normalise a prior spec: dicts pass through, pairs become ``[lo, hi]``."""
    if isinstance(spec, dict):
        return spec
    return [float(spec[0]), float(spec[1])]

def make_free_params(priors):
    """Encode a prior dictionary into the form :class:`Parameters` samples.

    This is the single entry point for priors.  The calling ``run_*.py`` writes
    out what it believes about its object as one flat dict; this function
    normalises it and checks it for outright mistakes.  The package holds no
    ranges of its own - see the module docstring.

    Parameters
    ----------
    priors : dict
        ``{name: (prior_spec, mathtext_label)}``.  Following the guidebook
        convention, the *shape* of ``prior_spec`` chooses the prior:

        * ``{'type': 'gaussian', 'mu': mu, 'sigma': sigma}``  -> Gaussian
        * anything else 2-element, e.g. ``[lo, hi]``          -> uniform

        so switching a parameter between the two is a one-line edit in the
        caller's dict and nothing here needs to know which is "right".  The
        label is optional: ``{name: prior_spec}`` works too.

    Returns
    -------
    dict
        The same mapping, normalised, ready for ``Parameters(free, fixed)``.

    Notes
    -----
    **Key order is significant** - MultiNest addresses parameters positionally,
    so this dict's order fixes the posterior's column order.  Keep it stable
    between a run and any later analysis of it.

    Names must be ones the forward model understands: the P-T set
    (``T_anchor``, ``dT_1``..``dT_7``), ``log_g``, ``rv``, optionally
    ``vsini``; then either equilibrium parameters (``C_H``, ``C/O``,
    ``log_12CO_13CO``, optionally ``log_Pquench``/``F_H``) or one
    ``log_<species>`` per free-chemistry species.  Unknown names pass through
    and the forward model raises if it cannot use them.
    """
    if not isinstance(priors, dict):
        raise TypeError('priors must be a dict {name: (spec, label)}, got %s'
                        % type(priors).__name__)
    if not priors:
        raise ValueError('priors is empty - nothing to sample')

    out = {}
    for name, entry in priors.items():
        # Accept both {name: (spec, label)} and the shorthand {name: spec}.
        if (isinstance(entry, tuple) and len(entry) == 2
                and isinstance(entry[1], str)):
            spec, label = entry
        else:
            spec, label = entry, name

        if isinstance(spec, dict) and spec.get('type') == 'gaussian':
            if 'mu' not in spec or 'sigma' not in spec:
                raise ValueError("%s: gaussian prior needs 'mu' and 'sigma'"
                                 % name)
            if not float(spec['sigma']) > 0:
                raise ValueError('%s: gaussian sigma must be > 0, got %r'
                                 % (name, spec['sigma']))
            spec = {'type': 'gaussian', 'mu': float(spec['mu']),
                    'sigma': float(spec['sigma'])}
        elif isinstance(spec, dict):
            raise ValueError(
                "%s: prior dicts must carry type='gaussian' (got %r).  For a "
                'uniform prior pass [lo, hi].' % (name, spec.get('type')))
        else:
            try:
                lo, hi = (float(spec[0]), float(spec[1]))
            except (TypeError, IndexError, ValueError, KeyError):
                raise ValueError(
                    '%s: prior must be [lo, hi] or a gaussian dict, got %r'
                    % (name, spec))
            if not hi > lo:
                raise ValueError('%s: uniform prior needs hi > lo, got [%g, %g]'
                                 % (name, lo, hi))
            spec = [lo, hi]

        out[name] = (spec, label)

    # The Piette profile is built from increments, so dT_i >= 0 is what keeps it
    # monotonic - Parameters.__call__ has no ordering logic.  A negative lower
    # bound is legal (an inversion may be what you want) but is almost always a
    # typo, so say so rather than silently producing a non-monotonic profile.
    for name, (spec, _) in out.items():
        if name.startswith('dT_') and not isinstance(spec, dict) and spec[0] < 0:
            warnings.warn(
                '%s has a negative lower bound (%g): the P-T profile will not '
                'be monotonic there.  Intended?' % (name, spec[0]))
    return out


# ============================================================
# 6. BUILDING A RETRIEVER FROM A PROJECT TARGET TABLE
# ============================================================


def build_retriever(config, **overrides):
    """Construct a :class:`SpectraRetriever` from a target-description dict.

    ``config`` is a plain dict of :class:`SpectraRetriever` keyword arguments -
    the shape a project's own target table takes.  ``overrides`` win, so one
    entry can be pointed at another day folder or stellar file without editing
    the table.

    This module ships no target table of its own: which folders, separations
    and system velocities exist is a property of a campaign, not of the
    retrieval method.  Define the table in the project's ``run_*.py``.

    Example
    -------
    >>> TARGETS = {'1b': dict(folder_name='WISPIT_1b', day='20260907_equa',
    ...                       model_name='1650-4.0--0.4', Sep_p=25, v_sys=19,
    ...                       vsini=20, prep_T=1650, prep_G=4.0, prep_Z=-0.4,
    ...                       star_dat='WISPIT_1b/.../upload_star.dat')}
    >>> r = build_retriever(dict(TARGETS['1b'], path_root='/path/to/project'))
    """
    cfg = dict(config)
    cfg.update(overrides)
    missing = [k for k in ('path_root', 'folder_name', 'day', 'model_name',
                           'star_dat') if k not in cfg]
    if missing:
        raise ValueError('target config is missing %s' % ', '.join(missing))
    return SpectraRetriever(**cfg)
