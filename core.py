import pandas as pd
import numpy as np
import sympy
from scipy.integrate import simpson, trapezoid, quad
from scipy.stats import linregress
from sympy import Integral, oo, Symbol, Rational
import plotly.graph_objects as go


reflect_df = pd.read_csv('Si_SiO2(30nm).txt',
                         sep='\s+',
                         skiprows=[0, 1],
                         names=['energy', 'reflectivity', 'transmission'])


def kramers_kronig_relation(r_e: pd.DataFrame, exact=False) -> pd.DataFrame:
    r"""
    Calculation of the phase shift between the reflected and incident radiation from the given dependency of
    reflectivity on energy R(E).

    .. math::
        \psi(E_0) = \frac{E_0}{\pi} v.p.\int_{0}^{+\infty} \frac{ln(R(E))}{E^2 - E_0^2}\,dE

    v.p. - Cauchy principal value of the integral: https://en.wikipedia.org/wiki/Cauchy_principal_value.

    To calculate v.p. integral the range is divided into four ranges:
        1. [min(E), E_0 - dE];
        2. [E0 - dE, E0 + dE];
        3. [E_0 + dE, max(E)].
        4. [max(E), +infinity)
    min(E) and max(E) - minimum and maximum values of energy in the R(E). dE - step between energies. Integrals over the
    1st and 3rd ranges are calculated by numerical integration using `scipy.integrate.simpson`. Integration over the 2nd
    range is conducted analytically using `scipy.integrate.quad`. R(E) in this region represented as linear function
    R = a + b * E, determined from two points E_0 - dE and E_0 + dE.
    The high energy tail of R(E) in the range of [max(E), +infinity) is calculated on the bases of extrapolation of the
    R(E) by the function of a + b * E^c, where c ~ -4. Then it is integrated analytically using `scipy.integrate.quad`.

    :param r_e: DataFrame with columns 'energy' and 'reflectivity'
    :param exact: If True, integration over the 2nd range will be included
    :return: Dataframe with columns 'energy' and 'phase shift'
    """

    def psi_e0(e0, r, form):
        # exclude point e0 (singularity) and form two regions
        r_before_e0 = r[r['energy'] < e0]
        r_after_e0 = r[r['energy'] > e0]

        # integrate over regions, excluding vicinity of e0
        i_before = e0 / np.pi * simpson(x=r_before_e0['energy'], y=integrand(e0, r_before_e0))
        i_after = e0 / np.pi * simpson(x=r_after_e0['energy'], y=integrand(e0, r_after_e0))

        if form == 'exact':
            return i_before + integral_in_e0_vicinity(e0, r) + i_after

        return i_before + i_after

    def integrand(e0, r):
        return np.log(r['reflectivity']) / (np.power(r['energy'], 2) - np.power(e0, 2))

    def integral_in_e0_vicinity(e0, r):
        # points, nearest to the e0
        point_before = r[r['energy'] < e0].iloc[-1, :]
        point_after = r[r['energy'] > e0].iloc[0, :]
        points = pd.concat([point_before, point_after], axis=1).T

        # determine slope and intercept in y = a + x * b between two points, nearest to e0
        res = linregress(x=points['energy'], y=points['reflectivity'])
        f = lambda x: e0 / np.pi * np.log(res.intercept + res.slope * x) / (np.power(x, 2) - np.power(e0, 2))

        return quad(f, point_before['energy'], point_after['energy'], points=[e0])[0]

    phase_shifts = pd.DataFrame()

    # TODO: ???????????????????????????? ?? ?????????????????????? e0 ???????????????? ?????????? ??????
    for e0 in r_e['energy'].values[1:-1]:
        phase_shifts = pd.concat([phase_shifts,
                                  pd.DataFrame({'energy': [e0],
                                                'phase_shift_with': [psi_e0(e0, r_e, 'exact')],
                                                'phase_shift_without': [psi_e0(e0, r_e, '')]})
                                  ]).reset_index(drop=True)

    return phase_shifts


# d = kramers_kronig_relation(reflect_df)
#
# print(d)
#
# fig = go.Figure()
#
# fig.add_trace(go.Scatter(x=d['energy'], y=d['phase_shift_with']))
# fig.add_trace(go.Scatter(x=d['energy'], y=d['phase_shift_without']))
#
# fig.show()