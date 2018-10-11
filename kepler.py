#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io.ascii import read as read_table, write as write_table
try:
    import pystan
except ImportError as e:
    print('Failed to import `pystan`:', str(e))
    print('Failure ignored, but will crash once pystan is used.')


# constants & settings
YR = 365.24  # year in [days]
ER = 1  # radius in [earth radii]
year_strip_width = 30.  # width in [days] to remove around 1 year in period
bin_size = 2.  # fractional size of the period and radius bins
inverted_errors = 1.3  # relative error assumed for inverted catalog


def load_catalog(path='data/q1_q17_dr25_koi.csv'):
    data = read_table(path)
    # column names relevant to us, and their simpler counterparts
    catalog_names = ['koi_period', 'koi_period_err1', 'koi_period_err2',
                     'koi_prad', 'koi_prad_err1', 'koi_prad_err2']
    simple_names = ['period', 'period_err_high', 'period_err_low',
                    'radius', 'radius_err_high', 'radius_err_low']
    # apply some very basic filters
    mask = (data['koi_pdisposition'] == 'CANDIDATE')
    for key in catalog_names:
        mask &= np.isfinite(data[key])
    data = data[mask]
    # rename the columns to be a bit more strightforward, and only
    #   keep the relevant columns.
    for old, new in zip(catalog_names, simple_names):
        data[old].name = new
    data = data[simple_names]
    # turn the lower uncertainties into their absolute values
    data['period_err_low'] = np.abs(data['period_err_low'])
    data['radius_err_low'] = np.abs(data['radius_err_low'])
    # add eror bounds
    data['period_bound_low'] = data['period'] - data['period_err_low']
    data['period_bound_high'] = data['period'] + data['period_err_high']
    data['radius_bound_low'] = data['radius'] - data['radius_err_low']
    data['radius_bound_high'] = data['radius'] + data['radius_err_high']
    return data


def load_inverted(path='data/kplr_dr25_inv_tces.txt'):
    # read the table
    data = read_table(path)
    # column names relevant to us, and their simpler counterparts
    catalog_names = ['period', 'Rp']
    simple_names = ['period', 'radius']
    # apply some very basic filters
    mask = (data['Disp'] == 'PC')
    for key in catalog_names:
        mask &= np.isfinite(data[key])
    data = data[mask]
    # rename the columns to be a bit more strightforward, and only
    #   keep the relevant columns.
    for old, new in zip(catalog_names, simple_names):
        data[old].name = new
    data = data[simple_names]
    # add new columsn with artificial uncertainty estimates
    data['period_err_high'] = np.zeros(len(data))
    data['period_err_low'] = np.zeros(len(data))
    data['radius_err_high'] = (inverted_errors - 1) * data['radius']
    data['radius_err_low'] = (1 - 1 / inverted_errors) * data['radius']
    # add eror bounds
    data['period_bound_low'] = data['period'] - data['period_err_low']
    data['period_bound_high'] = data['period'] + data['period_err_high']
    data['radius_bound_low'] = data['radius'] - data['radius_err_low']
    data['radius_bound_high'] = data['radius'] + data['radius_err_high']
    return data


def filter_data(data):
    simple_names = ['period', 'period_err_high', 'period_err_low',
                    'radius', 'radius_err_high', 'radius_err_low']
    mask = np.full(len(data), True, dtype=np.bool)
    for key in simple_names:
        if 'err' in key:  # uncertainties may be zero
            mask &= data[key] >= 0
        else:  # all others should be strictly positive
            mask &= data[key] > 0
    if np.any(~mask):
        print('Removing {} out of {} events due to invalid parameters.'
              ''.format(np.count_nonzero(~mask), len(data)))
    return data[mask]


def sanity_checks(data):
    simple_names = ['period', 'period_err_high', 'period_err_low',
                    'radius', 'radius_err_high', 'radius_err_low']
    for key in simple_names:
        assert np.all(np.isfinite(data[key]))
    for key in simple_names:
        if 'err' in key:  # uncertainties may be zero
            assert np.count_nonzero(data[key] >= 0) == len(data)
        else:  # all others should be strictly positive
            assert np.count_nonzero(data[key] > 0) == len(data)
    # make sure the lower uncertainty has smaller magnitude than the median
    assert np.all(data['period'] > np.abs(data['period_err_low']))
    assert np.all(data['radius'] > np.abs(data['radius_err_low']))
    return


def plot(catalog, inverted, *, p_edges, r_edges,
         show_cat_err=True, show_inv_err=True):
    # some convenience values
    fmt = '.'
    edge_factor = 1.2
    p_min, p_max = min(p_edges), max(p_edges)
    r_min, r_max = min(r_edges), max(r_edges)
    earth_opts = dict(color='k', alpha=0.5, zorder=12)

    plt.figure(figsize=(16, 6))
    for idx, title, data, show_err in zip([0, 1],
                                          ['KOIs', 'Inverted'],
                                          [catalog, inverted],
                                          [show_cat_err, show_inv_err]):
        plt.subplot(121+idx)
        plt.title(title)
        if show_err:
            plt.errorbar(np.array(data['period'].data),
                         np.array(data['radius'].data), fmt=fmt,
                         yerr=[data['radius_err_high'].data,
                               data['radius_err_low'].data])
        else:
            plt.plot(np.array(data['period'].data),
                     np.array(data['radius'].data), fmt)
        # draw the bins
        for p_edge in p_edges:
            plt.plot([p_edge]*2, [r_min, r_max], 'r:')
        for r_edge in r_edges:
            plt.plot([p_min, p_max], [r_edge]*2, 'r:')
        # draw the earth location and the removed year strip
        plt.axvline(YR, **earth_opts)
        plt.axhline(ER, **earth_opts)
        plt.axvspan(YR - year_strip_width/2, YR + year_strip_width/2,
                    alpha=0.2, color='k')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(p_min/edge_factor, p_max*edge_factor)
        plt.ylim(r_min/edge_factor, r_max*edge_factor)
        plt.xlabel('Period [days]')
        plt.ylabel('Radius [R$_\\bigoplus$]')
    plt.tight_layout()
    plt.savefig('kepler_and_inv_P_vs_R.pdf')
    return


def fuzzy_hist(data, p_edges, r_edges, n_samples=10_000, log_normal=True,
               individual_histograms=False):
    # extract relevant columns for easy log conversion
    period = np.array(data['period'].data)
    period_bound_low = np.array(data['period_bound_low'].data)
    period_bound_high = np.array(data['period_bound_high'].data)
    radius = np.array(data['radius'].data)
    radius_bound_low = np.array(data['radius_bound_low'].data)
    radius_bound_high = np.array(data['radius_bound_high'].data)
    # convenience
    n_events = period.size
    n_p_bins = p_edges.size - 1
    n_r_bins = r_edges.size - 1
    if log_normal:
        # apply log to all data quantities
        p_edges = np.log(p_edges)
        r_edges = np.log(r_edges)

        period = np.log(period)
        period_bound_low = np.log(period_bound_low)
        period_bound_high = np.log(period_bound_high)
        radius = np.log(radius)
        radius_bound_low = np.log(radius_bound_low)
        radius_bound_high = np.log(radius_bound_high)
    # re-compute the uncertainty interval (necessary in log space)
    period_err_low = period - period_bound_low
    period_err_high = period_bound_high - period
    radius_err_low = radius - radius_bound_low
    radius_err_high = radius_bound_high - radius
    # rescale uncertainty intervals to multiples of sigma
    # c = 1 / 1.64485 for 90% intervals, c = 1. for 68% intervals
    c = 1.  # if 68% intervals
    # to draw samples, we draw samples from the unit gaussian
    #   it is rescaled on each side seperately for the uncertainty of each
    #   event seperately, and then shifted by the median of each event.
    period_samples = np.random.normal(size=(n_samples, n_events))
    period_samples = np.where(period_samples > 0,
                              period_samples * period_err_high * c,
                              period_samples * period_err_low * c)
    period_samples += period

    radius_samples = np.random.normal(size=(n_samples, n_events))
    radius_samples = np.where(radius_samples > 0,
                              radius_samples * radius_err_high * c,
                              radius_samples * radius_err_low * c)
    radius_samples += radius

    if individual_histograms:
        H = np.zeros((n_events, n_p_bins, n_r_bins))
        for event_idx in range(n_events):
            h = np.histogram2d(period_samples[:, event_idx],
                               radius_samples[:, event_idx],
                               bins=[p_edges, r_edges])[0]
            H[event_idx] = h / n_samples
        return H  # shape == (n_events, n_p_bins, n_r_bins)
    H = np.histogram2d(period_samples.flatten(), radius_samples.flatten(),
                       bins=[p_edges, r_edges])[0]
    H /= n_samples
    return H  # shape == (n_p_bins, n_r_bins)


catalog = load_catalog()
inverted = load_inverted()
catalog = filter_data(catalog)
inverted = filter_data(inverted)
sanity_checks(catalog)
sanity_checks(inverted)

# define bins via edges. they form a regular grid in log space
p_edges = bin_size**np.arange(-2, 2) * YR
r_edges = bin_size**np.arange(-1, 2) * ER

# also compute the centers of bin (in log)
p_center = np.exp(0.5 * (np.log(p_edges[1:]) + np.log(p_edges[:-1])))
r_center = np.exp(0.5 * (np.log(r_edges[1:]) + np.log(r_edges[:-1])))

# plot the data before further filtering and processing
plot(catalog, inverted, p_edges=p_edges, r_edges=r_edges)

# remove the strip around periods of ~1 year
catalog = catalog[~((catalog['period'] < YR + year_strip_width/2) &
                    (catalog['period'] > YR - year_strip_width/2))]
inverted = inverted[~((inverted['period'] < YR + year_strip_width/2) &
                      (inverted['period'] > YR - year_strip_width/2))]

print('Period bin edges:', p_edges)
print('Radius bin edges:', r_edges)

# histograms of median values
H_cat = np.histogram2d(catalog['period'], catalog['radius'],
                       bins=[p_edges, r_edges])[0]
H_inv = np.histogram2d(inverted['period'], inverted['radius'],
                       bins=[p_edges, r_edges])[0]
H_diff = H_cat - H_inv
print('Point CAT:\n', H_cat.T[::-1], 'total:', np.sum(H_cat))
print('Point INV:\n', H_inv.T[::-1], 'total:', np.sum(H_inv))
print('Point DIFF:\n', H_diff.T[::-1], 'total:', np.sum(H_diff))

# fuzzy histograms using error estimates (even if they are made-up for INV)
Hf_cat = fuzzy_hist(catalog, p_edges, r_edges)
Hf_inv = fuzzy_hist(inverted, p_edges, r_edges)
Hf_diff = Hf_cat - Hf_inv

print('Fuzzy CAT:\n', Hf_cat.T[::-1], 'total:', np.sum(Hf_cat))
print('Fuzzy INV:\n', Hf_inv.T[::-1], 'total:', np.sum(Hf_inv))
print('Fuzzy DIFF:\n', Hf_diff.T[::-1], 'total:', np.sum(Hf_diff))

# histograms on an event-by-event basis
Hf_cat_ind = fuzzy_hist(catalog, p_edges, r_edges, individual_histograms=True)
Hf_inv_ind = fuzzy_hist(inverted, p_edges, r_edges, individual_histograms=True)

## throw into pystan
#mixture_data = {'nbin': 6,
#                'npl': len(P_fg),
#                'Vbin': np.outer(np.log(p_edges), np.log(r_edges)).flatten(),
#                'pl_wts': (Hf_fg_ind - Hf_bg_ind).T,
#                'pl_extra_wts': np.zeros_like(P_fg),
#                'summed_Pdet': []nbin,
#                'mu_log_contam': []nbin,
#                'sigma_log_contam': []nbin,
#                'Vextra':,
#                'epsilon': 1e-4,
#                'bin_centers': [0.5*(p_edges[1:]+p_edges[:-1]),
#                                0.5*(r_edges[1:]+r_edges[:-1])]}
#stan_model = pystan.StanModel(file='BinnedContamMixture.stan')
#fit = stan_model.sampling(data=mixture_data, iter=1000, chains=4)
#fit.plot()
