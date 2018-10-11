data {
  int nbin; /* Number of bins */
  int npl; /* Number of planets */

  vector[nbin] Vbin; /* Volume (d(ln(R)) d(ln(P))) for each bin. */

  vector[nbin] pl_wts[npl]; /* Integral of each planet's likelihood function over each bin. */
  real pl_extra_wts[npl]; /* Integral of each planet's likelihood function into the "extra" bin. */

  vector[nbin] summed_Pdet; /* Sum over all stars of the averaged Pdet in each bin. */

  vector[nbin] mu_log_contam; /* log(mean contamination density) estimated for each bin. */
  vector[nbin] sigma_log_contam; /* relative error on the contamination density estimate. */

  real Vextra; /* Volume of the extra catch-all bin */

  real epsilon; /* The fractional variance of the white noise added for
  stability (suggest 1e-4, for 1% of total sigma?)*/

  vector[2] bin_centers[nbin]; /* Bin centers in ln(P)-ln(R) */
}

transformed data {
  vector[nbin] log_pl_wts[npl];
  real log_pl_extra_wts[npl];

  for (i in 1:npl) {
    log_pl_wts[i] = log(pl_wts[i]);
    log_pl_extra_wts[i] = log(pl_extra_wts[i]);
  }
}

parameters {
  vector[nbin] log_nfg_unit; /* The standardized log of the foreground *density* in each bin. */
  vector[nbin] log_nbg_unit; /* A standardized background *density* in each bin. */
  real log_nextra; /* The density in the extra, catch-all bin. */

  real mu; /* The GP mean */
  real<lower=0> sigma; /* s.d. bin-by-bin. */
  real<lower=0> lambda; /* The GP correlation lengthscale. */
}

transformed parameters {
  vector[nbin] log_nfg; /* We transform this using the GP covariance matrix. */
  vector[nbin] log_nbg; /* We sample in N(0,1) variables for the bg, but transfrom to N(mu, sigma). */

  {
    matrix[nbin,nbin] cov;
    matrix[nbin,nbin] L; /* Cholesky covariance */
    vector[nbin] mu;

    cov = cov_exp_quad(bin_centers, sigma, lambda);

    for (i in 1:nbin) {
      cov[i,i] = cov[i,i] * (1 + epsilon);
    }

    L = cholesky_decompose(cov);

    log_nfg = mu + L*log_nfg_unit; /* Transforms from N(0,1) to N(mu, Sigma). */
  }

  log_nbg = mu_log_contam + sigma_log_contam*log_nbg_unit;
}

model {
  log_nfg_unit ~ normal(0,1); /* Combined with transformation above => log_nfg ~ N(mu, Sigma) GP. */
  log_nbg_unit ~ normal(0,1); /* Combined with transformation => log_nbg ~ N(mu, sigma). */

  target += 0.5*log_nextra; /* p(n_extra) ~ 1/sqrt(n_extra). */

  /* Flat prior on mu (it's well-constrained by the likelihood anyway). */
  sigma ~ lognormal(0, 1); /* Broad-ish prior on the bin-to-bin variation */
  lambda ~ lognormal(0, 1); /* Broad-ish prior on the length scale in ln(P)-ln(R) space */

  for (i in 1:npl) {
    real log_fg;
    real log_bg;

    log_fg = log_sum_exp(log_pl_wts + log_nfg);
    log_bg = log_sum_exp(log_pl_wts + log_nbg);

    log_extra = log_pl_extra_wts[i] + log_nextra;

    target += log_sum_exp(log_fg, log_sum_exp(log_bg, log_extra));
  }

  /* Calculating Normalization */
  {
    real Nex_fg;
    real Nex_extra;
    real Nex_bg;

    /* We sum the (detection fraction)*dN/dln(P)dln(R)*dln(P)*dln(R) for the foreground. */
    Nex_fg = sum(summed_Pdet * exp(log_nfg) * Vbin);

    /* For the extra, there is no selection function applied (we are just counting the total number in the extra bin) */
    Nex_extra = exp(log_nextra) * Vextra;

    /* Same for the background---no selection function is applied here. */
    Nex_bg = sum(exp(log_nbg) * Vbin);

    target += -Nex_fg -Nex_bg -Nex_extra;
  }
}
