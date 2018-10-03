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
  vector[nbin] log_nfg; /* The log of the foreground *density* in each bin. */
  vector[nbin] log_nbg_unit; /* A standardized background *density* in each bin. */
  real log_nextra; /* The density in the extra, catch-all bin. */
}

transformed parameters {
  vector[nbin] log_nbg; /* We sample in N(0,1) variables for the bg, but transfrom to N(mu, sigma). */

  log_nbg = mu_log_contam + sigma_log_contam*log_nbg_unit;
}

model {
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
