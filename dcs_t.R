# =============================================================================
# Score-driven (DCS) location model with Student-t innovations
#
#   y_t      = mu_t + eps_t,           eps_t ~ t_nu(0, varsigma)
#   mu_{t+1} = omega + phi (mu_t - omega) + kappa u_t
#   u_t      = v_t / (1 + v_t^2 / (nu varsigma)),   v_t = y_t - mu_t
#
# u_t is (proportional to) the score of the Student-t log-density with respect
# to mu_t. It is bounded and redescending, so extreme observations have a
# vanishing effect on the filter (Harvey 2013; Creal, Koopman & Lucas 2013).
# =============================================================================

# Student-t log-density with location mu, scale parameter varsigma (variance-like)
dt_loc <- function(y, mu, varsigma, nu) {
  lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * nu * varsigma) -
    (nu + 1) / 2 * log1p((y - mu)^2 / (nu * varsigma))
}

# Score-based innovation
score_u <- function(v, varsigma, nu) v / (1 + v^2 / (nu * varsigma))

# Run the filter forward for theta = (omega, phi, kappa, varsigma, nu)
dcs_t_filter <- function(y, theta) {
  omega <- theta[1]; phi <- theta[2]; kappa <- theta[3]
  varsigma <- theta[4]; nu <- theta[5]
  n  <- length(y)
  mu <- numeric(n); u <- numeric(n)
  mu[1] <- omega
  for (t in seq_len(n)) {
    u[t] <- score_u(y[t] - mu[t], varsigma, nu)
    if (t < n) mu[t + 1] <- omega + phi * (mu[t] - omega) + kappa * u[t]
  }
  ll <- dt_loc(y, mu, varsigma, nu)
  list(mu = mu, u = u, v = y - mu, loglik_t = ll, loglik = sum(ll))
}

# Maximum likelihood with box constraints (nlminb, PORT routines)
dcs_t_fit <- function(y, start = c(omega = mean(y), phi = 0.9, kappa = 0.7,
                                   varsigma = var(diff(y)), nu = 5),
                      lower = c(-Inf, -0.999, -2, 1e-6, 2.1),
                      upper = c( Inf,  0.999,  2,  Inf, 200)) {
  nll <- function(theta) {
    ll <- dcs_t_filter(y, theta)$loglik
    if (!is.finite(ll)) 1e10 else -ll
  }
  opt <- nlminb(start, nll, lower = lower, upper = upper)
  theta <- setNames(opt$par, c("omega", "phi", "kappa", "varsigma", "nu"))
  list(theta = theta, loglik = -opt$objective,
       aic = 2 * length(theta) + 2 * opt$objective,
       convergence = opt$convergence, message = opt$message)
}

# Simulate a path from the model (useful to check the estimator recovers theta)
dcs_t_simulate <- function(n, theta) {
  omega <- theta[1]; phi <- theta[2]; kappa <- theta[3]
  varsigma <- theta[4]; nu <- theta[5]
  y <- mu <- numeric(n); mu[1] <- omega
  for (t in seq_len(n)) {
    y[t] <- mu[t] + sqrt(varsigma) * rt(1, df = nu)
    if (t < n) mu[t + 1] <- omega + phi * (mu[t] - omega) +
        kappa * score_u(y[t] - mu[t], varsigma, nu)
  }
  list(y = y, mu = mu)
}

# The likelihood is multimodal (omega is weakly identified when phi is close to
# one), so a single starting point can stop at a local optimum. Fit from many
# random starts and keep the best.
dcs_t_fit_multistart <- function(y, n_starts = 60, seed = 42, ...) {
  set.seed(seed)
  fits <- lapply(seq_len(n_starts), function(i) {
    start <- c(runif(1, min(y), quantile(y, 0.75)), runif(1, 0.5, 0.998),
               runif(1, -1.5, 1.9), exp(runif(1, log(0.5), log(50))),
               runif(1, 2.2, 30))
    tryCatch(dcs_t_fit(y, start, ...), error = function(e) NULL)
  })
  fits <- Filter(Negate(is.null), fits)
  aics <- vapply(fits, `[[`, numeric(1), "aic")
  best <- fits[[which.min(aics)]]
  best$all_aic <- sort(aics)
  best
}
