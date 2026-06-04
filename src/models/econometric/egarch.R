install.packages(c("data.table", "xts", "rugarch", "fGarch"), dependencies = TRUE)
suppressPackageStartupMessages({
  library(data.table)
  library(xts)
  library(rugarch)
  library(fGarch)   
})


VERSIONS <- c("EGARCH", "IV", "RV", "IV_RV")
#VERSIONS <- c("EGARCH", "IV", "RV", "IV_RV")


DATA_PATH <- file.path("data", "RV_IV_combo_1.csv")
OUT_DIR   <- file.path("src", "predictions")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

DATE_COL   <- "date"
TARGET_COL <- "log_return"

WINDOW_SIZE <- 2000L

# Out-of-sample start date
OOS_START_DATE <- as.Date("2016-02-05")

QUANTILES <- c(
  0.002, 0.004, 0.005, 0.006, 0.008, 0.010,
  0.015, 0.020, 0.025, 0.030, 0.040, 0.050,
  0.950, 0.960, 0.970, 0.975, 0.980, 0.985,
  0.990, 0.992, 0.994, 0.995, 0.996, 0.998
)

ES_LEVELS <- c(0.01, 0.025, 0.05, 0.95, 0.975, 0.99)
# Small step for monotonicity fix
EPS_MONO <- 1e-4


# Enforce nondecreasing quantiles across QUANTILES for each row
enforce_monotone_row <- function(qvec, eps = 1e-4) {
  out <- qvec
  for (j in 2:length(out)) {
    if (is.na(out[j - 1]) || is.na(out[j])) next
    if (out[j] < out[j - 1]) out[j] <- out[j - 1] + eps
  }
  out
}

# Expected Shortfall under standardized skewed Student-t 
es_sstd <- function(mu, sigma, nu, skew, alpha) {
  integrand <- function(y) y * dsstd(y, mean = 0, sd = 1, nu = nu, xi = skew)

  q_alpha <- qsstd(alpha, mean = 0, sd = 1, nu = nu, xi = skew)

  if (alpha < 0.5) {
    tail_integral <- integrate(integrand, lower = -Inf, upper = q_alpha,
                               rel.tol = 1e-4, subdivisions = 2000)$value
    es_std <- tail_integral / alpha
  } else {
    tail_integral <- integrate(integrand, lower = q_alpha, upper = Inf,
                               rel.tol = 1e-4, subdivisions = 2000)$value
    es_std <- tail_integral / (1 - alpha)
  }

  mu + sigma * es_std
}

# Build regressors for the 4 variants 
get_xreg_cols <- function(version) {
  v <- toupper(gsub("-", "_", trimws(version)))
  if (v == "NONE" || v == "EGARCH") return(character(0))
  if (v == "IV" || v == "EGARCH_IV") return(c("IV_D1", "IV_W1", "IV_M1"))
  if (v == "RV" || v == "EGARCH_RV") return(c("RV", "RV_W", "RV_M"))
  if (v == "IV_RV" || v == "EGARCH_IV_RV") return(c("IV_D1","IV_W1","IV_M1", "RV","RV_W","RV_M"))
  stop("Unknown version: ", version)
}

# Create EGARCH spec (with or without variance regressors)
make_spec <- function(xreg_dim) {
  ugarchspec(
    variance.model = list(
      model = "eGARCH",
      garchOrder = c(1, 1),
      external.regressors = if (xreg_dim > 0) matrix(0, nrow = 1, ncol = xreg_dim) else NULL
    ),
    mean.model = list(
      armaOrder = c(0, 0),
      include.mean = FALSE,
      arfima = FALSE
    ),
    distribution.model = "sstd"
  )
}

safe_fit_with_xreg <- function(y_train, X_train) {
  spec_x <- ugarchspec(
    variance.model = list(
      model = "eGARCH",
      garchOrder = c(1, 1),
      external.regressors = as.matrix(X_train)
    ),
    mean.model = list(
      armaOrder = c(0, 0),
      include.mean = FALSE,
      arfima = FALSE
    ),
    distribution.model = "sstd"
  )
  fit <- try(ugarchfit(spec = spec_x, data = y_train, solver = "hybrid"), silent = TRUE)
  if (inherits(fit, "try-error")) return(NULL)
  fit
}

forecast_next <- function(fit, x_next) {
  if (is.null(x_next)) {
    fc <- ugarchforecast(fitORspec = fit, n.ahead = 1)
  } else {
    fc <- ugarchforecast(
      fitORspec = fit,
      n.ahead = 1,
      external.forecasts = list(vregfor = as.matrix(x_next))
    )
  }
  fc
}

# ---------------- MAIN ----------------

df <- fread(DATA_PATH)
df[, (DATE_COL) := as.IDate(get(DATE_COL))]
setorderv(df, cols = DATE_COL)

# Keep only needed columns + drop NA rows
needed_cols <- c(DATE_COL, TARGET_COL,
                 "RV","RV_W","RV_M","IV_D1","IV_W1","IV_M1")
missing_cols <- setdiff(needed_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns in data: ", paste(missing_cols, collapse = ", "))
}

df <- df[, ..needed_cols]
df <- na.omit(df)

# xts target series
y_xts <- xts(df[[TARGET_COL]], order.by = as.Date(df[[DATE_COL]]))


for (ver in VERSIONS) {

  xcols <- get_xreg_cols(ver)
  cat("\n==============================\n")
  cat("Running EGARCH version:", ver, "\n")
  cat("Variance xreg cols:", if (length(xcols) == 0) "None" else paste(xcols, collapse = ", "), "\n")

  # xreg matrix aligned by date (same index as y_xts)
  X_all <- NULL
  if (length(xcols) > 0) {
    X_all <- as.matrix(df[, ..xcols])
    rownames(X_all) <- as.character(as.Date(df[[DATE_COL]]))
  }

  # OOS dates
  oos_idx <- index(y_xts) >= OOS_START_DATE
  dates_oos <- index(y_xts)[oos_idx]

  out_list <- vector("list", length(dates_oos))

  for (i in seq_along(dates_oos)) {
    test_date <- dates_oos[i]

    # training window ends strictly before test_date
    train_end_positions <- which(index(y_xts) < test_date)
    if (length(train_end_positions) < WINDOW_SIZE) next
    train_pos <- tail(train_end_positions, WINDOW_SIZE)

    y_train <- y_xts[train_pos]

    fit <- NULL
    x_next <- NULL

    if (is.null(X_all)) {
      # no xreg
      spec0 <- make_spec(0)
      fit <- try(ugarchfit(spec = spec0, data = y_train, solver = "hybrid"), silent = TRUE)
      if (inherits(fit, "try-error")) fit <- NULL
    } else {
      # xreg in variance: build X_train aligned to y_train
      idx_match <- match(as.character(index(y_train)), rownames(X_all))
      if (any(is.na(idx_match))) next
      X_train <- X_all[idx_match, , drop = FALSE]

      fit <- safe_fit_with_xreg(y_train, X_train)
      if (is.null(fit)) next

      # xreg known at time test_date (predict next day)
      x_next <- X_all[rownames(X_all) == as.character(test_date), , drop = FALSE]
      if (nrow(x_next) == 0) next
    }

    # forecast
    fc <- forecast_next(fit, x_next)

    mu_hat <- 0.0
    sigma_hat <- as.numeric(sigma(fc))

    cf <- coef(fit)
    skew_hat <- unname(cf["skew"])
    nu_hat   <- unname(cf["shape"])

    out_list[[i]] <- data.table(
      Date  = as.IDate(test_date),
      Mu    = mu_hat,
      Sigma = sigma_hat,
      Skew  = skew_hat,
      Nu    = nu_hat
    )

    if (i %% 20 == 0) cat("..", i, "/", length(dates_oos), "done\n")
  }

  oos <- rbindlist(out_list, fill = TRUE)
  if (nrow(oos) == 0) {
    cat("No successful fits for version", ver, "\n")
    next
  }

  # Merge with realized returns (truth) for the same Date
  truth <- data.table(Date = as.IDate(index(y_xts)), TrueY = as.numeric(y_xts))
  oos <- merge(oos, truth, by = "Date", all.x = TRUE)

  # Quantiles
  for (q in QUANTILES) {
    oos[[sprintf("Quantile_%0.3f", q)]] <-
      oos$Mu + oos$Sigma * qsstd(q, mean = 0, sd = 1, nu = oos$Nu, xi = oos$Skew)
  }

  # Monotonicity fix (row-wise)
  qcols <- sprintf("Quantile_%0.3f", QUANTILES)
  qmat <- as.matrix(oos[, ..qcols])
  qmat_fixed <- t(apply(qmat, 1, enforce_monotone_row, eps = EPS_MONO))
  oos[, (qcols) := as.data.table(qmat_fixed)]

  # ES via numerical integration on standardized sstd
  for (a in ES_LEVELS) {
    oos[[sprintf("ES_%0.3f", a)]] <- mapply(
      es_sstd, oos$Mu, oos$Sigma, oos$Nu, oos$Skew,
      MoreArgs = list(alpha = a)
    )
  }

  # Arrange output columns
  cols_front <- c("Date", "TrueY")
  qcols <- sprintf("Quantile_%0.3f", QUANTILES)
  es_cols <- sprintf("ES_%0.3f", ES_LEVELS)
  
  out_dt <- oos[, c(cols_front, qcols, es_cols), with = FALSE]

  out_dt[, VERSION := ver]

  setcolorder(out_dt, c("Date", "TrueY", "VERSION", qcols, es_cols))

  setnames(out_dt, "Date", "date")

  out_path <- file.path(OUT_DIR, sprintf("egarch_vol_window2000_%s.csv", ver))
  fwrite(out_dt, out_path)
  cat("Saved:", out_path, "\n")
}