required_packages <- c("data.table", "sandwich", "lmtest", "np")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(data.table)
library(sandwich)
library(lmtest)
 
# Configuration 
PROJECT_ROOT <- "C:/tio4900_master_thesis"
 
PRED_DIR      <- file.path(PROJECT_ROOT, "src", "predictions", "final_aligned_main_results")

TAU_LEVELS <- c(0.010, 0.025, 0.050, 0.950, 0.975, 0.990)
 
# Bootstrap settings
N_BOOT <- 1000 
SEED   <- 42
 
# Loss Functions 
# Left-tail convention: VaR < 0, ES < 0
# For right tail, pass negated values and tau = 1 - alpha
fz0_loss <- function(r, VaR, ES, tau) {
  L    <- as.numeric(r < VaR)
  ES   <- pmin(ES, -1e-8)   # numerical guard
  -L * (VaR - r) / (tau * ES) + VaR / ES + log(-ES) - 1
}
 
al_loss <- function(r, VaR, ES, tau) {
  L  <- as.numeric(r < VaR)
  ES <- pmin(ES, -1e-8)
  -log((tau - 1) / ES) - (r - VaR) * (tau - L) / (tau * ES)
}
 
# Load Prediction File 
load_prediction_file <- function(filepath) {
  if (!file.exists(filepath)) stop("File not found: ", filepath)
  dt <- fread(filepath)
 
  # Accept both "date" and "Date"
  date_col <- names(dt)[tolower(names(dt)) == "date"]
  if (length(date_col) == 0) stop("Missing date column ('date' or 'Date') in ", basename(filepath))
  date_col <- date_col[1]
 
  if (!"TrueY" %in% names(dt)) stop("Missing 'TrueY' column in ", basename(filepath))
 
  setnames(dt, date_col, "date")
  dt[, date := as.Date(date)]
  setkey(dt, date)
  dt
}
 
 
# Helper: pull VaR and ES vectors for a given tau
get_var_es <- function(dt, tau) {
  q_col  <- sprintf("Quantile_%.3f", tau)
  es_col <- sprintf("ES_%.3f",       tau)
 
  if (!q_col  %in% names(dt)) stop("Missing column: ", q_col)
  if (!es_col %in% names(dt)) stop("Missing column: ", es_col)
 
  list(VaR = as.numeric(dt[[q_col]]),
       ES  = as.numeric(dt[[es_col]]))
}
 
# Stationary Bootstrap Block Length 
# input is the loss difference series, output is the selected block length for resampling
select_block_length <- function(x) {
  n <- length(x) 
  if (requireNamespace("np", quietly = TRUE)) {
    tryCatch({
      b <- np::b.star(x, round = TRUE)[1] 
      return(max(1L, as.integer(b)))
    }, error = function(e) NULL)
  }
  max(1L, as.integer(ceiling(n^(1/3)))) # since n is 1714, this is about 12
}
 
# Stationary Bootstrap DM Test 
# H0: E[loss_A] >= E[loss_B]  (A not better than B)
# One-sided reject => A has significantly lower loss than B
dm_bootstrap <- function(loss_a, loss_b, n_boot = 1000, seed = 42) {
  d <- loss_a - loss_b 
  n <- length(d) 
  set.seed(seed)
 
  b      <- select_block_length(d) 
  d_mean <- mean(d, na.rm = TRUE) 
 
  lm_fit <- lm(d ~ 1) 
  hac_se <- tryCatch(
    sqrt(as.numeric(sandwich::NeweyWest(lm_fit, lag = b, prewhite = FALSE)[1, 1])), # computes the HAC standard error of the mean loss difference using the Newey-West estimator 
    error = function(e) sd(d, na.rm = TRUE) / sqrt(n) # if Newey-West fails, fall back to the standard error assuming i.i.d. data
  )
  t_stat <- d_mean / hac_se # computes the DM test statistic
 
  # Stationary bootstrap resamples with geometric block lengths
  boot_stats <- vapply(seq_len(n_boot), function(.) {
    idx <- integer(n); pos <- 1L 
    while (pos <= n) {
      start <- sample.int(n, 1) 
      len   <- rgeom(1, 1 / b) + 1L 
      block <- ((start - 1L + seq_len(len) - 1L) %% n) + 1L 
      take  <- min(len, n - pos + 1L) 
      idx[pos:(pos + take - 1L)] <- block[seq_len(take)]
      pos <- pos + take
    }
    d_b  <- d[idx] 
    m_b  <- mean(d_b, na.rm = TRUE) 
    lm_b <- lm(d_b ~ 1) 
    se_b <- tryCatch(
      sqrt(as.numeric(sandwich::NeweyWest(lm_b, lag = b, prewhite = FALSE)[1, 1])), 
      error = function(e) sd(d_b, na.rm = TRUE) / sqrt(n)
    )
    (m_b - d_mean) / se_b   
  }, numeric(1))
 
  list(
    t_stat      = t_stat,
    mean_loss_a = mean(loss_a, na.rm = TRUE),
    mean_loss_b = mean(loss_b, na.rm = TRUE),
    mean_d      = d_mean,
    block_len   = b,
    n           = n,
    p_two_sided = mean(abs(boot_stats) >= abs(t_stat), na.rm = TRUE),
    p_one_sided = mean(boot_stats <= t_stat, na.rm = TRUE)
  )
}
 
# Run Pairwise Tests Across All Tau Levels 
run_pairwise_tests <- function(dtA, dtB,
                               modelA_name   = "Model A",
                               modelB_name   = "Model B",
                               use_bootstrap = TRUE,
                               loss_fn       = c("FZ0", "AL")) {
  loss_fn <- match.arg(loss_fn)
 
  # Merge dates so both files are aligned
  common_dates <- intersect(dtA$date, dtB$date)
  if (length(common_dates) == 0) stop("No overlapping dates between the two files.")
  dtA_sub <- dtA[date %in% common_dates]; setkey(dtA_sub, date)
  dtB_sub <- dtB[date %in% common_dates]; setkey(dtB_sub, date)
 
  cat(sprintf("\n--- %s vs %s | loss=%s | n_dates=%d ---\n",
              modelA_name, modelB_name, loss_fn, length(common_dates)))
 
  out <- vector("list", length(TAU_LEVELS))
 
  for (i in seq_along(TAU_LEVELS)) {
    tau <- TAU_LEVELS[i]
 
    fcA <- get_var_es(dtA_sub, tau)
    fcB <- get_var_es(dtB_sub, tau)
    r   <- as.numeric(dtA_sub$TrueY)   # returns are the same in both files
 
    # Right-tail: negate everything, use 1 - tau
    is_upper <- tau > 0.5
    tau_use  <- if (is_upper) 1 - tau else tau
    if (is_upper) {
      r <- -r; fcA$VaR <- -fcA$VaR; fcA$ES <- -fcA$ES
      fcB$VaR <- -fcB$VaR; fcB$ES <- -fcB$ES
    }
 
    # Drop rows where ES >= 0 (invalid for log(-ES))
    valid <- is.finite(r) & is.finite(fcA$VaR) & is.finite(fcA$ES) &
             is.finite(fcB$VaR) & is.finite(fcB$ES) &
             fcA$ES < 0 & fcB$ES < 0
    r       <- r[valid]
    fcA$VaR <- fcA$VaR[valid]; fcA$ES <- fcA$ES[valid]
    fcB$VaR <- fcB$VaR[valid]; fcB$ES <- fcB$ES[valid]
 
    n_used <- sum(valid)
    cat(sprintf("  tau=%.3f  n=%d  tail=%s\n", tau, n_used, if(is_upper) "right" else "left"))
 
    if (n_used < 30) { warning("Too few obs for tau=", tau, ", skipping."); next }
 
    loss_fn_call <- if (loss_fn == "FZ0") fz0_loss else al_loss
    lossA <- loss_fn_call(r, fcA$VaR, fcA$ES, tau_use)
    lossB <- loss_fn_call(r, fcB$VaR, fcB$ES, tau_use)
 
    if (use_bootstrap) {
      res <- dm_bootstrap(lossA, lossB, n_boot = N_BOOT, seed = SEED)
    } else {
      d      <- lossA - lossB
      lm_fit <- lm(d ~ 1) 
      ct     <- lmtest::coeftest(lm_fit, vcov. = sandwich::NeweyWest(lm_fit, prewhite = FALSE))
      res    <- list(
        t_stat      = ct[1, "t value"],
        mean_loss_a = mean(lossA, na.rm = TRUE),
        mean_loss_b = mean(lossB, na.rm = TRUE),
        mean_d      = mean(d, na.rm = TRUE),
        n           = n_used,
        p_two_sided = ct[1, "Pr(>|t|)"],
        p_one_sided = pt(ct[1, "t value"], df = n_used - 1)
      )
    }
 
    out[[i]] <- data.table(
      model_A       = modelA_name,
      model_B       = modelB_name,
      loss_fn       = loss_fn,
      tau           = tau,
      n_used        = res$n,
      mean_loss_A   = round(res$mean_loss_a, 6),
      mean_loss_B   = round(res$mean_loss_b, 6),
      mean_diff     = round(res$mean_d,      6),  # negative = A better
      t_stat        = round(res$t_stat,      4),
      p_two_sided   = round(res$p_two_sided, 4),
      p_one_sided   = round(res$p_one_sided, 4),
      sig_A_beats_B = ifelse(res$p_one_sided < 0.01, "***",
                      ifelse(res$p_one_sided < 0.05, "**",
                      ifelse(res$p_one_sided < 0.10, "*", "")))
    )
  }
 
  rbindlist(out[!sapply(out, is.null)])
}
 
# Main: run all model pairs 
model_pairs <- list(
  c("LSTM_IV",       "LSTM_RV"),
  c("LSTM_IV_CJ",    "LSTM_RV_CJ"),
  #c("LSTM_IV_CQ",    "LSTM_RV_CQ"),
  #c("LSTM_IV_SJ",    "LSTM_RV_SJ"),
  c("LSTM_IV_SV",    "LSTM_RV_SV"),
  c("lgbm_IV",       "lgbm_RV"),
  c("lgbm_IV_CJ",    "lgbm_RV_CJ"),
  #c("lgbm_IV_CQ",    "lgbm_RV_CQ"),
  #c("lgbm_IV_SJ",    "lgbm_RV_SJ"),
  c("lgbm_IV_SV",    "lgbm_RV_SV"),
  c("catboost_IV",       "catboost_RV"),
  c("catboost_IV_CJ",    "catboost_RV_CJ"),
  #c("catboost_IV_CQ",    "catboost_RV_CQ"),
  #c("catboost_IV_SJ",    "catboost_RV_SJ"),
  c("catboost_IV_SV",    "catboost_RV_SV"),
  c("DB_IV",       "DB_RV"),
  c("DB_IV_CJ",    "DB_RV_CJ"),
  #c("DB_IV_CQ",    "DB_RV_CQ"),
  #c("DB_IV_SJ",    "DB_RV_SJ"),
  c("DB_IV_SV",    "DB_RV_SV"),
  c("QR_IV",       "QR_RV"),
  c("QR_IV_CJ",    "QR_RV_CJ"),
  #c("QR_IV_CQ",    "QR_RV_CQ"),
  #c("QR_IV_SJ",    "QR_RV_SJ"),
  c("QR_IV_SV",    "QR_RV_SV"),
  c("EGARCH_IV",   "EGARCH_RV")
)

cat("\nRunning DM tests for all model pairs...\n")

all_results <- rbindlist(lapply(model_pairs, function(pair) {
  modelA_name <- pair[1]
  modelB_name <- pair[2]

  fileA <- file.path(PRED_DIR, paste0(modelA_name, ".csv"))
  fileB <- file.path(PRED_DIR, paste0(modelB_name, ".csv"))

  cat("\n====================================================\n")
  cat("Loading files:\n")
  cat("  A:", fileA, "\n")
  cat("  B:", fileB, "\n")

  if (!file.exists(fileA)) {
    warning("Skipping pair because file is missing: ", fileA)
    return(NULL)
  }
  if (!file.exists(fileB)) {
    warning("Skipping pair because file is missing: ", fileB)
    return(NULL)
  }

  dtA <- load_prediction_file(fileA)
  dtB <- load_prediction_file(fileB)

  cat(sprintf("  %s: %d rows\n  %s: %d rows\n",
              modelA_name, nrow(dtA), modelB_name, nrow(dtB)))

  results_fz <- run_pairwise_tests(
    dtA, dtB,
    modelA_name   = modelA_name,
    modelB_name   = modelB_name,
    use_bootstrap = TRUE,
    loss_fn       = "FZ0"
  )

  results_al <- run_pairwise_tests(
    dtA, dtB,
    modelA_name   = modelA_name,
    modelB_name   = modelB_name,
    use_bootstrap = TRUE,
    loss_fn       = "AL"
  )

  rbindlist(list(results_fz, results_al), fill = TRUE)
}), fill = TRUE)

# Print 
cat("\n\n====== All Diebold-Mariano Results ======\n")
cat("H0: A not better than B  |  sig_A_beats_B: * p<.10  ** p<.05  *** p<.01\n\n")
print(all_results)

# Save 
out_file <- file.path(PROJECT_ROOT, "src", "testing", "dm_all_pairs_bootstrap_wlstm.csv")
fwrite(all_results, out_file)
cat("\nSaved to:", out_file, "\n")


