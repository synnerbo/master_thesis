suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
  library(reticulate)
  library(pbapply)
})


# ---------------- CONFIG ----------------
DATA_PATH <- "data/RV_IV_combo_1.csv"        
OUT_DIR   <- "src/predictions"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

DATE_COL   <- "date"
TARGET_COL <- "log_return" 


WINDOW_SIZE <- 2000L

# Start of evaluation sample 
OOS_START_DATE <- as.Date("2016-02-05")

VERSION <- "IV_SJ"   # e.g. "RV", "RV_CJ", "IV", "IV_RV", ...

ES_LEFT  <- c(0.01, 0.025, 0.05)
ES_RIGHT <- c(0.95, 0.975, 0.99)

# p = number of quantiles used per ES approximation.
p_es <- 5L

# Non-crossing epsilon
EPS <- 1e-4


fc <- import_from_path("feature_combination", path = "src/settings")
feature_list <- fc$get_feature_cols(VERSION)

cat("VERSION:", VERSION, "\n")
cat("Features:", paste(feature_list, collapse = ", "), "\n")


# ---------------- DATA LOAD ----------------
dt <- fread(DATA_PATH)

dt[[DATE_COL]]   <- as.Date(dt[[DATE_COL]])
dt[[TARGET_COL]] <- as.numeric(dt[[TARGET_COL]])

for (f in feature_list) dt[[f]] <- as.numeric(dt[[f]])

cat("Columns in dt:\n")
print(names(dt))
cat("nrow(dt) =", nrow(dt), "\n")
cat("DATE_COL =", DATE_COL, " TARGET_COL =", TARGET_COL, "\n")

setorderv(dt, DATE_COL)

# ---------------- Build tau grid needed for ES approximation ----------------
# We need the p_es quantiles used for each ES level.

build_tau_grid <- function(es_levels, p, side = c("left","right")) {
  side <- match.arg(side)
  taus <- c()
  for (es in es_levels) {
    if (side == "left") {
      new_q <- es - (es * (0:(p-1)) / p)
    } else {
      new_q <- es + ((1 - es) * (0:(p-1)) / p)
    }
    taus <- c(taus, new_q)
  }
  sort(unique(round(taus, 3)))
}

taus_left  <- build_tau_grid(ES_LEFT,  p_es, side = "left")
taus_right <- build_tau_grid(ES_RIGHT, p_es, side = "right")
quantile_levels <- sort(unique(c(taus_left, taus_right)))

cat("Quantile levels used (n=", length(quantile_levels), "):\n", sep = "")
cat(paste(quantile_levels, collapse = ", "), "\n\n")


# ---------------- Rolling forecast loop ----------------

idx_seq <- seq.int(WINDOW_SIZE + 1L, nrow(dt))

all_preds <- pbapply::pblapply(idx_seq, function(t_end) {

  t_start <- t_end - WINDOW_SIZE
  train_idx <- t_start:(t_end - 1L)

  train_y <- dt[[TARGET_COL]][train_idx]
  train_X <- as.data.frame(dt[train_idx, ..feature_list])
  pred_X  <- as.data.frame(dt[t_end, ..feature_list])

  pred_row <- list(
    date  = dt[[DATE_COL]][t_end],
    TrueY = dt[[TARGET_COL]][t_end],
    VERSION = VERSION
  )

  # Fit/predict each quantile
  for (tau in quantile_levels) {
    fit_tau <- rq(train_y ~ ., tau = tau, data = train_X, method = "br")
    qhat    <- as.numeric(predict(fit_tau, newdata = pred_X))
    pred_row[[sprintf("Quantile_%0.3f", tau)]] <- qhat
  }
  
  pred_row
})

pred_dt <- rbindlist(all_preds, fill = TRUE)
cat("\nBuilt quantile forecasts:", nrow(pred_dt), "rows\n")


# ---------------- Enforce monotonicity (no crossings) ----------------
qcols <- sprintf("Quantile_%0.3f", quantile_levels)

enforce_monotone <- function(xvec, eps = EPS, debug = FALSE) {
    adjusted <- FALSE
    for (k in 2:length(xvec)) {
        if (!is.na(xvec[k]) && !is.na(xvec[k-1]) && xvec[k] < xvec[k-1]) {
        xvec[k] <- xvec[k-1] + eps
        adjusted <- TRUE
        }
    }
    if (debug && adjusted) {
        cat("Monotonicity fix applied to one row\n")
    }
    xvec
}

pred_dt[, (qcols) := {
  fixed <- enforce_monotone(as.numeric(.SD), debug = FALSE)
  as.list(fixed)
}, .SDcols = qcols, by = seq_len(nrow(pred_dt))]


# ---------------- Compute ES from VaR ----------------
compute_ES_row <- function(qrow_named) {
  taus  <- as.numeric(sub("Quantile_", "", names(qrow_named)))
  qvals <- as.numeric(qrow_named)
  out <- list()

  for (alpha in ES_LEFT) {
    use_taus <- alpha - (alpha * (0:(p_es - 1)) / p_es)
    use_taus <- round(use_taus, 3)
    vals <- qvals[taus %in% use_taus]
    out[[sprintf("ES_%0.3f", alpha)]] <- mean(vals, na.rm = TRUE)
  }

  for (alpha in ES_RIGHT) {
    use_taus <- alpha + ((1 - alpha) * (0:(p_es - 1)) / p_es)
    use_taus <- round(use_taus, 3)
    vals <- qvals[taus %in% use_taus]
    out[[sprintf("ES_%0.3f", alpha)]] <- mean(vals, na.rm = TRUE)
  }

  out
}

es_list <- apply(pred_dt[, ..qcols], 1, function(qrow) compute_ES_row(qrow))
ES_cols_unique <- unique(unlist(lapply(es_list, names)))

for (col in ES_cols_unique) {
  pred_dt[[col]] <- vapply(es_list, function(x) x[[col]], numeric(1))
}

cat("ES columns added:", paste(ES_cols_unique, collapse = ", "), "\n")


# ---------------- Filter OOS period (optional) ----------------
final_dt <- pred_dt[date >= OOS_START_DATE]


# ---------------- Save ----------------
outfile <- file.path(OUT_DIR, sprintf("QR_%s_window%d.csv", VERSION, WINDOW_SIZE))
fwrite(final_dt, outfile)

cat("Done. Wrote:", outfile, "\n")
print(head(final_dt))