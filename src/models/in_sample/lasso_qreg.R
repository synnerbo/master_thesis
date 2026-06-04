suppressPackageStartupMessages({
  library(data.table)
  library(quantreg)
  library(rqPen)
})

set.seed(42)


# CONFIG

DATA_PATH <- "data/RV_IV_combo_1.csv"

DATE_COL   <- "date"
TARGET_COL <- "log_return"

INSAMPLE_END_DATE <- as.Date("2016-02-04")

CANDIDATE_VARS <- c(
  "IV_D1", "IV_W1", "IV_M1",
  "RV", "RV_W", "RV_M",
  "PV", "NV", 
  "SJ", "CQ",
  "JC", "CC"
)

TAUS_LEFT  <- c(0.01, 0.025, 0.05)
TAUS_RIGHT <- c(0.95, 0.975, 0.99)

N_FOLDS <- 5
N_BOOT  <- 1000

OUT_DIR <- "src/testing/lasso_qr_screening_allvars_1000bs_rep"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)


# LOAD & PREPARE DATA

dt <- fread(DATA_PATH)
dt[[DATE_COL]]   <- as.Date(dt[[DATE_COL]])
dt[[TARGET_COL]] <- as.numeric(dt[[TARGET_COL]])
for (v in CANDIDATE_VARS) dt[[v]] <- as.numeric(dt[[v]])
setorderv(dt, DATE_COL)

dt_in     <- dt[get(DATE_COL) <= INSAMPLE_END_DATE]
screen_dt <- dt_in[, c(DATE_COL, TARGET_COL, CANDIDATE_VARS), with = FALSE]
screen_dt <- na.omit(screen_dt)

cat("Rows used for screening:", nrow(screen_dt), "\n")
cat("Date range:",
    as.character(min(screen_dt[[DATE_COL]])), "to",
    as.character(max(screen_dt[[DATE_COL]])), "\n\n")


# STANDARDIZE X

X_raw    <- as.matrix(screen_dt[, ..CANDIDATE_VARS])
y        <- screen_dt[[TARGET_COL]]
X_scaled <- scale(X_raw)
colnames(X_scaled) <- CANDIDATE_VARS

x_center <- attr(X_scaled, "scaled:center")
x_scale  <- attr(X_scaled, "scaled:scale")

scaling_dt <- data.table(
  variable = CANDIDATE_VARS,
  center   = as.numeric(x_center),
  scale    = as.numeric(x_scale)
)
fwrite(scaling_dt, file.path(OUT_DIR, "scaling_parameters.csv"))
cat("Scaling parameters saved.\n\n")


# CORRELATION CHECK — inspect before running LASSO

cor_mat <- round(cor(X_scaled), 2)
cat("Pairwise correlations among candidate variables:\n")
print(cor_mat)

# Print only the high-correlation pairs (abs(r) > 0.7, excluding diagonal)
cat("\nHigh-correlation pairs (|r| > 0.70):\n")
high_cor <- which(abs(cor_mat) > 0.70 & upper.tri(cor_mat), arr.ind = TRUE)
if (nrow(high_cor) == 0) {
  cat("  None found.\n\n")
} else {
  for (k in seq_len(nrow(high_cor))) {
    r <- cor_mat[high_cor[k, 1], high_cor[k, 2]]
    cat(sprintf("  %-8s — %-8s : r = %+.2f\n",
                rownames(cor_mat)[high_cor[k, 1]],
                colnames(cor_mat)[high_cor[k, 2]], r))
  }
  cat("\n")
}


# HELPER: COEFFICIENT EXTRACTION

extract_lasso_selected <- function(cvfit, var_names, tol = 1e-8) {

  # best lambda 
  if (!is.null(cvfit$cverr)) {
    cverr_vec <- as.numeric(cvfit$cverr)
  } else if (!is.null(cvfit$cv)) {
    cverr_vec <- rowMeans(as.matrix(cvfit$cv))
  } else if (!is.null(cvfit$cv.error)) {
    cverr_vec <- as.numeric(cvfit$cv.error)
  } else {
    stop("Cannot find CV errors in rq.pen.seq.cv object. Check rqPen version.")
  }

  best_idx    <- which.min(cverr_vec)
  lambda_grid <- if (!is.null(cvfit$fit$lambda)) cvfit$fit$lambda else cvfit$lambda
  best_lambda <- lambda_grid[best_idx]
  cat("    best lambda:", round(best_lambda, 6),
      "(index", best_idx, "of", length(lambda_grid), ")\n")

  cf_mat <- coef(cvfit)

  if (is.matrix(cf_mat)) {
    if (ncol(cf_mat) == 1) {
      cf_vec <- setNames(as.numeric(cf_mat), rownames(cf_mat))
    } else if (nrow(cf_mat) == 1) {
      cf_vec <- setNames(as.numeric(cf_mat), colnames(cf_mat))
    } else if (nrow(cf_mat) == length(lambda_grid)) {
      cf_vec <- cf_mat[best_idx, ]
    } else {
      cf_vec <- cf_mat[, best_idx]
    }
  } else {
    cf_vec <- setNames(as.numeric(cf_mat), names(cf_mat))
  }

  # strip intercept
  slope_idx <- !(tolower(names(cf_vec)) %in% c("intercept", "(intercept)"))
  cf_slopes <- cf_vec[slope_idx]

  if (length(cf_slopes) == length(var_names) &&
      !any(names(cf_slopes) %in% var_names)) {
    names(cf_slopes) <- var_names
  }

  cat("    Non-zero slopes:", sum(abs(cf_slopes) > tol), "of", length(cf_slopes), "\n")

  cat("    Coefficient extraction audit:\n")
  for (nm in names(cf_slopes)) {
    val      <- cf_slopes[nm]
    is_sel   <- abs(val) > tol
    cat(sprintf("      %-10s  coef = %+.8f  %s\n",
                nm, val, if (is_sel) "<-- SELECTED" else ""))
  }


  selected <- intersect(names(cf_slopes)[abs(cf_slopes) > tol], var_names)
  list(selected = selected, coef_all = cf_slopes, best_lambda = best_lambda)
}


# HELPER: LASSO CV FIT

run_lasso_cv <- function(X_sc, y_vec, tau, nfolds) {
  cat("  Fitting LASSO CV at tau =", tau, "\n")
  rq.pen.cv(
    x       = X_sc,
    y       = y_vec,
    tau     = tau,
    penalty = "LASSO",
    nfolds  = nfolds
  )
}


#  DROP NEAR-COLLINEAR COLUMNS (abs(r) > threshold)

remove_collinear <- function(X, threshold = 0.95) {
  if (ncol(X) <= 1) return(colnames(X))
  cm   <- cor(X)
  keep <- rep(TRUE, ncol(X))
  nms  <- colnames(X)
  for (i in seq_len(ncol(X) - 1)) {
    if (!keep[i]) next
    for (j in (i + 1):ncol(X)) {
      if (keep[j] && abs(cm[i, j]) > threshold) {
        keep[j] <- FALSE
        cat("    Dropping", nms[j], "( |cor| =",
            round(abs(cm[i, j]), 3), "with", nms[i], ")\n")
      }
    }
  }
  nms[keep]
}


# POST-LASSO INFERENCE via rq() + bootstrap SE

run_post_lasso_qr <- function(y_vec, X_sc, selected_vars, tau, n_boot) {

  if (length(selected_vars) == 0) {
    cat("  No variables selected — skipping.\n")
    return(NULL)
  }

  # Drop near-collinear predictors before fitting
  X_sub     <- X_sc[, selected_vars, drop = FALSE]
  kept_vars <- remove_collinear(X_sub, threshold = 0.95)

  if (length(kept_vars) < length(selected_vars)) {
    dropped <- setdiff(selected_vars, kept_vars)
    cat("  Collinearity filter dropped:", paste(dropped, collapse = ", "), "\n")
    cat("  Kept:", paste(kept_vars, collapse = ", "), "\n")
  }

  if (length(kept_vars) == 0) {
    cat("  All variables removed by collinearity filter — skipping.\n")
    return(NULL)
  }

  df_post <- data.frame(y = y_vec, X_sc[, kept_vars, drop = FALSE])

  # Fit QR; fall back from br to fn (interior point) if singular
  fit <- tryCatch(
    rq(y ~ ., tau = tau, data = df_post, method = "br"),
    error = function(e) {
      cat("  br failed (", conditionMessage(e), ") retrying with fn.\n")
      tryCatch(
        rq(y ~ ., tau = tau, data = df_post, method = "fn"),
        error = function(e2) {
          cat("  fn also failed:", conditionMessage(e2), "\n")
          NULL
        }
      )
    }
  )
  if (is.null(fit)) return(NULL)

  # xy pairs bootstrap — robust to heavy tails; falls back to rank SE
  summ <- tryCatch(
    summary(fit, se = "boot", R = n_boot, bsmethod = "xy"),
    error = function(e) {
      cat("  xy bootstrap failed — trying rank SE.\n")
      tryCatch(
        summary(fit, se = "rank"),
        error = function(e2) {
          cat("  rank SE failed:", conditionMessage(e2), "\n")
          NULL
        }
      )
    }
  )
  if (is.null(summ)) return(NULL)

  cm           <- as.data.frame(summ$coefficients)
  cm$term      <- rownames(cm)
  rownames(cm) <- NULL
  n_cols       <- ncol(cm) - 1L
  if (n_cols >= 4L) names(cm)[1:4] <- c("estimate", "std_error", "t_value", "p_value")
  else if (n_cols == 3L) names(cm)[1:3] <- c("estimate", "lower_ci", "upper_ci")

  coef_dt       <- as.data.table(cm)
  coef_dt[, tau := tau]

  list(fit = fit, summary = summ, coef_table = coef_dt)
}


# MAIN SCREENING FUNCTION

screen_tail <- function(X_sc, y_vec, taus, var_names, tail_name,
                        nfolds = 5, n_boot = 500) {
  cat("\n============================================================\n")
  cat("SCREENING:", tail_name, "tail | Taus:", paste(taus, collapse = ", "), "\n")
  cat("============================================================\n")

  tau_labels      <- sprintf("%.3f", taus)
  lasso_fits      <- setNames(vector("list", length(taus)), tau_labels)
  selected_by_tau <- setNames(vector("list", length(taus)), tau_labels)
  lambda_chosen   <- setNames(numeric(length(taus)), tau_labels)

  # ---- LASSO selection pass ----
  for (i in seq_along(taus)) {
    tau <- taus[i]; lbl <- tau_labels[i]
    cvfit                  <- run_lasso_cv(X_sc, y_vec, tau, nfolds)
    lasso_fits[[lbl]]      <- cvfit
    extr                   <- extract_lasso_selected(cvfit, var_names)
    selected_by_tau[[lbl]] <- extr$selected
    lambda_chosen[lbl]     <- extr$best_lambda
    cat("  Selected at tau =", tau, ":",
        if (length(extr$selected) == 0) "None"
        else paste(extr$selected, collapse = ", "), "\n")
  }

  union_sel <- sort(unique(unlist(selected_by_tau)))
  inter_sel <- Reduce(intersect, selected_by_tau)

  cat("\nUnion selected       :",
      if (length(union_sel) == 0) "None" else paste(union_sel, collapse = ", "), "\n")
  cat("Intersection selected:",
      if (length(inter_sel) == 0) "None" else paste(inter_sel, collapse = ", "), "\n")

  # Choose variable set for post-LASSO inference
  # Always use the intersection: variables the LASSO selected at every tau
  # This is the only set with consistent evidence across all quantile levels
  # If the intersection is empty we skip inference entirely rather than
  # silently falling back to a larger noisier set
  inference_vars  <- inter_sel
  inference_label <- "intersection"

  cat("\nUsing", inference_label, "for post-LASSO inference:",
      if (length(inference_vars) == 0) "None (skipping inference)"
      else paste(inference_vars, collapse = ", "), "\n")

  post_results <- setNames(vector("list", length(taus)), tau_labels)
  coef_tables  <- list()

  for (i in seq_along(taus)) {
    tau <- taus[i]; lbl <- tau_labels[i]
    cat("\nPost-LASSO QR inference, tau =", tau, "\n")
    post                <- run_post_lasso_qr(y_vec, X_sc, inference_vars, tau, n_boot)
    post_results[[lbl]] <- post
    if (!is.null(post)) {
      tbl         <- copy(post$coef_table)
      tbl[, tail := tail_name]
      coef_tables[[lbl]] <- tbl
    }
  }

  coef_dt <- if (length(coef_tables) > 0) rbindlist(coef_tables, fill = TRUE) else data.table()

  selected_dt <- rbindlist(lapply(tau_labels, function(lbl) {
    vars <- selected_by_tau[[lbl]]
    if (length(vars) == 0)
      data.table(tail = tail_name, tau = as.numeric(lbl), variable = NA_character_)
    else
      data.table(tail = tail_name, tau = as.numeric(lbl), variable = vars)
  }), fill = TRUE)

  lambda_dt <- data.table(
    tail   = tail_name,
    tau    = as.numeric(tau_labels),
    lambda = as.numeric(lambda_chosen)
  )

  list(
    lasso_fits      = lasso_fits,
    selected_by_tau = selected_by_tau,
    union_selected  = union_sel,
    inter_selected  = inter_sel,
    inference_vars  = inference_vars,
    inference_label = inference_label,
    post_results    = post_results,
    coef_dt         = coef_dt,
    selected_dt     = selected_dt,
    lambda_dt       = lambda_dt
  )
}


# RUN SCREENING

left_res  <- screen_tail(X_scaled, y, TAUS_LEFT,  CANDIDATE_VARS, "left",  N_FOLDS, N_BOOT)
right_res <- screen_tail(X_scaled, y, TAUS_RIGHT, CANDIDATE_VARS, "right", N_FOLDS, N_BOOT)


# SAVE OUTPUTS
fwrite(
  rbindlist(list(left_res$selected_dt, right_res$selected_dt), fill = TRUE),
  file.path(OUT_DIR, "selected_variables_by_tau.csv")
)

fwrite(
  rbindlist(list(
    data.table(tail = "left",
               variable = if (length(left_res$union_selected) > 0)
                            left_res$union_selected else NA_character_),
    data.table(tail = "right",
               variable = if (length(right_res$union_selected) > 0)
                            right_res$union_selected else NA_character_)
  ), fill = TRUE),
  file.path(OUT_DIR, "selected_variables_union.csv")
)

fwrite(
  rbindlist(list(
    data.table(tail = "left",
               variable = if (length(left_res$inter_selected) > 0)
                            left_res$inter_selected else NA_character_),
    data.table(tail = "right",
               variable = if (length(right_res$inter_selected) > 0)
                            right_res$inter_selected else NA_character_)
  ), fill = TRUE),
  file.path(OUT_DIR, "selected_variables_intersection.csv")
)

fwrite(
  rbindlist(list(left_res$lambda_dt, right_res$lambda_dt), fill = TRUE),
  file.path(OUT_DIR, "lambda_chosen.csv")
)

coef_all <- rbindlist(list(left_res$coef_dt, right_res$coef_dt), fill = TRUE)
if (nrow(coef_all) > 0) {
  fwrite(coef_all, file.path(OUT_DIR, "post_lasso_inference.csv"))
}

# FINAL SUMMARY
cat("\n\n================ FINAL SUMMARY ================\n")

cat("\nLEFT tail — union selected:\n  ")
cat(if (length(left_res$union_selected) == 0) "None"
    else paste(left_res$union_selected, collapse = ", "), "\n")
cat("LEFT tail — intersection selected:\n  ")
cat(if (length(left_res$inter_selected) == 0) "None"
    else paste(left_res$inter_selected, collapse = ", "), "\n")
cat("LEFT tail — used for inference (", left_res$inference_label, "):\n  ")
cat(paste(left_res$inference_vars, collapse = ", "), "\n")

cat("\nRIGHT tail — union selected:\n  ")
cat(if (length(right_res$union_selected) == 0) "None"
    else paste(right_res$union_selected, collapse = ", "), "\n")
cat("RIGHT tail — intersection selected:\n  ")
cat(if (length(right_res$inter_selected) == 0) "None"
    else paste(right_res$inter_selected, collapse = ", "), "\n")
cat("RIGHT tail — used for inference (", right_res$inference_label, "):\n  ")
cat(paste(right_res$inference_vars, collapse = ", "), "\n")

cat("\nVariables in BOTH tail unions:\n  ")
both <- intersect(left_res$union_selected, right_res$union_selected)
cat(if (length(both) == 0) "None" else paste(both, collapse = ", "), "\n")

cat("\nSaved to:", OUT_DIR, "\n")