#install.packages("data.table") 
#install.packages("esreg")
#install.packages("pbapply")
#install.packages("parallel")

suppressPackageStartupMessages({
    library(data.table)
    library(esreg)
    library(reticulate)
    library(pbapply)
    library(parallel)
})

# ---------------- CONFIG ----------------
DATA_PATH <- "data/RV_IV_combo_1.csv"
OUT_DIR   <- "src/predictions"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

DATE_COL   <- "date"
TARGET_COL <- "log_return"

WINDOW_SIZE <- 2000L

ES_LEVELS <- c(0.01, 0.025, 0.05, 0.95, 0.975, 0.99)

VERSIONS <- c(
  "IV_CJ", "IV_CQ", "IV_RV", "IV_SJ", "IV_SV", "IV",
  "RV_CJ", "RV_CQ", "RV_SJ", "RV_SV", "RV"
)

# parallel workers
NC <- min(10L, max(1L, parallel::detectCores() - 2L))

# small eps 
EPS <- 1e-4

# OOS filter
OOS_START_DATE <- as.Date("2016-02-05")

fc <- import_from_path("feature_combination", path = "src/settings")

feature_map <- setNames(vector("list", length(VERSIONS)), VERSIONS)
for (v in VERSIONS) {
  feature_map[[v]] <- fc$get_feature_cols(v)
}

all_feature_cols <- unique(unlist(feature_map, use.names = FALSE))

cat("Versions to run:\n")
print(VERSIONS)
cat("\nAll unique feature columns needed:\n")
print(all_feature_cols)
cat("\nWindow:", WINDOW_SIZE, " | Cores:", NC, "\n\n")

dt <- fread(DATA_PATH)
dt[[DATE_COL]]   <- as.Date(dt[[DATE_COL]])
dt[[TARGET_COL]] <- as.numeric(dt[[TARGET_COL]])

needed_cols <- unique(c(DATE_COL, TARGET_COL, all_feature_cols))
missing_cols <- setdiff(needed_cols, names(dt))
if (length(missing_cols) > 0) {
  stop("These required columns are missing from the data: ",
       paste(missing_cols, collapse = ", "))
}

for (f in all_feature_cols) {
  dt[[f]] <- as.numeric(dt[[f]])
}

setorderv(dt, DATE_COL)

# first OOS index 
oos_start_idx <- which(dt[[DATE_COL]] >= OOS_START_DATE)[1]

stopifnot(!is.na(oos_start_idx))                 
stopifnot(oos_start_idx > WINDOW_SIZE)           

cat("OOS starts at index:", oos_start_idx,
    " date:", as.character(dt[[DATE_COL]][oos_start_idx]), "\n")

cat("\nAlignment check (first 3 rows):\n")
print(dt[1:3, .(date = get(DATE_COL), y = get(TARGET_COL))])

cat("\nAlignment check around OOS start:\n")
print(dt[(oos_start_idx-2):(oos_start_idx+2), .(date = get(DATE_COL), y = get(TARGET_COL))])

# checks columns exist and we have enough data
stopifnot(all(c(DATE_COL, TARGET_COL) %in% names(dt)))
stopifnot(nrow(dt) > WINDOW_SIZE + 5)

# ---------------- Dimitriadis & Bayer: one-step forecast ----------------
forecast_DB_one <- function(y_win_plus1, x_win_plus1, alpha) {
    if (!("matrix" %in% class(x_win_plus1))) x_win_plus1 <- as.matrix(x_win_plus1)

    chng <- FALSE
    y <- y_win_plus1
    x <- x_win_plus1
    a <- alpha

    # right tail, convert to left tail by negating y and a, and then flip back the forecasts at the end
    if (a > 0.5) {
        y <- -y
        a <- 1 - a
        chng <- TRUE
    }
    
    # this is 2001 - 1 = 2000 window size
    win <- length(y) - 1L

    # fit esreg on the window (y[1:win], x[1:win, ]) and forecast for x[win+1, ]
    fit <- esreg::esreg(y[1:win] ~ x[1:win, , drop = FALSE], alpha = a)

    # predict using the covariates of the last point in the window (x[win+1, ]) to get qhat and ehat for that point
    x_last <- x[length(y), , drop = FALSE]
    qhat <- as.numeric(c(1, x_last) %*% fit$coefficients_q)
    ehat <- as.numeric(c(1, x_last) %*% fit$coefficients_e)

    # if we transformed for the right tail, flip back the forecasts
    if (chng) {
        qhat <- -qhat
        ehat <- -ehat
    }

    list(q = qhat, e = ehat)
}

# ---------------- Rolling engine for one alpha ----------------
roll_DB_alpha <- function(dt, y_col, x_cols, date_col, alpha, windowSize, nc, idx) {
    y <- dt[[y_col]]
    x <- as.matrix(dt[, ..x_cols])
    dates <- dt[[date_col]]

    # cluster
    cl <- parallel::makePSOCKcluster(nc)
    on.exit(parallel::stopCluster(cl), add = TRUE)

    # export
    parallel::clusterExport(
        cl,
        varlist = c("forecast_DB_one", "y", "x", "dates", "alpha", "windowSize"),
        envir = environment()
    )

    res_list <- pbapply::pblapply(idx, function(t_end) {
        t_start <- t_end - windowSize
        ind <- t_start:t_end

        out <- forecast_DB_one(
            y_win_plus1 = y[ind],
            x_win_plus1 = x[ind, , drop = FALSE],
            alpha = alpha
        )

        list(
            date = dates[t_end],
            TrueY = y[t_end],
            q = out$q,
            e = out$e
        )
    }, cl = cl)

    rbindlist(res_list)
}



# ---------------- OOS index sequence ----------------
idx_oos <- seq.int(oos_start_idx, nrow(dt))

cat("OOS idx_oos length:", length(idx_oos), "\n")
cat("OOS first date:", as.character(dt[[DATE_COL]][idx_oos[1]]),
    "| OOS last date:", as.character(dt[[DATE_COL]][idx_oos[length(idx_oos)]]), "\n\n")

# ---------------- Run one version ----------------
run_one_version <- function(VERSION, dt, feature_list, idx_oos) {
    cat("\n====================================================\n")
    cat("Running VERSION:", VERSION, "\n")
    cat("Features:", paste(feature_list, collapse = ", "), "\n")
    cat("====================================================\n")

    stopifnot(all(c(DATE_COL, TARGET_COL, feature_list) %in% names(dt)))

    base_out <- data.table(
        date    = dt[[DATE_COL]][idx_oos],
        TrueY   = dt[[TARGET_COL]][idx_oos],
        VERSION = VERSION
    )

    for (a in ES_LEVELS) {
        cat("Running DB esreg for VERSION =", VERSION, " alpha =", a, "\n")

        tmp <- roll_DB_alpha(
        dt = dt,
        y_col = TARGET_COL,
        x_cols = feature_list,
        date_col = DATE_COL,
        alpha = a,
        windowSize = WINDOW_SIZE,
        nc = NC,
        idx = idx_oos
        )

        qname <- sprintf("Quantile_%0.3f", a)
        ename <- sprintf("ES_%0.3f", a)

        setnames(tmp, c("q", "e"), c(qname, ename))

        stopifnot(identical(base_out$date, tmp$date))

        base_out[, (qname) := tmp[[qname]]]
        base_out[, (ename) := tmp[[ename]]]
    }

    # optional crossing check
    alphas_sorted <- sort(unique(ES_LEVELS))
    qcols <- sprintf("Quantile_%0.3f", alphas_sorted)
    qcols <- qcols[qcols %in% names(base_out)]

    cross_count <- base_out[, {
        q <- as.numeric(unlist(.SD, use.names = FALSE))
        any_cross <- any(diff(q) < 0, na.rm = TRUE)
        .(any_cross = any_cross)
    }, .SDcols = qcols][, sum(any_cross, na.rm = TRUE)]

    cat("Rows with quantile crossing in", VERSION, ":", cross_count, "\n")

    outfile <- file.path(OUT_DIR, sprintf("DB_%s.csv", VERSION))
    fwrite(base_out, outfile)

    cat("Saved:", outfile, "\n")
    print(head(base_out))

    invisible(base_out)
}

# ---------------- Main loop over all versions ----------------
all_results <- vector("list", length(VERSIONS))
names(all_results) <- VERSIONS

for (i in seq_along(VERSIONS)) {
    v <- VERSIONS[i]
    feature_list <- feature_map[[v]]

    all_results[[v]] <- tryCatch(
        run_one_version(
        VERSION = v,
        dt = dt,
        feature_list = feature_list,
        idx_oos = idx_oos
        ),
        error = function(e) {
        cat("\nERROR while running version:", v, "\n")
        cat(conditionMessage(e), "\n")
        NULL
        }
    )
}

cat("\n========================================\n")
cat("Finished all versions.\n")
cat("Files written to:", OUT_DIR, "\n")
cat("Successful versions:\n")
print(names(all_results)[!vapply(all_results, is.null, logical(1))])
cat("Failed versions:\n")
print(names(all_results)[vapply(all_results, is.null, logical(1))])
cat("========================================\n")