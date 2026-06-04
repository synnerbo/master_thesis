suppressPackageStartupMessages({
  library(esback)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(knitr)
  library(kableExtra)
  library(scales)
})


VERSION     <- "IV"
WINDOW_SIZE <- 1500L

csv_path <- file.path("src", "predictions", "lgbm_predictions_IV_RV_EURUSD.csv")

alphas <- c(0.010, 0.025, 0.050, 0.950, 0.975, 0.990)


esr_version <- 1
sig_level   <- 0.05

cov_cfg <- list(sparsity = "nid", sigma_est = "ind", misspec = TRUE)

DATE_COL <- "Date"
TRUE_COL <- "TrueY"

q_col  <- function(a) sprintf("Quantile_%0.3f", a)
es_col <- function(a) sprintf("ES_%0.3f", a)


if (!file.exists(csv_path)) stop("File not found: ", normalizePath(csv_path, winslash="/", mustWork=FALSE))

df <- read_csv(csv_path, show_col_types = FALSE) %>%
  mutate(
    "{DATE_COL}" := as.Date(.data[[DATE_COL]]),
    "{TRUE_COL}" := as.numeric(.data[[TRUE_COL]]),
    across(matches("^Quantile_"), as.numeric),
    across(matches("^ES_"), as.numeric)
  ) %>%
  arrange(.data[[DATE_COL]])

# ESR helper 
run_one_alpha_tailaware <- function(alpha) {
  qc <- q_col(alpha)
  ec <- es_col(alpha)

  if (!(qc %in% names(df))) {
    return(tibble(alpha=alpha, tail=ifelse(alpha < 0.5, "left", "right"), n=0,
                  p_value=NA_real_, pass=NA, note=paste("Missing", qc)))
  }
  if (!(ec %in% names(df))) {
    return(tibble(alpha=alpha, tail=ifelse(alpha < 0.5, "left", "right"), n=0,
                  p_value=NA_real_, pass=NA, note=paste("Missing", ec)))
  }

  sub <- df %>%
    select(all_of(c(DATE_COL, TRUE_COL, qc, ec))) %>%
    filter(is.finite(.data[[TRUE_COL]]),
           is.finite(.data[[qc]]),
           is.finite(.data[[ec]]))

  if (nrow(sub) < 30) {
    return(tibble(alpha=alpha, tail=ifelse(alpha < 0.5, "left", "right"),
                  n=nrow(sub), p_value=NA_real_, pass=NA, note="Too few obs"))
  }

  is_upper <- alpha > 0.5

  # Right tail: flip sign so we can reuse left-tail ESR on -returns
  r_use <- if (is_upper) -sub[[TRUE_COL]] else sub[[TRUE_COL]]
  q_use <- if (is_upper) -sub[[qc]]       else sub[[qc]]
  e_use <- if (is_upper) -sub[[ec]]       else sub[[ec]]
  a_use <- if (is_upper) 1 - alpha        else alpha

  res <- tryCatch(
    esr_backtest(
      r = r_use,
      q = q_use,
      e = e_use,
      alpha = a_use,
      version = esr_version,
      cov_config = cov_cfg
    ),
    error = function(e) NULL
  )

  pval <- if (is.null(res)) NA_real_ else res$pvalue_twosided_asymptotic

  tibble(
    alpha = alpha,
    tail  = ifelse(is_upper, "right", "left"),
    n     = nrow(sub),
    p_value = pval,
    pass  = !is.na(pval) & pval >= sig_level,
    note  = ifelse(is.na(pval), "ESR error/NA pval", "")
  )
}

# run 
results_long <- bind_rows(lapply(alphas, run_one_alpha_tailaware)) %>%
  mutate(alpha_lab = sprintf("%s @ %.1f%%",
                             ifelse(tail=="right", "Right", "Left"),
                             100*alpha))

cat("\n# ESR strict backtest (version", esr_version, "), sig =", sig_level, "\n")
print(
  kable(
    results_long %>%
      transmute(
        Alpha = alpha_lab,
        N = n,
        `p-value` = round(p_value, 3),
        Result = ifelse(is.na(p_value), "—", ifelse(pass, "PASS", "FAIL")),
        Note = ifelse(note == "", "—", note)
      ),
    format = "pipe", align = "c"
  ) %>% kable_styling()
)
# summary 
results_wide_left <- results_long %>%
  filter(tail=="left") %>%
  transmute(`Left α` = percent(alpha, accuracy=0.1), p_value) %>%
  pivot_wider(names_from = `Left α`, values_from = p_value) %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), NA, round(.x, 3))))

results_wide_right <- results_long %>%
  filter(tail=="right") %>%
  transmute(`Right α` = percent(alpha, accuracy=0.1), p_value) %>%
  pivot_wider(names_from = `Right α`, values_from = p_value) %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), NA, round(.x, 3))))

cat("\n# ESR p-values (Left tail)\n")
print(kable(results_wide_left, format="pipe", align="c"))

cat("\n# ESR p-values (Right tail)\n")
print(kable(results_wide_right, format="pipe", align="c"))