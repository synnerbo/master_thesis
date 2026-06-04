from __future__ import annotations

import os
import gc
import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

SRC_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC_DIR))

from src.settings.feature_combination import get_feature_cols
from src.data_handling.lstm_data_processing import get_lstm_train_test_new, VALIDATION_TEST_SPLIT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, type=str)
    p.add_argument("--params", required=True, type=str, help="Path to best-params JSON file")
    return p.parse_args()


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


args = parse_args()
VERSION = args.version
PARAMS = load_params(args.params)

FEATURE_COLS = get_feature_cols(VERSION)
DATA_PATH = SRC_DIR / "data" / "RV_IV_extended_slope_curve.csv"


LOOKBACK_DAYS = 30
SUFFIX = "_EURUSD"
MODEL_NAME = f"LSTM_{VERSION}"

Y_SCALE = 100.0
TEST_BLOCK_DAYS = 30

# Read tuned hyperparameters from JSON
HIDDEN_UNITS = int(PARAMS["hidden_units"])
DROPOUT = float(PARAMS["dropout"])
L2_REG = float(PARAMS["l2_reg"])
NUM_HIDDEN_LAYERS = int(PARAMS["num_hidden_layers"])
LEARNING_RATE = float(PARAMS["learning_rate"])
WEIGHT_DECAY = float(PARAMS["weight_decay"])
BATCH_SIZE = int(PARAMS["batch_size"])


EPOCHS = 15

ES_QUANTILES = [0.01, 0.025, 0.05, 0.95, 0.975, 0.99]
P_ES = 5

quantiles = []
for es in ES_QUANTILES:
    if es < 0.5:
        new_q = [es - (es * i / P_ES) for i in range(P_ES)]
    else:
        new_q = [es + ((1 - es) * i / P_ES) for i in range(P_ES)]
    quantiles.extend(new_q)

QUANTILES = sorted(set(np.round(quantiles, 3)))
N_QUANTILES = len(QUANTILES)

print(f"VERSION: {VERSION}")
print(f"FEATURE_COLS: {FEATURE_COLS}")
print("Loaded hyperparameters:")
print(json.dumps(PARAMS, indent=2))
print(f"Quantile levels used (n={N_QUANTILES}): {QUANTILES}")


class PinballLoss(Loss):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.q = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        e = y_true - y_pred
        loss = tf.maximum(self.q * e, (self.q - 1.0) * e)
        return tf.reduce_mean(loss)


def build_lstm_qr(num_features: int) -> Model:
    seq_input = Input(shape=(LOOKBACK_DAYS, num_features), name="feature_sequence")

    x = LSTM(
        HIDDEN_UNITS,
        activation="tanh",
        kernel_regularizer=l2(L2_REG),
        name="lstm",
    )(seq_input)

    if DROPOUT > 0:
        x = Dropout(DROPOUT, name="dropout")(x)

    for i in range(NUM_HIDDEN_LAYERS):
        x = Dense(
            HIDDEN_UNITS,
            activation="relu",
            kernel_regularizer=l2(L2_REG),
            name=f"dense_{i}",
        )(x)
        if DROPOUT > 0:
            x = Dropout(DROPOUT, name=f"dropout_{i}")(x)

    q_out = Dense(N_QUANTILES, name="quantile_output")(x)
    return Model(seq_input, q_out, name="LSTM_QR")


def standardize_by_train(train_X: np.ndarray, test_X: np.ndarray, eps: float = 1e-8):
    flat = train_X.reshape(-1, train_X.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)

    train_Xs = (train_X - mu) / sigma
    test_Xs = (test_X - mu) / sigma
    return train_Xs.astype(np.float32), test_Xs.astype(np.float32), mu, sigma


if __name__ == "__main__":
    print(f"Expanding LSTM-QR: {VERSION}")

    df_raw = pd.read_csv(DATA_PATH)

    data = get_lstm_train_test_new(
        df=df_raw,
        feature_cols=FEATURE_COLS,
        lookback_days=LOOKBACK_DAYS,
        date_col="date",
        target_col="log_return_unshifted",
    )

    first_test = pd.to_datetime(VALIDATION_TEST_SPLIT)
    test_block_days = TEST_BLOCK_DAYS if TEST_BLOCK_DAYS else np.inf

    def end_date():
        return first_test + pd.DateOffset(days=test_block_days)

    all_preds = []
    dates = []
    true_y = []

    while True:
        test = data.get_test_set_for_date(prediction_date=first_test, to_date=end_date())
        if test.y.shape[0] == 0:
            break

        train = data.get_training_set_for_date(prediction_date=first_test)

        print("Initial expanding-train period:")
        print("Train start:", train.dates.min())
        print("Train end  :", train.dates.max())
        print("First test :", first_test)
        print("N train samples:", train.X.shape[0])

        train_Xs, test_Xs, _, _ = standardize_by_train(train.X, test.X)

        y_train_s = (train.y * Y_SCALE).astype(np.float32)

        print("Train:", train.dates.min(), "to", train.dates.max())
        print("Test :", test.dates.min(), "to", test.dates.max())

        model = build_lstm_qr(num_features=train.X.shape[2])
        model.compile(
            optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
            loss=PinballLoss(quantiles=QUANTILES),
        )

        model.fit(
            train_Xs,
            y_train_s,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        preds_s = model.predict(test_Xs, batch_size=BATCH_SIZE, verbose=0)
        preds = preds_s / Y_SCALE
        preds = np.maximum.accumulate(preds, axis=1)

        all_preds.append(preds)
        dates += list(test.dates)
        true_y += list(test.y)

        first_test = end_date() + pd.DateOffset(days=1)

        del model
        gc.collect()
        tf.keras.backend.clear_session()

    y_pred = np.vstack(all_preds)
    true_y = np.array(true_y)

    out = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "TrueY": true_y,
        "VERSION": VERSION,
    }).sort_values("Date").reset_index(drop=True)

    quant_arr = np.array(QUANTILES, dtype=float)

    for j, q in enumerate(quant_arr):
        out[f"Quantile_{q:.3f}"] = y_pred[:, j]

    qcols = [c for c in out.columns if c.startswith("Quantile_")]
    qcols = sorted(qcols, key=lambda c: float(c.split("_")[1]))
    out[qcols] = np.maximum.accumulate(out[qcols].to_numpy(), axis=1)

    for es in ES_QUANTILES:
        if es < 0.5:
            sub = [es - (es * i / P_ES) for i in range(P_ES)]
        else:
            sub = [es + ((1 - es) * i / P_ES) for i in range(P_ES)]

        sub = [f"{x:.3f}" for x in np.round(sub, 3)]
        sub_cols = [f"Quantile_{x}" for x in sub if f"Quantile_{x}" in out.columns]

        if not sub_cols:
            print(f"WARNING: No quantile columns found for ES alpha {es:.3f}. Skipping.")
            continue

        out[f"ES_{es:.3f}"] = out[sub_cols].mean(axis=1)

    quantile_cols = sorted(
        [c for c in out.columns if c.startswith("Quantile_")],
        key=lambda c: float(c.split("_")[1])
    )
    es_cols = sorted(
        [c for c in out.columns if c.startswith("ES_")],
        key=lambda c: float(c.split("_")[1])
    )

    out = out[["Date", "TrueY", "VERSION"] + quantile_cols + es_cols]

    OUT_DIR = SRC_DIR / "src" / "predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_DIR / f"{MODEL_NAME}.csv"
    out.to_csv(out_csv, index=False)

    print(f"Saved predictions to {out_csv}")