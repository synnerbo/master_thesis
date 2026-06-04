import os
import gc
import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping

SRC_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_DIR))

from src.settings.feature_combination import get_feature_cols
from src.data_handling.lstm_data_processing import get_lstm_train_test_new

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True)
parser.add_argument("--trials", type=int, default=50)
parser.add_argument("--db", type=str, default=None)
args = parser.parse_args()

VERSION = args.version
N_TRIALS = args.trials

DATA_PATH = SRC_DIR / "data" / "RV_IV_extended_slope_curve.csv"
FEATURE_COLS = get_feature_cols(VERSION)

LOOKBACK_DAYS = 30
Y_SCALE = 100.0
MAX_EPOCHS = 50
PATIENCE = 5

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


class PinballLoss(Loss):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.q = tf.constant(quantiles, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        e = y_true - y_pred
        loss = tf.maximum(self.q * e, (self.q - 1.0) * e)
        return tf.reduce_mean(loss)


def build_lstm_qr(num_features, hidden_units, dropout, l2_reg, num_hidden_layers):
    seq_input = Input(shape=(LOOKBACK_DAYS, num_features), name="feature_sequence")
    x = LSTM(
        hidden_units,
        activation="tanh",
        kernel_regularizer=l2(l2_reg),
        name="lstm",
    )(seq_input)

    if dropout > 0:
        x = Dropout(dropout, name="dropout")(x)

    for i in range(num_hidden_layers):
        x = Dense(
            hidden_units,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            name=f"dense_{i}",
        )(x)
        if dropout > 0:
            x = Dropout(dropout, name=f"dropout_{i}")(x)

    q_out = Dense(N_QUANTILES, name="quantile_output")(x)
    return Model(seq_input, q_out, name="LSTM_QR")


def standardize_by_train(train_X, other_X, eps=1e-8):
    flat = train_X.reshape(-1, train_X.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)

    train_Xs = (train_X - mu) / sigma
    other_Xs = (other_X - mu) / sigma
    return train_Xs.astype(np.float32), other_Xs.astype(np.float32), mu, sigma


def avg_pinball_loss_np(y_true, y_pred, quantiles):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    q = np.asarray(quantiles).reshape(1, -1)
    e = y_true - y_pred
    loss = np.maximum(q * e, (q - 1.0) * e)
    return float(np.mean(loss))


def pinball_loss_per_quantile_np(y_true, y_pred, quantiles):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    q = np.asarray(quantiles).reshape(1, -1)
    e = y_true - y_pred
    loss = np.maximum(q * e, (q - 1.0) * e)
    return np.mean(loss, axis=0)


def objective(trial):
    hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 0, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    df_raw = pd.read_csv(DATA_PATH)

    data = get_lstm_train_test_new(
        df=df_raw,
        feature_cols=FEATURE_COLS,
        lookback_days=LOOKBACK_DAYS,
        date_col="date",
        target_col="log_return_unshifted",
    )

    print("Train shape:", data.train.X.shape, data.train.y.shape)
    print("Val shape  :", data.validation.X.shape, data.validation.y.shape)
    print("Test shape :", data.test.X.shape, data.test.y.shape)
    print("Train dates:", min(data.train.y_dates), "to", max(data.train.y_dates))
    print("Val dates  :", min(data.validation.y_dates), "to", max(data.validation.y_dates))
    print("Test dates :", min(data.test.y_dates), "to", max(data.test.y_dates))

    if data.validation.X.shape[0] == 0:
        raise ValueError("Validation set is empty. Check TRAIN_VALIDATION_SPLIT and VALIDATION_TEST_SPLIT.")

    train_Xs, val_Xs, _, _ = standardize_by_train(data.train.X, data.validation.X)

    y_train_s = (data.train.y * Y_SCALE).astype(np.float32)
    y_val_s = (data.validation.y * Y_SCALE).astype(np.float32)

    model = build_lstm_qr(
        num_features=data.train.X.shape[2],
        hidden_units=hidden_units,
        dropout=dropout,
        l2_reg=l2_reg,
        num_hidden_layers=num_hidden_layers,
    )

    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=PinballLoss(quantiles=QUANTILES),
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        train_Xs,
        y_train_s,
        validation_data=(val_Xs, y_val_s),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    trial.set_user_attr("epochs_trained", len(history.history["loss"]))
    trial.set_user_attr("best_keras_val_loss", float(min(history.history["val_loss"])))

    preds_val_s = model.predict(val_Xs, batch_size=batch_size, verbose=0)
    preds_val_s = np.maximum.accumulate(preds_val_s, axis=1)

    avg_loss = avg_pinball_loss_np(y_val_s, preds_val_s, QUANTILES)
    per_q = pinball_loss_per_quantile_np(y_val_s, preds_val_s, QUANTILES)

    for q, loss_q in zip(QUANTILES, per_q):
        trial.set_user_attr(f"pinball_{q:.3f}", float(loss_q))

    del model
    gc.collect()
    tf.keras.backend.clear_session()

    return avg_loss


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    OUT_DIR = HERE / "optuna" / "lstm"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.db is None:
        storage_url = f"sqlite:///{(OUT_DIR / 'optuna_lstm_qreg.db').as_posix()}"
    else:
        storage_url = f"sqlite:///{Path(args.db).as_posix()}"

    study = optuna.create_study(
        direction="minimize",
        study_name=f"lstm_qreg_tuning_{VERSION}",
        storage=storage_url,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("Best trial:")
    print("Value:", study.best_trial.value)
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save full trial history
    df_results = study.trials_dataframe()
    df_results.to_csv(OUT_DIR / f"LSTM_QREG_optuna_trials_{VERSION}.csv", index=False)

    # Save best parameters only
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df.to_csv(OUT_DIR / f"LSTM_QREG_EURUSD_hyperparams_{VERSION}.csv", index=False)

    # also save best params as JSON
    with open(OUT_DIR / f"LSTM_QREG_EURUSD_hyperparams_{VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("Done.")