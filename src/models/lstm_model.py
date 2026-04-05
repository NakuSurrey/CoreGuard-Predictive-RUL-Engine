"""
LSTM Model for RUL Prediction.

This is the ADVANCED model. Unlike XGBoost (which sees one row at a time),
the LSTM reads a SEQUENCE of consecutive cycles and learns degradation
patterns over time.

LSTM stands for Long Short-Term Memory. It is a type of neural network
designed for ordered data (sequences). As it reads each cycle in the
sequence, it decides what to remember and what to forget. By the end
of the sequence, it has a compressed understanding of the trend.

Why LSTM is the advanced model:
    - Captures temporal patterns (sensor rising over 30 cycles)
    - Sees rate of change naturally (no need for manual lag features)
    - Can learn complex non-linear degradation curves
    - More impressive on CV — shows deep learning capability
"""

from typing import Tuple, Optional

import numpy as np
import joblib

from src.config import (
    LSTM_PARAMS,
    LSTM_MODEL_PATH,
    SCALER_PATH,
    LSTM_SEQUENCE_LENGTH,
    RANDOM_SEED,
)


def build_lstm_model(
    input_shape: Tuple[int, int],
) -> "tf.keras.Model":
    """
    Build the LSTM neural network architecture.

    Architecture (layer by layer, top to bottom):

        Input Shape: (sequence_length, num_features) = (30, 44)
            ↓
        LSTM Layer 1 (64 units, return_sequences=True)
            - Reads all 30 timesteps
            - Outputs a sequence (one hidden state per timestep)
            - 64 units means it creates 64 internal "memory channels"
            ↓
        Dropout (20%)
            - Randomly turns off 20% of connections during training
            - Forces the network to not rely on any single connection
            - Prevents overfitting
            ↓
        LSTM Layer 2 (32 units, return_sequences=False)
            - Reads the output sequence from Layer 1
            - return_sequences=False means it outputs only the LAST hidden state
            - This single output summarizes the entire 30-cycle sequence
            ↓
        Dropout (20%)
            ↓
        Dense Layer (1 unit, linear activation)
            - Takes the 32-number summary and produces 1 number: predicted RUL
            - Linear activation means no transformation — raw number output

    Args:
        input_shape: (sequence_length, num_features) — shape of one input sample.

    Returns:
        Compiled Keras model ready for training.
    """
    # importing tensorflow here (not at top of file) because tensorflow
    # takes several seconds to load. We only import it when we actually
    # need it, so other parts of the app stay fast.
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)

    model = tf.keras.Sequential([
        # First LSTM layer: reads the sequence, passes full sequence to next layer
        tf.keras.layers.LSTM(
            units=LSTM_PARAMS["lstm_units"],     # 64 memory channels
            return_sequences=True,                # output at every timestep
            input_shape=input_shape,              # (30, 44)
        ),
        tf.keras.layers.Dropout(LSTM_PARAMS["dropout_rate"]),  # 20% dropout

        # Second LSTM layer: reads the processed sequence, outputs single summary
        tf.keras.layers.LSTM(
            units=LSTM_PARAMS["lstm_units"] // 2,  # 32 memory channels
            return_sequences=False,                 # output only at last timestep
        ),
        tf.keras.layers.Dropout(LSTM_PARAMS["dropout_rate"]),  # 20% dropout

        # Output layer: single number = predicted RUL
        tf.keras.layers.Dense(1),
    ])

    # Compile: tell the model how to learn
    # optimizer=adam: the algorithm that adjusts weights during training.
    #     Adam adapts the learning rate for each weight individually.
    # loss=mse: Mean Squared Error — the model tries to minimize the
    #     average squared difference between predicted and actual RUL.
    # metrics=mae: Mean Absolute Error — easier to interpret than MSE.
    #     "On average, the prediction is off by X cycles."
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LSTM_PARAMS["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    print("[lstm] Model architecture built:")
    model.summary(print_fn=lambda x: print(f"  {x}"))

    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple["tf.keras.Model", dict]:
    """
    Train the LSTM model on sequence data.

    Steps:
        1. Build the model architecture
        2. Set up early stopping (stop if validation loss stops improving)
        3. Train for up to 50 epochs
        4. Save the best model to disk

    What is an epoch:
        One epoch = the model has seen every single training sample once.
        50 epochs = the model goes through all training data 50 times.
        Each time it sees the data, it gets slightly better (ideally).

    What is early stopping:
        If the validation loss does not improve for 10 consecutive epochs,
        training stops early. This prevents the model from memorizing
        the training data (overfitting) instead of learning real patterns.

    Args:
        X_train: Training sequences, shape (num_samples, 30, 44).
        y_train: Training RUL targets, shape (num_samples,).
        X_val: Validation sequences (optional).
        y_val: Validation RUL targets (optional).

    Returns:
        Tuple of:
            - Trained Keras model
            - Training history dict with loss values per epoch
    """
    import tensorflow as tf

    input_shape: Tuple[int, int] = (X_train.shape[1], X_train.shape[2])
    print(f"[lstm] Input shape: {input_shape} (timesteps, features)")
    print(f"[lstm] Training samples: {X_train.shape[0]}")

    # Build the model
    model = build_lstm_model(input_shape)

    # Set up callbacks
    callbacks = []

    # Early stopping: monitors validation loss, stops if no improvement for 10 epochs
    # restore_best_weights=True: after stopping, revert to the weights from
    # the epoch that had the BEST validation loss (not the last epoch)
    if X_val is not None:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stop)

    # Reduce learning rate if validation loss plateaus
    # If loss does not improve for 5 epochs, multiply learning rate by 0.5.
    # Smaller learning rate = smaller weight adjustments = finer tuning.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss" if X_val is not None else "loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    callbacks.append(reduce_lr)

    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        print(f"[lstm] Validation samples: {X_val.shape[0]}")

    # Train
    print(f"[lstm] Training started (max {LSTM_PARAMS['epochs']} epochs)...")

    history = model.fit(
        X_train, y_train,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    # Extract metrics from training history
    training_metrics: dict = {
        "final_train_loss": round(history.history["loss"][-1], 4),
        "final_train_mae": round(history.history["mae"][-1], 4),
        "epochs_completed": len(history.history["loss"]),
    }

    if validation_data is not None:
        training_metrics["final_val_loss"] = round(history.history["val_loss"][-1], 4)
        training_metrics["final_val_mae"] = round(history.history["val_mae"][-1], 4)

    print(f"[lstm] Training complete after {training_metrics['epochs_completed']} epochs")
    print(f"[lstm] Final train MAE: {training_metrics['final_train_mae']}")
    if "final_val_mae" in training_metrics:
        print(f"[lstm] Final val MAE: {training_metrics['final_val_mae']}")

    # Save the trained model
    LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(LSTM_MODEL_PATH))
    print(f"[lstm] Model saved to {LSTM_MODEL_PATH}")

    return model, training_metrics


def predict_lstm(
    model: "tf.keras.Model",
    X: np.ndarray,
) -> np.ndarray:
    """
    Make RUL predictions using a trained LSTM model.

    Args:
        model: Trained Keras LSTM model.
        X: Input sequences, shape (num_samples, sequence_length, num_features).

    Returns:
        Numpy array of predicted RUL values, shape (num_samples,).
    """
    predictions: np.ndarray = model.predict(X, verbose=0)

    # model.predict returns shape (num_samples, 1) — flatten to (num_samples,)
    predictions = predictions.flatten()

    # RUL cannot be negative
    predictions = np.clip(predictions, 0, None)

    return predictions


def load_lstm_model() -> "tf.keras.Model":
    """
    Load a previously saved LSTM model from disk.

    Returns:
        Trained Keras model.
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
    print(f"[lstm] Model loaded from {LSTM_MODEL_PATH}")
    return model
