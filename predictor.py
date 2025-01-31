import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import joblib

from google.colab import drive
drive.mount('/content/drive')

# Load datasets
test_data = pd.read_csv("./drive/MyDrive/test.csv")
train_data = pd.read_csv("./drive/MyDrive/train.csv")
validation_data = pd.read_csv("./drive/MyDrive/validation.csv")

# Inspect shapes
print(f"Test data shape: {test_data.shape}")
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {validation_data.shape}")

# Drop non-numeric columns
def drop_non_numeric(data):
    non_numeric_columns = [col for col in data.columns if data[col].dtype == 'object' or not np.issubdtype(data[col].dtype, np.number)]
    for col in non_numeric_columns:
        print(f"Non-numeric column detected: {col}. Dropping it.")
    data.drop(columns=non_numeric_columns, inplace=True)
    return data

# Apply to all datasets
train_data = drop_non_numeric(train_data)
validation_data = drop_non_numeric(validation_data)
test_data = drop_non_numeric(test_data)

# Ensure 'datetime' is excluded during preprocessing
for data in [train_data, validation_data, test_data]:
    if 'datetime' in data.columns:
        data.drop(columns=['datetime'], inplace=True)

# Ensure 'PM10' exists in data
def pm_ten_exists(data):
  if 'PM10' not in data.columns:
      print("Column 'PM10' is missing in test data. Adding it with default value 0.")
      data['PM10'] = 0
      return data

pm_ten_exists(test_data)
pm_ten_exists(validation_data)

# Handle missing values
train_data.fillna(train_data.mean(), inplace=True)
validation_data.fillna(validation_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Ensure column alignment
missing_in_validation = set(train_data.columns) - set(validation_data.columns)
missing_in_train = set(validation_data.columns) - set(train_data.columns)

for col in missing_in_validation:
    print(f"Column '{col}' missing in validation data. Adding it with default value 0.")
    validation_data[col] = 0

for col in missing_in_train:
    print(f"Column '{col}' missing in train data. Adding it with default value 0.")
    train_data[col] = 0

'''
# Define the outlier clipping function
def clip_outliers(data, columns, lower_quantile=0.01, upper_quantile=0.99):
    """
    Clips outliers in the specified columns of a DataFrame to the given quantile range.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to clip.
    - lower_quantile (float): Lower quantile threshold (default: 0.01).
    - upper_quantile (float): Upper quantile threshold (default: 0.99).

    Returns:
    - pd.DataFrame: DataFrame with clipped values.
    """
    for column in columns:
        lower_bound = data[column].quantile(lower_quantile)
        upper_bound = data[column].quantile(upper_quantile)
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        print(f"Clipped outliers in '{column}': [{lower_bound}, {upper_bound}]")
    return data

# Get numeric columns (excluding PM10)
numeric_columns = [col for col in train_data.columns if col != 'PM10' and np.issubdtype(train_data[col].dtype, np.number)]

# Apply outlier clipping to train, validation, and test datasets
train_data = clip_outliers(train_data, numeric_columns)
validation_data = clip_outliers(validation_data, numeric_columns)
test_data = clip_outliers(test_data, numeric_columns)
'''

# Reorder validation data columns to match train data
validation_data = validation_data[train_data.columns]

# Ensure column consistency
assert list(train_data.columns) == list(validation_data.columns), "Train and validation data must have the same structure."

# Clip extreme values in PM10
train_data['PM10'] = train_data['PM10'].clip(lower=0, upper=train_data['PM10'].quantile(0.99))

# Preprocess data (scaling between 0 and 1)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
validation_scaled = scaler.transform(validation_data)

# Save the scaler object after fitting
scaler_path = "./drive/MyDrive/sub/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as '{scaler_path}'")

# Function to create sequences
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :-1])  # All columns except the last (target)
        y.append(data[i + time_step, -1])    # Last column is the target (PM10)
    return np.array(X), np.array(y)

# Define time_step for LSTM
# time_steps = [5, 10, 15, 24, 36, 48]

validation_loss = 0.0
time_step = 36

'''
# Loop through time_step
for ts in time_steps:

  # Create training and validation sequences
  X_train, y_train = create_sequences(train_scaled, ts)
  X_validation, y_validation = create_sequences(validation_scaled, ts)

  early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Revert to the best weights
  )

  # Check for NaNs in sequences
  if np.isnan(X_train).any() or np.isnan(y_train).any():
      raise ValueError("NaN detected in X_train or y_train")
  if np.isnan(X_validation).any() or np.isnan(y_validation).any():
      raise ValueError("NaN detected in X_validation or y_validation")

  # Define the LSTM model
  model = Sequential([
      tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
      LSTM(50, return_sequences=True),
      LSTM(50, return_sequences=False),
      Dense(25, activation='relu'),
      Dense(1)  # Output layer for regression
  ])

  # Compile the model
  model.compile(optimizer='adam', loss='mean_absolute_error')

  # Train the model
  print("Training the model...")

  # Include early_stopping in model.fit()
  history = model.fit(
      X_train, y_train,
      validation_data=(X_validation, y_validation),
      epochs=100,  # Maximum number of epochs
      batch_size=32,
      callbacks=[early_stopping]
  )

  # Evaluate the model
  print("Evaluating on validation data...")
  current_validation_loss = model.evaluate(X_validation, y_validation, verbose=1)
  if current_validation_loss < validation_loss or validation_loss == 0.0:
    validation_loss = current_validation_loss
    time_step = ts
    print(f"Validation loss is better this time: {current_validation_loss}\n")
  else:
    print(f"Validation loss is worse this time: {current_validation_loss}\n")

print(f"Best time step for our model is: {time_step}.\n")
print(f"Best validation loss for our model is: {validation_loss}.")
'''

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,#10,          # Stop after 5 epochs with no improvement
    restore_best_weights=True  # Revert to the best weights
)

# Create training and validation sequences
X_train, y_train = create_sequences(train_scaled, time_step)
X_validation, y_validation = create_sequences(validation_scaled, time_step)

# Inspect shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_validation shape: {X_validation.shape}\n")

# Check for NaNs in sequences
if np.isnan(X_train).any() or np.isnan(y_train).any():
    raise ValueError("NaN detected in X_train or y_train")
if np.isnan(X_validation).any() or np.isnan(y_validation).any():
    raise ValueError("NaN detected in X_validation or y_validation")

# Define the GRU model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    GRU(50, return_sequences=True),
    GRU(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

'''
# Modified GRU model architecture with additional layers and regularization
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),

    # First GRU layer with dropout
    GRU(100, return_sequences=True, dropout=0.3),

    # Second GRU layer with dropout
    GRU(50, return_sequences=True, dropout=0.3),

    # Third GRU layer with dropout
    GRU(30, return_sequences=False, dropout=0.3),

    # Dense layer with ReLU activation
    Dense(50, activation='relu'),

    # Output layer for regression
    Dense(1)
])
'''

'''
# Define the LSTM model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])
'''

'''
# Define the LSTM model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer for regression
])
'''

'''
# Define the LSTM model
model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
#optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compile the model
model.compile(optimizer, loss='mean_absolute_error')

# Train the model
print("Training the model...")

'''
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
'''

# Include early_stopping in model.fit()
history = model.fit(
    X_train, y_train,
    validation_data=(X_validation, y_validation),
    epochs=100,  # Maximum number of epochs
    batch_size=32,
    callbacks=[early_stopping]
)


# Loss Curve (Training vs Validation Loss) - track progress over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_validation)

# Rescale predictions and actuals back to original scale
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([np.zeros((y_pred.shape[0], train_data.shape[1] - 1)), y_pred], axis=1)
)[:, -1]
y_validation_rescaled = scaler.inverse_transform(
    np.concatenate([np.zeros((y_validation.shape[0], train_data.shape[1] - 1)), y_validation.reshape(-1, 1)], axis=1)
)[:, -1]

# Prediction vs Actuals (Scatter Plot):
plt.scatter(y_validation_rescaled, y_pred_rescaled)
plt.plot([y_validation_rescaled.min(), y_validation_rescaled.max()],
         [y_validation_rescaled.min(), y_validation_rescaled.max()],
         'k--', lw=2)
plt.title('Predicted vs Actual PM10')
plt.xlabel('Actual PM10')
plt.ylabel('Predicted PM10')
plt.show()

# Create a DataFrame for comparison
results = pd.DataFrame({"Predicted": y_pred_rescaled.flatten(), "Actual": y_validation_rescaled.flatten()})
print("\nPredictions vs Actuals:")
print(results.tail())

# Save the model
model.save("./drive/MyDrive/sub/gru_pm10_model.h5")
print("Model saved as './drive/MyDrive/sub/gru_pm10_model.h5'")

# Preprocess the test data
test_data = drop_non_numeric(test_data)

# Fill missing values in test data using the mean of the training data
test_data.fillna(train_data.mean(), inplace=True)

# Align test data columns with train data
missing_in_test = set(train_data.columns) - set(test_data.columns)
for col in missing_in_test:
    print(f"Column '{col}' missing in test data. Adding it with default value 0.")
    test_data[col] = 0

# Reorder test data columns to match train data
test_data = test_data[train_data.columns]

# Ensure column alignment
assert list(test_data.columns) == list(train_data.columns), "Test data must have the same structure as train data."

# Scale the test data using the same scaler instance
test_scaled = scaler.transform(test_data)  # Use the same scaler instance

# Create test sequences
X_test, _ = create_sequences(test_scaled, time_step)

# Make predictions for the test dataset
y_test_pred = model.predict(X_test)

# Rescale predictions back to the original scale
y_test_pred_rescaled = scaler.inverse_transform(
    np.concatenate([np.zeros((y_test_pred.shape[0], train_data.shape[1] - 1)), y_test_pred], axis=1)
)[:, -1]

# Load original test data to recover datetime column
original_test_data = pd.read_csv("./drive/MyDrive/test.csv")

# Ensure 'datetime' exists in the original test data
if 'datetime' not in original_test_data.columns:
    potential_time_column = [col for col in original_test_data.columns if 'time' in col.lower()]
    if potential_time_column:
        original_test_data.rename(columns={potential_time_column[0]: 'datetime'}, inplace=True)
    else:
        raise KeyError("No time-related column found. Please ensure the data has a 'datetime' column.")

# Convert datetime to datetime type
original_test_data['datetime'] = pd.to_datetime(original_test_data['datetime'])

# Add back the first time_step rows with NaN for PM10
missing_rows = original_test_data.iloc[:time_step, :].copy()
missing_rows = missing_rows[['datetime']]  # Only keep the datetime column
missing_rows['PM10'] = np.nan  # Placeholder for missing predictions

# Create the results DataFrame for rows with predictions
predicted_rows = original_test_data.iloc[time_step:, :].copy()
predicted_rows = predicted_rows[['datetime']]  # Only keep the datetime column
predicted_rows['PM10'] = y_test_pred_rescaled  # Add predicted PM10 values

# Combine missing rows and predicted rows
submission_data = pd.concat([missing_rows, predicted_rows], ignore_index=True)

# Replace NaN with 0 for PM10 if necessary
submission_data['PM10'].fillna(0, inplace=True)  # Replace NaN with 0
# submission_data['PM10'].fillna(submission_data['PM10'].mean(), inplace=True)


# Smooth predictions using a rolling mean
# submission_data['PM10'] = submission_data['PM10'].rolling(window=3, min_periods=1).mean()

# Final shape check
print("Final results shape:", submission_data.shape)

# Save the final submission file
submission_data.to_csv("./drive/MyDrive/sub/submission.csv", index=False)
print("Submission file 'submission.csv' created successfully.")