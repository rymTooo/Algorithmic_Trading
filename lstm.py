import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Sample DataFrame (Replace this with your actual data)
data = pd.read_csv("data/your_file.csv")
data.drop(index=range(10), columns=["Open", "High","Low"],axis=1, inplace=True)
df = pd.DataFrame(data)
df = df.reset_index(drop=True)
print(df)
goal = df["percentage_change"]
feature = df.drop(columns=["percentage_change"])

columns_to_scale = ["Close", "Volume", "ma_20", "ma_40", "ma_80", "MACD", "Signal Line"]
columns_to_divide = ["rsi", "%K", "%D", "DX", "ADX"]


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_feature = scaler.fit_transform(feature)
feature[columns_to_scale] = scaler.fit_transform(feature[columns_to_scale])

feature[columns_to_divide] = feature[columns_to_divide]/100

scaled_feature = feature.values

# print(goal)
# print(feature)

# Convert to sequences for LSTM
sequence_length = 14  # Number of previous timesteps

X = []
y = []
for i in range(sequence_length, len(goal)-1):
    X.append(scaled_feature[i - sequence_length:i, :])  # Previous data
    y.append(goal[i])  # Target is `percentage_change`

print(X[0])
print(y[0])

# print(X)
# print(y)

X, y = np.array(X), np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Predicting `percentage_change`
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict
predictions = model.predict(X_test)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
