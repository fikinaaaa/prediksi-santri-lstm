import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title("Prediksi Jumlah Santri dengan LSTM")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file Excel (.xlsx) berisi data jumlah santri", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(df.head())

    # Preprocessing
    df['Putri'] = df['Putri'].astype(str).str.replace(',', '.', regex=False).astype(float)
    df['Total'] = df['Total'].astype(str).str.replace(',', '.', regex=False).astype(float)
    data = df[['Total']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Buat data untuk LSTM
    def create_dataset(dataset, time_step=3):
        X, y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:i + time_step, 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(data_scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Model LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], 1), return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=200, batch_size=1, verbose=0)

    # Prediksi
    train_predict = model.predict(X)
    train_predict = scaler.inverse_transform(train_predict)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Prediksi tahun berikutnya
    last_sequence = data_scaled[-3:].reshape((1, 3, 1))
    next_pred = scaler.inverse_transform(model.predict(last_sequence))
    pred_value = int(next_pred[0][0])
    st.success(f"Prediksi jumlah santri tahun berikutnya: {pred_value}")

    # Plot hasil
    fig, ax = plt.subplots()
    ax.plot(y_actual, label='Aktual')
    ax.plot(train_predict, label='Prediksi')
    ax.set_title("Prediksi Jumlah Santri")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Jumlah")
    ax.legend()
    st.pyplot(fig)

    # Evaluasi
    rmse = np.sqrt(mean_squared_error(y_actual, train_predict))
    mae = mean_absolute_error(y_actual, train_predict)
    mape = mean_absolute_percentage_error(y_actual, train_predict) * 100

    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**MAE**: {mae:.2f}")
    st.write(f"**MAPE**: {mape:.2f}%")
