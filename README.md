# 📈 Stock Price Prediction using LSTM (PyTorch)

This project implements a Long Short-Term Memory (LSTM) network using PyTorch to predict future Apple stock prices based on historical OHLC (Open, High, Low, Close) data.

---

## 🚀 Project Highlights

- 🧠 Built a **PyTorch-based LSTM** model to predict next-day stock prices
- 📊 Used **4-year Apple OHLC data** (~1600 records)
- 📉 Achieved **Train RMSE: ₹23.57** and **Test RMSE: ₹40.52** using normalized OHLC averages
- 🧪 Trained and evaluated on a **75/25 train-test split**
- 📈 Plotted predicted vs actual prices to visualize model performance

---

## 🛠️ Technologies Used

- Python 3.13
- PyTorch
- Pandas, NumPy
- scikit-learn
- Matplotlib

---

## 🧪 Model Architecture

- 1 LSTM layer with 32 hidden units
- 1 Dense layer with linear activation
- Loss Function: MSE
- Optimizer: Adam
- Trained for 5 epochs

---

## 📁 Dataset

- `apple_share_price.csv` containing historical stock data
- Used average of Open, High, Low, and Close as input sequence

---

## 📉 Model Output (Training Logs)
Epoch 1, Loss: 0.0818
Epoch 2, Loss: 0.0690
Epoch 3, Loss: 0.0587
Epoch 4, Loss: 0.0508
Epoch 5, Loss: 0.0451
Train RMSE: 23.57
Test RMSE: 40.52

