# Stock Price Prediction using LSTM

## Overview

This project focuses on predicting stock prices using a Long Short-Term Memory (LSTM) neural network. The model is trained on technical analysis data, specifically the Moving Average Convergence Divergence (MACD) and the Relative Strength Index (RSI). By leveraging these technical indicators, the model aims to capture the patterns and trends in stock price movements to make accurate predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stock price prediction is a challenging task due to the volatile nature of the financial markets. This project utilizes deep learning techniques, specifically LSTM networks, which are well-suited for time series forecasting. The model is trained on technical indicators MACD and RSI to predict future stock prices.

## Dataset

The dataset consists of historical stock price data, including:

- Open price
- Close price
- High price
- Low price
- Volume

The technical indicators, MACD and RSI, are calculated from the historical price data and used as features for training the LSTM model.

## Feature Engineering

Before training the model, we preprocess the data and calculate the following features:

- **MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator that shows the relationship between two moving averages of a stock’s price.
- **RSI (Relative Strength Index):** A momentum oscillator that measures the speed and change of price movements.

These indicators are normalized using MinMaxScaler to ensure that they are within the same scale, which is crucial for the LSTM model.

## Model Architecture

The LSTM model is designed to capture the temporal dependencies in the stock price data. The architecture includes:

- **Input Layer:** Accepts the MACD, RSI, and other relevant features.
- **LSTM Layers:** Two stacked LSTM layers to learn the sequence patterns in the data.
- **Dense Layer:** Fully connected layer to aggregate the learned features.
- **Output Layer:** Predicts the future stock price.

## Training

The model is trained using the following configurations:

- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam optimizer with a learning rate of 0.001
- **Batch Size:** 64
- **Epochs:** 50

The dataset is split into training and validation sets, with 80% of the data used for training and 20% for validation.

## Evaluation

The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The evaluation metrics are calculated on both the training and validation sets to assess the model's performance.

## Results

The LSTM model shows promising results in predicting stock prices based on the MACD and RSI indicators. The evaluation metrics indicate that the model is able to capture the underlying trends and make accurate predictions.

## Usage

To run this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Hieucaohd/LSTM-predict-stock-by-technical-analysis
   cd LSTM-predict-stock-by-technical-analysis

2. **Install the required packages:**

	Required python 3.9 or lower.

   ```bash
	 pip -m venv venv
	 .\venv\Scripts\activate
	 pip install -r requirements.txt

3. **Prepare the dataset:**

   Download har file from this link.

4. **Run the training script:**

	 Just run the file crawl/ssi/ssi_stock_price.ipynb

	
## Future Work

- Incorporate additional technical indicators to improve prediction accuracy.
- Explore the use of more advanced deep learning architectures, such as GRU or Transformer models.
- Implement real-time stock price prediction using a live data feed.


## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.