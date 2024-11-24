# Stock Price Prediction Using CNN-LSTM  
This project leverages a hybrid **CNN-LSTM** model to predict the percentage changes in opening and closing prices of stocks for the next 3 days. The model uses **TensorFlow** and takes candlestick chart images along with EMA_9 slope and MACD histogram slope as input.  

## Project Overview  
### Input:  
- **Candlestick chart images**: Each image represents the candlestick pattern of the past 3 days.  
- **EMA_9 slope**: The slope of the 9-day Exponential Moving Average (EMA) corresponding to each of the 3 days.  
- **MACD histogram slope**: The slope of the MACD histogram for the same 3 days.  

### Output:  
- **Percentage change in opening price** for the next 3 days.  
- **Percentage change in closing price** for the next 3 days.  

## Architecture  
The model combines:  
1. **CNN**: Extracts features from candlestick chart images, capturing visual patterns from the candlestick and price movement.  
2. **LSTM**: Processes time-series data (EMA_9 and MACD slopes) to capture temporal dependencies.  
3. **Fully Connected Layer**: Combines CNN and LSTM outputs to predict percentage changes in stock prices.  

## Loss Function and Metrics  
- **Loss Function**: Mean Squared Error (MSE).  
- **Evaluation Metric**: Mean Absolute Error (MAE).  

## Dataset  
- **Candlestick Charts**: Generated using historical stock data, with each chart resized to `287x287x3`.  
- **EMA_9 and MACD Histogram Slopes**: Calculated using standard formulas for technical indicators.  

## Prerequisites  
- Python 3.9+  
- TensorFlow 2.12+  
- Libraries:  
  - `numpy`  
  - `matplotlib`  
  - `pandas`  
  - `mplfinance`  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/stock-prediction-cnn-lstm.git  
   cd stock-prediction-cnn-lstm  

2. Install required packages:  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Usage  
### 1. Prepare Data  
- Place candlestick chart images in the `data/images` directory.  
- Save EMA_9 and MACD slope data as a CSV file in the format:  
  ```
  date, ema_9_slope_day1, ema_9_slope_day2, ema_9_slope_day3, macd_slope_day1, macd_slope_day2, macd_slope_day3  
  ```  

### 2. Train the Model  
Run the training script:  
```bash  
python train.py  
```  
Adjust hyperparameters in `config.py` for better performance.  

### 3. Predict Future Prices  
Use the trained model to make predictions:  
```bash  
python predict.py --input data/sample_input.csv  
```  
Output will be saved as a CSV file in the `results/` directory.  

## Results  
### Evaluation Metrics  
- **Loss**: Mean Squared Error (MSE).  
- **Metric**: Mean Absolute Error (MAE).  

### Sample Predictions  
| Date       | Predicted Open (%) | Predicted Close (%) |  
|------------|--------------------|---------------------|  
| Day 1      | +2.3%              | -1.1%              |  
| Day 2      | +0.8%              | +0.5%              |  
| Day 3      | -0.6%              | +1.9%              |  

## Future Work  
- Incorporate more technical indicators.  
- Experiment with different CNN and LSTM architectures.  
- Optimize for real-time inference.  

## Contributing  
Contributions are welcome! Please submit a pull request or open an issue for discussion.  

## License  
This project is licensed under the MIT License.  

## Acknowledgements  
- **TensorFlow** for providing the tools to implement CNN-LSTM.  
- Financial data sourced from [Yahoo Finance](https://finance.yahoo.com).  
