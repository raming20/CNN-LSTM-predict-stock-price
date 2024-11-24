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
	- `pandas==2.2.2`
	- `hdfs==2.7.3`
	- `numpy==1.26.4`
	- `ta==0.11.0`
	- `pyspark==3.5.1`
	- `scikit-learn==1.5.1`
	- `matplotlib==3.9.2`
	- `yfinance==0.2.41`
	- `scipy==1.13.1`
	- `mplfinance==0.12.10b0`
	- `opencv-python==4.10.0.84`
	- `torch==2.4.1`
	- `torchvision==0.19.1`
	- `torchaudio==2.4.1`
	- `tensorflow==2.17.0`
	- `keras==3.6.0`
	- `tensorflow-docs==2024.10.14.18741`
	- `yahoo-finance==1.4.0`
	- `openpyxl==3.1.5`
	- `tabulate==0.9.0`

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Hieucaohd/LSTM-predict-stock-by-technical-analysis.git  
   cd LSTM-predict-stock-by-technical-analysis  

2. Install required packages:  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Usage  
### 1. Prepare Data
- Run file: 
	```
	data_analytics/CV_stock_market/generate_data/generate_datase.ipynb
	```
- The output is a tensorflow dataset, placed in folder: 
	```
	data_analytics/CV_stock_market/dataset
	```
	The output contains candle images in numpy arrays format, 4 type of prices in 6 days.

### 2. Train the Model  
Run the training script:  
```bash  
data_analytics/CV_stock_market/train_model/model_use_train_and_test.ipynb
```  
Adjust hyperparameters in prompt for better performance.  

### 3. Predict Future Prices  
Use the trained model to make predictions, run the file:  
```bash  
data_analytics/CV_stock_market/test_model/test_model_use_train_and_test_with_ema_macd_trend.ipynb 
```  
Output will be saved as a .png file in the `data_analytics/CV_stock_market/dataset/<dataset_name>/output_prediction_image` directory.  

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
- Financial data sourced from [Yahoo Finance](https://finance.yahoo.com) and [SSI](https://www.ssi.com.vn/).  
