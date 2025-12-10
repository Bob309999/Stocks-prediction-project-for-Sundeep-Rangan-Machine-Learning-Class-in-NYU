# Stocks-prediction-project-for-Sundeep-Rangan-Machine-Learning-Class-in-NYU

# 📈 Quant-ML-Trading: Machine Learning Stock Trend Prediction

**Quant-ML-Trading** is a robust quantitative trading framework based on **LSTM** networks. It is designed to predict the **price direction (Up/Down)** of stocks for the next trading day using **China A-Share historical data** downloaded via **Baostock**, combined with technical indicators.

Unlike traditional price regression models that often suffer from "lagging" or mean reversion, this project utilizes a **Binary Classification** approach combined with **Logarithmic Returns**. It effectively mitigates overfitting and the "Permabull" bias (always predicting UP), demonstrating strong defensive capabilities and alpha generation in backtests.

-----

## 👥 Team Members

* **Sijia Hu** (ssh9976)
* **Mengzhe Wang** (mw6093)
* **Franklin Bozhi Chen** (fc2636)

-----

## 📖 Background & Motivation

In financial time-series forecasting, predicting the exact stock price (Regression) is notoriously difficult due to low signal-to-noise ratios. Common failures include:

1.  **Lagging:** The model simply predicts $P_{t} \approx P_{t-1}$.
2.  **Overfitting:** The model memorizes noise rather than learning market patterns.

**This project solves these issues by:**

* **Simplification:** Switching from Regression (predicting price) to **Classification** (predicting direction).
* **Robustness:** Using a lightweight network architecture (Tiny GRU) with strong regularization.
* **Data Science:** Using **Log Returns** for stationarity and **Class Weighting** to handle market imbalances.

-----

## ✨ Key Features

* **Stable Data Source:** Integrated with **Baostock API** for reliable, free, and fast A-share (China market) historical data.
* **Advanced Feature Engineering:** Automatically computes 9 key features:
    * **OHLCV:** Open, High, Low, Close, Volume.
    * **Trend:** MA5, MA20.
    * **Momentum:** RSI, MACD.
* **Anti-Overfitting Architecture:**
    * Implements **Dropout**, **Batch Normalization**, and **Weight Decay**.
    * Uses **Early Stopping** to prevent training on noise.
* **Smart Training:**
    * **Auto-Weighted Loss:** Automatically calculates positive/negative sample ratios to penalize the model for biased guessing.
    * Uses `BCEWithLogitsLoss` for numerical stability.
* **Professional Backtesting:**
    * Visualizes Equity Curves (Strategy vs. Benchmark).
    * Calculates Precision (Win Rate), Accuracy, and Cumulative Returns.

-----

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install baostock pandas numpy matplotlib seaborn torch scikit-learn tqdm
````

-----

## 🚀 Usage

### 1\. Run the Strategy

Simply execute the main script. The system will download data, process features, train the model, and generate a backtest report.

```bash
python main.py
```

### 2\. Configuration

You can customize the target stock, date range, and model hyperparameters in the `AppConfig` class within `main.py`:

```python
class AppConfig:
    # Target Stock (Baostock format: 'sh.xxxxxx' or 'sz.xxxxxx')
    SYMBOL: str = 'sh.600519'  # e.g., Kweichow Moutai
    
    # Backtest Period
    START_DATE: str = '2015-01-01'
    
    # Strategy Threshold (Confidence level required to Buy)
    # Higher = More conservative; Lower = More aggressive
    THRESHOLD: float = 0.485
    
    # ... other params
```

-----

## 📊 Performance & Results

### 1. Defensive Strategy (e.g., Kweichow Moutai)
![Moutai Result][(.Pictures/茅台.png)](https://github.com/Bob309999/Stocks-prediction-project-for-Sundeep-Rangan-Machine-Learning-Class-in-NYU/blob/main/Pictures/%E8%8C%85%E5%8F%B0.png?raw=true)


* **Precision (Win Rate):** Achieved **50.98%** with a highly effective risk-control logic. While the win rate is balanced, the strategy excels in timing the market exits.
* **Downside Protection:** The strategy successfully identified the major market crash (as seen in the flat purple line segments), staying in **Cash** to avoid a significant drawdown (~20% loss avoided).
* **Outperformance:** By preserving capital during the downturn and re-entering during the rebound, the strategy significantly outperformed the benchmark (Buy & Hold), demonstrating genuine **Alpha**.



*(Note: The charts above are generated automatically by `main.py` after training.)*

-----

## 📂 Project Structure

```text
.
├── main.py              # Core entry point (Config, Data, Model)
├── final_report.png     # Generated visualization of the backtest result
└── README.md            # Project documentation
```

-----

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Financial markets are highly risky and unpredictable. The model is trained on historical data, which does not guarantee future performance.

**Do not use this code for live trading with real money without professional risk management.** The authors assume no responsibility for any financial losses.

-----

**License:** MIT
