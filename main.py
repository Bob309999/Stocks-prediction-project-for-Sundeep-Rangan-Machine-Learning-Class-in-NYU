import os
import random
import traceback
from typing import Tuple, List, Dict

import baostock as bs 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Force non-interactive backend for headless servers
matplotlib.use('Agg')

# ==========================================
# 1. Global Configuration
# ==========================================
class AppConfig:
    """Hyperparameters and file paths."""
    # SYMBOL: str = 'sh.600900'  # 长江电力 
    # SYMBOL: str = 'sh.601088'  # 中国神华 
    # SYMBOL: str = 'sh.601398'  # 工商银行 
    # SYMBOL: str = 'sh.601857'  # 中国石油 
    SYMBOL: str = 'sh.600519'  # 贵州茅台 
    # SYMBOL: str = 'sz.000858'  # 五 粮 液 
    # SYMBOL: str = 'sh.600887'  # 伊利股份 
    # SYMBOL: str = 'sh.603288'  # 海天味业 
    # SYMBOL: str = 'sh.601318'  # 中国平安
    # SYMBOL: str = 'sh.600030'  # 中信证券 
    # SYMBOL: str = 'sz.000333'  # 美的集团 
    # SYMBOL: str = 'sz.002594'  # 比亚迪   
    # SYMBOL: str = 'sh.601899'  # 紫金矿业  
    # SYMBOL: str = 'sh.601668'  # 中国建筑 
    # SYMBOL: str = 'sh.600276'  # 恒瑞医药 
    # SYMBOL: str = 'sh.601012'  # 隆基绿能 
    # SYMBOL: str = 'sh.601728'  # 中国电信 
    # SYMBOL: str = 'sz.002415'  # 海康威视 
    START_DATE: str = '2015-01-01'
    END_DATE: str = '2025-11-01'
    
    # Data Processing
    LOOK_BACK: int = 60          # Context window size
    SPLIT_RATIO: float = 0.8     # Train/Test split
    
    # Features & Target
    # Baostock returns lowercase column names
    FEATURE_COLS: List[str] = ['close', 'open', 'high', 'low', 'volume', 'ma5', 'ma20', 'rsi', 'macd']
    TARGET_COL: str = 'target'
    
    # Model Architecture
    HIDDEN_SIZE: int = 64        
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.5
    
    # Trading Strategy
    THRESHOLD: float = 0.5   
    
    # Training
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4  
    PATIENCE: int = 20           
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED: int = 42

def set_reproducibility(seed: int = 42):
    """Fix random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# ==========================================
# 2. Data Pipeline (Baostock Implementation)
# ==========================================
class MarketDataProvider:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self) -> pd.DataFrame:
        print(f"[*] Connecting to Baostock server...")
        
        lg = bs.login()
        if lg.error_code != '0':
            raise ConnectionError(f"Baostock login failed: {lg.error_msg}")

        print(f"[*] Fetching daily data for {self.cfg.SYMBOL}...")
        try:
            rs = bs.query_history_k_data_plus(
                self.cfg.SYMBOL,
                "date,open,high,low,close,volume",
                start_date=self.cfg.START_DATE, 
                end_date=self.cfg.END_DATE,
                frequency="d", 
                adjustflag="2"
            )

            if rs.error_code != '0':
                raise ValueError(f"Query failed: {rs.error_msg}")

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            if df.empty:
                raise ValueError(f"Data empty. Check symbol '{self.cfg.SYMBOL}' or date range.")

            df.replace('', np.nan, inplace=True)
            
            df.dropna(inplace=True)

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = df[col].astype(float)
            
            df.sort_index(inplace=True)
            
            return self._feature_engineering(df)
            
        except Exception as e:
            print(f"[!] Data processing failed: {e}")
            raise
        finally:
            bs.logout()

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:

        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        

        df['return'] = np.log(df['close'] / df['close'].shift(1))
        df[self.cfg.TARGET_COL] = (df['return'] > 0).astype(int)
        
        df.dropna(inplace=True)
        print(f"[*] Processed data shape: {df.shape}")
        return df

    def create_dataloaders(self, df: pd.DataFrame) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, int]:
        features = df[self.cfg.FEATURE_COLS].values
        targets = df[self.cfg.TARGET_COL].values
        
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.cfg.LOOK_BACK, len(scaled_features)):
            X.append(scaled_features[i - self.cfg.LOOK_BACK : i])
            y.append(targets[i])
            
        X = np.array(X)
        y = np.array(y)
        
        train_size = int(len(X) * self.cfg.SPLIT_RATIO)
        
        to_tensor = lambda x: torch.FloatTensor(x).to(self.cfg.DEVICE)
        
        X_train = to_tensor(X[:train_size])
        y_train = to_tensor(y[:train_size]).view(-1, 1)
        X_test = to_tensor(X[train_size:])
        y_test = to_tensor(y[train_size:]).view(-1, 1)
        
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        
        return train_loader, X_test, y_test, train_size

# ==========================================
# 3. Model Definition
# ==========================================
class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for trend prediction.
    Outputs raw logits (no Sigmoid) to work with BCEWithLogitsLoss.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Logits output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.bn(out)
        return self.fc(out)

# ==========================================
# 4. Training Engine
# ==========================================
class ModelTrainer:
    """Manages the training loop, validation, and early stopping."""
    
    def __init__(self, model: nn.Module, config: AppConfig, train_loader: DataLoader):
        self.model = model
        self.cfg = config
        
        # Calculate Class Weights
        labels = torch.cat([y for _, y in train_loader])
        pos_count = (labels == 1).sum().item()
        neg_count = (labels == 0).sum().item()
        
        pos_weight = torch.tensor([neg_count / pos_count]).to(config.DEVICE)
        print(f"[*] Class Balance - Up: {pos_count}, Down: {neg_count}. Weight: {pos_weight.item():.4f}")
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def fit(self, train_loader: DataLoader, X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, list]:
        print(f"[*] Training started on {self.cfg.DEVICE}...")
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.cfg.EPOCHS):
            self.model.train()
            batch_losses = []
            
            loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{self.cfg.EPOCHS}")
            for X_batch, y_batch in loop:
                self.optimizer.zero_grad()
                out = self.model(X_batch)
                loss = self.criterion(out, y_batch)
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss.item())
                loop.set_postfix(loss=loss.item())
            
            avg_train_loss = np.mean(batch_losses)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = self.criterion(val_out, y_val).item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_checkpoint.pth')
            else:
                patience_counter += 1
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
            if patience_counter >= self.cfg.PATIENCE:
                print(f"[*] Early stopping triggered at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load('best_checkpoint.pth'))
        return history

# ==========================================
# 5. Evaluation & Visualization
# ==========================================
def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, 
             history: dict, df: pd.DataFrame, train_size: int):
    """Evaluates model performance and plots results."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    preds = (probs > AppConfig.THRESHOLD).astype(int)
    
    # Metrics
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    print("\n" + "="*40)
    print(f"📊 Evaluation Report")
    print(f"Threshold      : {AppConfig.THRESHOLD}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f} (Win Rate)")
    print("="*40)
    
    # Backtesting
    start_idx = train_size + AppConfig.LOOK_BACK
    real_returns = df['return'].values[start_idx:]
    
    n = min(len(real_returns), len(preds))
    real_returns = real_returns[:n]
    preds = preds[:n].flatten()
    
    strat_returns = real_returns * preds
    cum_bench = np.concatenate(([1], (1 + real_returns).cumprod()))
    cum_strat = np.concatenate(([1], (1 + strat_returns).cumprod()))
    
    _plot_results(history, probs, cum_bench, cum_strat, prec)

def _plot_results(history, probs, cum_bench, cum_strat, precision):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss Curve
    ax1 = fig.add_subplot(2, 2, 1)
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    ax1.plot(train_loss, label='Train', color='#2ecc71', linewidth=1.5)
    ax1.plot(val_loss, label='Validation', color='#e67e22', linewidth=1.5)
    
    # Adaptive Y-axis
    all_losses = train_loss + val_loss
    mid = (min(all_losses) + max(all_losses)) / 2
    spread = max(max(all_losses) - min(all_losses), 0.05) * 0.6
    ax1.set_ylim(mid - spread, mid + spread)
    
    ax1.set_title('Learning Curve', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # 2. Probability Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    sns.histplot(probs, bins=30, kde=True, ax=ax2, color='#3498db')
    ax2.axvline(AppConfig.THRESHOLD, color='red', linestyle='--', label=f'Threshold {AppConfig.THRESHOLD}')
    ax2.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # 3. Backtest
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(cum_bench, label='Benchmark (Buy & Hold)', color='#7f8c8d', linewidth=2, alpha=0.6)
    ax3.plot(cum_strat, label='LSTM Strategy', color='#8e44ad', linewidth=2, linestyle='--')
    
    ax3.fill_between(range(len(cum_bench)), cum_bench, cum_strat, 
                     where=(cum_strat > cum_bench), color='#27ae60', alpha=0.1, label='Outperform')
    ax3.fill_between(range(len(cum_bench)), cum_bench, cum_strat, 
                     where=(cum_strat < cum_bench), color='#c0392b', alpha=0.1, label='Underperform')
                     
    ax3.set_title(f'Strategy Performance (Precision: {precision:.2%})', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Normalized Wealth')
    ax3.legend()
    
    output_file = 'final_report.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ [Output] Report saved to '{output_file}'")
    plt.close()

# ==========================================
# Main Entry Point
# ==========================================
if __name__ == '__main__':
    try:
        set_reproducibility(AppConfig.SEED)
        
        # 1. Prepare Data
        provider = MarketDataProvider(AppConfig)
        df_raw = provider.fetch_data()
        train_loader, X_test, y_test, train_sz = provider.create_dataloaders(df_raw)
        
        # 2. Build Model
        model = LSTMClassifier(
            input_dim=len(AppConfig.FEATURE_COLS),
            hidden_dim=AppConfig.HIDDEN_SIZE,
            num_layers=AppConfig.NUM_LAYERS,
            dropout=AppConfig.DROPOUT
        ).to(AppConfig.DEVICE)
        
        # 3. Train
        trainer = ModelTrainer(model, AppConfig, train_loader)
        history = trainer.fit(train_loader, X_test, y_test)
        
        # 4. Evaluate
        evaluate(model, X_test, y_test, history, df_raw, train_sz)
        
    except Exception as e:
        print(f"[!] Critical Error: {str(e)}")
        traceback.print_exc()