"""
Support/Resistance Model Training Pipeline
Goal: Fine-tune model to achieve 80%+ accuracy comparing predicted lines to ground truth

The evaluation compares:
- Predicted S/R levels (y-coordinates or prices)
- Ground truth S/R levels (from price data local min/max)

Accuracy metric: % of predicted levels within tolerance of a ground truth level
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from utils.chart_generator import PolygonDataFetcher, draw_candlestick_chart


# =============================================================================
# DATASET GENERATION WITH GROUND TRUTH S/R LABELS
# =============================================================================

def find_ground_truth_sr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 5,
    cluster_threshold: float = 0.02
) -> dict:
    """
    Find ground truth support/resistance levels from price data.
    
    Returns normalized levels (0-1 range based on price range).
    """
    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min
    
    if price_range == 0:
        return {'support': [], 'resistance': [], 'price_range': (price_min, price_max)}
    
    # Find local minima (support)
    support_prices = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            support_prices.append(lows[i])
    
    # Find local maxima (resistance)
    resistance_prices = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_prices.append(highs[i])
    
    # Cluster nearby levels
    def cluster_levels(prices):
        if not prices:
            return []
        prices = sorted(prices)
        clusters = []
        current = [prices[0]]
        
        for p in prices[1:]:
            if (p - current[-1]) / current[-1] < cluster_threshold:
                current.append(p)
            else:
                clusters.append(np.mean(current))
                current = [p]
        clusters.append(np.mean(current))
        return clusters
    
    support_levels = cluster_levels(support_prices)
    resistance_levels = cluster_levels(resistance_prices)
    
    # Normalize to 0-1 range
    def normalize(price):
        return (price - price_min) / price_range
    
    return {
        'support': [normalize(p) for p in support_levels[:5]],  # Top 5
        'resistance': [normalize(p) for p in resistance_levels[:5]],
        'support_prices': support_levels[:5],
        'resistance_prices': resistance_levels[:5],
        'price_range': (float(price_min), float(price_max))
    }


def generate_sr_training_data(
    api_key: str,
    output_dir: str = "data/sr_training",
    num_samples: int = 500,
    window_size: int = 60
):
    """
    Generate training dataset with ground truth S/R labels.
    
    Each sample includes:
    - Chart image
    - Ground truth S/R levels (normalized 0-1)
    - Price data for verification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
        'SPY', 'QQQ', 'IWM', 'GLD', 'V', 'MA', 'WMT', 'DIS', 'NFLX'
    ]
    
    samples = []
    sample_idx = 0
    
    print(f"Generating {num_samples} S/R training samples...")
    
    for ticker in tickers:
        if sample_idx >= num_samples:
            break
            
        print(f"\nFetching {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        
        try:
            df = fetcher.get_daily_bars(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            continue
        
        if df is None or len(df) < window_size + 20:
            continue
        
        # Slide through data
        for start_idx in range(0, len(df) - window_size, window_size // 3):
            if sample_idx >= num_samples:
                break
            
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            
            # Get ground truth S/R
            sr_data = find_ground_truth_sr(
                window_df['High'].values,
                window_df['Low'].values,
                window_df['Close'].values
            )
            
            # Skip if no clear S/R levels
            if len(sr_data['support']) < 1 and len(sr_data['resistance']) < 1:
                continue
            
            # Generate chart image
            filename = f"sr_{sample_idx:05d}_{ticker}.png"
            filepath = output_path / filename
            
            if draw_candlestick_chart(window_df, str(filepath), figsize=(8, 6), dpi=100):
                sample = {
                    'filename': filename,
                    'ticker': ticker,
                    'support': sr_data['support'],
                    'resistance': sr_data['resistance'],
                    'support_prices': sr_data['support_prices'],
                    'resistance_prices': sr_data['resistance_prices'],
                    'price_range': sr_data['price_range'],
                    'start_date': str(window_df.index[0].date()),
                    'end_date': str(window_df.index[-1].date())
                }
                samples.append(sample)
                sample_idx += 1
                
                if sample_idx % 50 == 0:
                    print(f"  Generated {sample_idx}/{num_samples} samples")
    
    # Save labels
    labels_path = output_path / 'sr_labels.json'
    with open(labels_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✅ Generated {len(samples)} samples")
    print(f"   Saved to: {output_path}")
    print(f"   Labels: {labels_path}")
    
    return samples


# =============================================================================
# S/R DETECTION MODEL (CNN-based regression)
# =============================================================================

class SRDataset(Dataset):
    """Dataset for S/R level prediction."""
    
    def __init__(self, data_dir: str, split: str = 'train', max_levels: int = 5):
        self.data_dir = Path(data_dir)
        self.max_levels = max_levels
        
        # Load labels
        with open(self.data_dir / 'sr_labels.json', 'r') as f:
            self.samples = json.load(f)
        
        # Split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / sample['filename']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Prepare target: [support_levels..., resistance_levels...]
        # Pad to max_levels with -1 (invalid marker)
        support = sample['support'][:self.max_levels]
        resistance = sample['resistance'][:self.max_levels]
        
        # Pad
        support = support + [-1] * (self.max_levels - len(support))
        resistance = resistance + [-1] * (self.max_levels - len(resistance))
        
        target = torch.tensor(support + resistance, dtype=torch.float32)
        
        return image, target


class SRDetectorModel(nn.Module):
    """
    CNN model to predict S/R levels from chart images.
    
    Output: [5 support levels, 5 resistance levels] normalized 0-1
    Invalid levels are marked as -1
    """
    
    def __init__(self, max_levels: int = 5):
        super().__init__()
        self.max_levels = max_levels
        
        # ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom head for S/R regression
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, max_levels * 2),  # Support + Resistance
            nn.Sigmoid()  # Output 0-1 range
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.head(features)
    
    def predict(self, image: torch.Tensor, threshold: float = 0.05) -> dict:
        """
        Predict S/R levels from image.
        
        Args:
            image: Preprocessed image tensor
            threshold: Minimum value to consider as valid level
        
        Returns:
            Dictionary with support and resistance levels
        """
        self.eval()
        with torch.no_grad():
            output = self(image)
            levels = output[0].cpu().numpy()
            
            support = levels[:self.max_levels]
            resistance = levels[self.max_levels:]
            
            # Filter out near-zero predictions
            support = [float(s) for s in support if s > threshold]
            resistance = [float(r) for r in resistance if r > threshold]
            
            return {
                'support': sorted(support),
                'resistance': sorted(resistance, reverse=True)
            }


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def sr_accuracy(pred_levels: list, true_levels: list, tolerance: float = 0.05) -> float:
    """
    Calculate accuracy: % of predictions within tolerance of a ground truth level.
    
    Args:
        pred_levels: Predicted levels (0-1 normalized)
        true_levels: Ground truth levels (0-1 normalized)
        tolerance: Max distance to count as correct (default 5% of price range)
    
    Returns:
        Accuracy score 0-1
    """
    if not pred_levels or not true_levels:
        return 0.0
    
    correct = 0
    for pred in pred_levels:
        if pred < 0:  # Invalid marker
            continue
        # Check if any ground truth is within tolerance
        for true in true_levels:
            if true < 0:
                continue
            if abs(pred - true) <= tolerance:
                correct += 1
                break
    
    return correct / max(len([p for p in pred_levels if p >= 0]), 1)


class SRTrainer:
    """Training pipeline for S/R detector."""
    
    def __init__(self, model, train_loader, val_loader, device='auto'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Custom loss: MSE only for valid levels (not -1)
        self.criterion = nn.MSELoss(reduction='none')
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def masked_loss(self, pred, target):
        """MSE loss ignoring -1 markers in target."""
        mask = target >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        loss = self.criterion(pred, torch.clamp(target, 0, 1))
        return (loss * mask).sum() / mask.sum()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(self.train_loader, desc='Training'):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.masked_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_accuracies = []
        
        max_levels = self.model.max_levels
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.masked_loss(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy per sample
                for i in range(len(outputs)):
                    pred = outputs[i].cpu().numpy()
                    true = targets[i].cpu().numpy()
                    
                    pred_support = [p for p in pred[:max_levels] if p > 0.05]
                    pred_resist = [p for p in pred[max_levels:] if p > 0.05]
                    true_support = [t for t in true[:max_levels] if t >= 0]
                    true_resist = [t for t in true[max_levels:] if t >= 0]
                    
                    acc_s = sr_accuracy(pred_support, true_support)
                    acc_r = sr_accuracy(pred_resist, true_resist)
                    
                    all_accuracies.append((acc_s + acc_r) / 2)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
        
        return avg_loss, avg_accuracy
    
    def train(self, epochs: int = 30, save_dir: str = 'checkpoints', target_acc: float = 0.80):
        """Train until target accuracy or max epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*60)
            
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_acc)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                }, save_path / 'sr_model_best.pt')
                print(f"✅ New best model saved (accuracy: {val_acc:.2%})")
            
            # Early stopping if target reached
            if val_acc >= target_acc:
                print(f"\n🎯 TARGET ACCURACY {target_acc:.0%} REACHED!")
                break
        
        # Plot training history
        self._plot_history(save_path)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Accuracy: {best_acc:.2%}")
        print(f"Target: {target_acc:.0%}")
        print(f"{'='*60}")
        
        return self.history, best_acc
    
    def _plot_history(self, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history['val_accuracy'])
        ax2.axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sr_training_history.png', dpi=150)
        plt.close()


def train_sr_model(data_dir: str = 'data/sr_training', epochs: int = 30):
    """Main training function for S/R model."""
    
    # Create datasets
    train_dataset = SRDataset(data_dir, split='train')
    val_dataset = SRDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    model = SRDetectorModel(max_levels=5)
    
    # Train
    trainer = SRTrainer(model, train_loader, val_loader)
    history, best_acc = trainer.train(epochs=epochs, target_acc=0.80)
    
    return model, history, best_acc


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train S/R Detection Model')
    parser.add_argument('--mode', choices=['generate', 'train', 'both'], default='both',
                        help='generate data, train model, or both')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    parser.add_argument('--samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--data-dir', type=str, default='data/sr_training', help='Data directory')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if args.mode in ['generate', 'both']:
        if not api_key:
            print("❌ Polygon API key required for data generation!")
            print("   Set POLYGON_API_KEY environment variable or use --api-key")
            return
        
        print("\n" + "="*60)
        print("STEP 1: Generating Training Data")
        print("="*60)
        generate_sr_training_data(api_key, args.data_dir, args.samples)
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training S/R Model")
        print("="*60)
        model, history, best_acc = train_sr_model(args.data_dir, args.epochs)
        
        if best_acc >= 0.80:
            print("\n✅ SUCCESS: Model achieved 80%+ accuracy!")
        else:
            print(f"\n⚠️ Model accuracy: {best_acc:.1%} (below 80% target)")
            print("   Try: more training data, more epochs, or adjust hyperparameters")


if __name__ == "__main__":
    main()
