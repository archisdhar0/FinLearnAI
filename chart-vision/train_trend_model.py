"""
Trend/Slope Classifier Training Pipeline
Goal: Train model to detect uptrend/downtrend/sideways with 80%+ accuracy

The model learns to identify the overall slope direction from chart images:
- Uptrend: Positive slope (prices generally rising)
- Downtrend: Negative slope (prices generally falling)  
- Sideways: Flat/no clear direction

Ground truth is calculated from linear regression slope of closing prices.
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from utils.chart_generator import PolygonDataFetcher, draw_candlestick_chart


# =============================================================================
# TREND CLASSIFICATION LOGIC
# =============================================================================

def calculate_trend_label(closes: np.ndarray, threshold: float = 0.15) -> tuple:
    """
    Calculate trend label from closing prices using linear regression slope.
    
    Args:
        closes: Array of closing prices
        threshold: Normalized slope threshold for trend classification
    
    Returns:
        (label, slope, confidence)
        label: 'uptrend', 'downtrend', or 'sideways'
        slope: Normalized slope value
        confidence: How clear the trend is (0-1)
    """
    if len(closes) < 5:
        return 'sideways', 0.0, 0.0
    
    # Linear regression
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    slope = coeffs[0]
    
    # Normalize slope by average price (percentage change per bar)
    avg_price = np.mean(closes)
    norm_slope = (slope / avg_price) * 100  # Percent per bar
    
    # Calculate R-squared for confidence
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((closes - y_pred) ** 2)
    ss_tot = np.sum((closes - np.mean(closes)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate volatility (noise)
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns)
    
    # Adjust threshold based on volatility
    adjusted_threshold = threshold * (1 + volatility * 10)
    
    # Classify
    if norm_slope > adjusted_threshold and r_squared > 0.3:
        label = 'uptrend'
    elif norm_slope < -adjusted_threshold and r_squared > 0.3:
        label = 'downtrend'
    else:
        label = 'sideways'
    
    # Confidence based on how clear the trend is
    confidence = min(abs(norm_slope) / (adjusted_threshold * 2), 1.0) * r_squared
    
    return label, float(norm_slope), float(confidence)


def visualize_trend(closes: np.ndarray, label: str, slope: float, save_path: str = None):
    """Visualize the trend with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(closes))
    ax.plot(x, closes, 'b-', linewidth=2, label='Price')
    
    # Regression line
    coeffs = np.polyfit(x, closes, 1)
    y_pred = np.polyval(coeffs, x)
    
    color = {'uptrend': 'green', 'downtrend': 'red', 'sideways': 'gray'}[label]
    ax.plot(x, y_pred, color=color, linestyle='--', linewidth=2, 
            label=f'{label.upper()} (slope: {slope:.3f})')
    
    ax.set_title(f'Trend: {label.upper()}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_trend_training_data(
    api_key: str,
    output_dir: str = "data/trend_training",
    samples_per_class: int = 200,
    window_size: int = 60
):
    """
    Generate balanced training dataset for trend classification.
    
    Ensures equal samples of uptrend, downtrend, and sideways.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
        'SPY', 'QQQ', 'IWM', 'GLD', 'V', 'MA', 'WMT', 'DIS', 'NFLX',
        'AMD', 'INTC', 'BA', 'XOM', 'CVX', 'PFE', 'KO', 'PEP'
    ]
    
    samples = {'uptrend': [], 'downtrend': [], 'sideways': []}
    
    print(f"Generating {samples_per_class} samples per class...")
    print(f"Target: {samples_per_class * 3} total samples")
    
    for ticker in tickers:
        # Check if we have enough
        if all(len(v) >= samples_per_class for v in samples.values()):
            break
        
        print(f"\nFetching {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        
        try:
            df = fetcher.get_daily_bars(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        if df is None or len(df) < window_size + 20:
            continue
        
        # Slide through data
        for start_idx in range(0, len(df) - window_size, window_size // 4):
            # Check if we need more of any class
            if all(len(v) >= samples_per_class for v in samples.values()):
                break
            
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            closes = window_df['Close'].values
            
            # Get trend label
            label, slope, confidence = calculate_trend_label(closes)
            
            # Skip low-confidence samples
            if confidence < 0.3 and label != 'sideways':
                continue
            
            # Skip if we have enough of this class
            if len(samples[label]) >= samples_per_class:
                continue
            
            # Generate chart image
            idx = len(samples[label])
            filename = f"trend_{label}_{idx:04d}_{ticker}.png"
            filepath = output_path / filename
            
            if draw_candlestick_chart(window_df, str(filepath), figsize=(8, 6), dpi=100):
                sample = {
                    'filename': filename,
                    'ticker': ticker,
                    'label': label,
                    'slope': slope,
                    'confidence': confidence,
                    'start_date': str(window_df.index[0].date()),
                    'end_date': str(window_df.index[-1].date()),
                    'price_start': float(closes[0]),
                    'price_end': float(closes[-1]),
                    'price_change_pct': float((closes[-1] - closes[0]) / closes[0] * 100)
                }
                samples[label].append(sample)
                
                total = sum(len(v) for v in samples.values())
                if total % 50 == 0:
                    print(f"  Progress: {total}/{samples_per_class * 3}")
                    print(f"    Up: {len(samples['uptrend'])}, Down: {len(samples['downtrend'])}, Side: {len(samples['sideways'])}")
    
    # Flatten and save
    all_samples = []
    for label, sample_list in samples.items():
        all_samples.extend(sample_list)
    
    # Save labels
    labels_path = output_path / 'trend_labels.json'
    with open(labels_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Save class mapping
    class_counts = {k: len(v) for k, v in samples.items()}
    
    print(f"\n✅ Generated {len(all_samples)} samples")
    print(f"   Uptrend: {class_counts['uptrend']}")
    print(f"   Downtrend: {class_counts['downtrend']}")
    print(f"   Sideways: {class_counts['sideways']}")
    print(f"   Saved to: {output_path}")
    
    return all_samples


# =============================================================================
# MODEL
# =============================================================================

class TrendDataset(Dataset):
    """Dataset for trend classification."""
    
    CLASSES = ['downtrend', 'sideways', 'uptrend']
    
    def __init__(self, data_dir: str, split: str = 'train', augment: bool = True):
        self.data_dir = Path(data_dir)
        
        # Load labels
        with open(self.data_dir / 'trend_labels.json', 'r') as f:
            self.samples = json.load(f)
        
        # Split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.8)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        # Transforms
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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
        
        img_path = self.data_dir / sample['filename']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = self.CLASSES.index(sample['label'])
        
        return image, label


class TrendClassifierModel(nn.Module):
    """
    CNN for trend classification.
    
    Uses EfficientNet-B0 backbone for better accuracy with smaller model.
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        # EfficientNet backbone (better than ResNet for this task)
        try:
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        except:
            # Fallback to ResNet18
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image: torch.Tensor) -> dict:
        """Make prediction on single image."""
        self.eval()
        with torch.no_grad():
            logits = self(image)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            return {
                'prediction': TrendDataset.CLASSES[pred_idx],
                'confidence': probs[0, pred_idx].item(),
                'probabilities': {
                    cls: probs[0, i].item()
                    for i, cls in enumerate(TrendDataset.CLASSES)
                }
            }


# =============================================================================
# TRAINING
# =============================================================================

class TrendTrainer:
    """Training pipeline for trend classifier."""
    
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
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc='Training'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, epochs: int = 30, save_dir: str = 'checkpoints', target_acc: float = 0.80):
        """Train until target accuracy or max epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, preds, labels = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                }, save_path / 'trend_model_best.pt')
                print(f"✅ New best model saved (accuracy: {val_acc:.2%})")
            
            if val_acc >= target_acc:
                print(f"\n🎯 TARGET ACCURACY {target_acc:.0%} REACHED!")
                break
        
        # Final evaluation
        self._plot_history(save_path)
        self._plot_confusion_matrix(preds, labels, save_path)
        
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
        plt.savefig(save_dir / 'trend_training_history.png', dpi=150)
        plt.close()
    
    def _plot_confusion_matrix(self, preds, labels, save_dir):
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=TrendDataset.CLASSES,
            yticklabels=TrendDataset.CLASSES
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_dir / 'trend_confusion_matrix.png', dpi=150)
        plt.close()
        
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=TrendDataset.CLASSES))


def train_trend_model(data_dir: str = 'data/trend_training', epochs: int = 30):
    """Main training function."""
    
    train_dataset = TrendDataset(data_dir, split='train', augment=True)
    val_dataset = TrendDataset(data_dir, split='val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = TrendClassifierModel(num_classes=3)
    
    trainer = TrendTrainer(model, train_loader, val_loader)
    history, best_acc = trainer.train(epochs=epochs, target_acc=0.80)
    
    return model, history, best_acc


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Trend Classifier Model')
    parser.add_argument('--mode', choices=['generate', 'train', 'both'], default='both')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    parser.add_argument('--samples', type=int, default=200, help='Samples per class')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--data-dir', type=str, default='data/trend_training')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if args.mode in ['generate', 'both']:
        if not api_key:
            print("❌ Polygon API key required!")
            print("   Set POLYGON_API_KEY or use --api-key")
            return
        
        print("\n" + "="*60)
        print("STEP 1: Generating Training Data")
        print("="*60)
        generate_trend_training_data(api_key, args.data_dir, args.samples)
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training Trend Model")
        print("="*60)
        model, history, best_acc = train_trend_model(args.data_dir, args.epochs)
        
        if best_acc >= 0.80:
            print("\n✅ SUCCESS: Model achieved 80%+ accuracy!")
        else:
            print(f"\n⚠️ Model accuracy: {best_acc:.1%} (below 80% target)")


if __name__ == "__main__":
    main()
