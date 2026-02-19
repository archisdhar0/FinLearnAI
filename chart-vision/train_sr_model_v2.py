"""
Support/Resistance Model Training Pipeline V2
SIMPLIFIED APPROACH: Predict S/R zones (bins) instead of exact values

This version:
1. Divides the price range into 10 bins (zones)
2. Model predicts which bins contain S/R levels (multi-label classification)
3. Much easier for CNN to learn than exact regression

Target: 60%+ accuracy
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

sys.path.insert(0, str(Path(__file__).parent))

from utils.chart_generator import PolygonDataFetcher, draw_candlestick_chart


# =============================================================================
# SIMPLIFIED: ZONE-BASED S/R DETECTION
# =============================================================================

NUM_ZONES = 10  # Divide price range into 10 zones


def find_sr_zones(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    num_zones: int = NUM_ZONES,
    window: int = 5
) -> dict:
    """
    Find which zones contain S/R levels.
    
    Returns binary arrays indicating which zones have support/resistance.
    """
    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min
    
    if price_range == 0:
        return {
            'support_zones': [0] * num_zones,
            'resistance_zones': [0] * num_zones,
            'price_range': (price_min, price_max)
        }
    
    def price_to_zone(price):
        """Convert price to zone index (0 to num_zones-1)."""
        normalized = (price - price_min) / price_range
        zone = int(normalized * num_zones)
        return min(zone, num_zones - 1)  # Clamp to valid range
    
    # Find local minima (support)
    support_zones = [0] * num_zones
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            zone = price_to_zone(lows[i])
            support_zones[zone] = 1
    
    # Find local maxima (resistance)
    resistance_zones = [0] * num_zones
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            zone = price_to_zone(highs[i])
            resistance_zones[zone] = 1
    
    return {
        'support_zones': support_zones,
        'resistance_zones': resistance_zones,
        'price_range': (float(price_min), float(price_max))
    }


def generate_sr_training_data(
    api_key: str,
    output_dir: str = "data/sr_training_v2",
    num_samples: int = 800,
    window_size: int = 60
):
    """Generate training dataset with zone-based S/R labels."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    # More tickers for diversity
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
        'SPY', 'QQQ', 'IWM', 'GLD', 'V', 'MA', 'WMT', 'DIS', 'NFLX',
        'AMD', 'INTC', 'BA', 'XOM', 'CVX', 'PFE', 'KO', 'PEP',
        'COST', 'HD', 'NKE', 'MCD', 'SBUX'
    ]
    
    samples = []
    sample_idx = 0
    
    print(f"Generating {num_samples} S/R training samples (zone-based)...")
    print(f"Using {NUM_ZONES} zones for classification")
    
    for ticker in tickers:
        if sample_idx >= num_samples:
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
            print(f"  Error fetching {ticker}: {e}")
            continue
        
        if df is None or len(df) < window_size + 20:
            print(f"  Skipping {ticker} - insufficient data")
            continue
        
        # Slide through data with smaller step for more samples
        for start_idx in range(0, len(df) - window_size, window_size // 4):
            if sample_idx >= num_samples:
                break
            
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            
            # Get zone-based S/R
            sr_data = find_sr_zones(
                window_df['High'].values,
                window_df['Low'].values,
                window_df['Close'].values
            )
            
            # Skip if no S/R zones found
            if sum(sr_data['support_zones']) == 0 and sum(sr_data['resistance_zones']) == 0:
                continue
            
            # Generate chart image
            filename = f"sr_{sample_idx:05d}_{ticker}.png"
            filepath = output_path / filename
            
            if draw_candlestick_chart(window_df, str(filepath), figsize=(8, 6), dpi=100):
                sample = {
                    'filename': filename,
                    'ticker': ticker,
                    'support_zones': sr_data['support_zones'],
                    'resistance_zones': sr_data['resistance_zones'],
                    'price_range': sr_data['price_range'],
                    'start_date': str(window_df.index[0].date()),
                    'end_date': str(window_df.index[-1].date())
                }
                samples.append(sample)
                sample_idx += 1
                
                if sample_idx % 100 == 0:
                    print(f"  Generated {sample_idx}/{num_samples} samples")
    
    # Save labels
    labels_path = output_path / 'sr_labels.json'
    with open(labels_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✅ Generated {len(samples)} samples")
    print(f"   Saved to: {output_path}")
    
    return samples


# =============================================================================
# ZONE-BASED MODEL (Multi-label Classification)
# =============================================================================

class SRZoneDataset(Dataset):
    """Dataset for zone-based S/R prediction."""
    
    def __init__(self, data_dir: str, split: str = 'train', augment: bool = True):
        self.data_dir = Path(data_dir)
        
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
        
        # Data augmentation for training
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
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
        
        # Target: [support_zones (10), resistance_zones (10)] = 20 binary values
        target = torch.tensor(
            sample['support_zones'] + sample['resistance_zones'],
            dtype=torch.float32
        )
        
        return image, target


class SRZoneModel(nn.Module):
    """
    CNN model to predict S/R zones (multi-label classification).
    
    Output: 20 values (10 support zones + 10 resistance zones)
    Each value is probability that zone contains S/R level.
    """
    
    def __init__(self, num_zones: int = NUM_ZONES):
        super().__init__()
        self.num_zones = num_zones
        
        # Use ResNet34 for better capacity
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Freeze early layers (transfer learning)
        for param in list(backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Custom head for zone classification
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_zones * 2),  # Support + Resistance zones
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.head(features)
    
    def predict_zones(self, image: torch.Tensor, threshold: float = 0.5) -> dict:
        """Predict which zones contain S/R levels."""
        self.eval()
        with torch.no_grad():
            logits = self(image)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            support_probs = probs[:self.num_zones]
            resistance_probs = probs[self.num_zones:]
            
            return {
                'support_zones': [int(p > threshold) for p in support_probs],
                'resistance_zones': [int(p > threshold) for p in resistance_probs],
                'support_probs': support_probs.tolist(),
                'resistance_probs': resistance_probs.tolist()
            }


# =============================================================================
# TRAINING
# =============================================================================

def zone_accuracy(pred_zones: list, true_zones: list) -> float:
    """
    Calculate accuracy for zone predictions.
    
    A prediction is correct if:
    - True positive: predicted zone=1 and actual zone=1
    - True negative: predicted zone=0 and actual zone=0
    
    Returns overall accuracy.
    """
    if len(pred_zones) != len(true_zones):
        return 0.0
    
    correct = sum(1 for p, t in zip(pred_zones, true_zones) if p == t)
    return correct / len(pred_zones)


def zone_f1_score(pred_zones: list, true_zones: list) -> float:
    """Calculate F1 score for zone predictions (handles class imbalance)."""
    tp = sum(1 for p, t in zip(pred_zones, true_zones) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_zones, true_zones) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred_zones, true_zones) if p == 0 and t == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class SRZoneTrainer:
    """Training pipeline for zone-based S/R detector."""
    
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
        
        # Binary cross entropy for multi-label classification
        # Use pos_weight to handle class imbalance (more 0s than 1s)
        pos_weight = torch.ones(NUM_ZONES * 2) * 3.0  # Weight positive class more
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-4,  # Higher LR since we're using transfer learning
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=50,
            steps_per_epoch=len(train_loader)
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, targets in tqdm(self.train_loader, desc='Training', leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_accuracies = []
        all_f1_scores = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy per sample
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                for i in range(len(preds)):
                    pred_list = preds[i].cpu().numpy().tolist()
                    true_list = targets[i].cpu().numpy().tolist()
                    
                    # Convert to int for comparison
                    pred_int = [int(p) for p in pred_list]
                    true_int = [int(t) for t in true_list]
                    
                    acc = zone_accuracy(pred_int, true_int)
                    f1 = zone_f1_score(pred_int, true_int)
                    
                    all_accuracies.append(acc)
                    all_f1_scores.append(f1)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = np.mean(all_accuracies)
        avg_f1 = np.mean(all_f1_scores)
        
        return avg_loss, avg_accuracy, avg_f1
    
    def train(self, epochs: int = 50, save_dir: str = 'checkpoints', target_acc: float = 0.60):
        """Train until target accuracy or max epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%} | Val F1: {val_f1:.3f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                }, save_path / 'sr_zone_model_best.pt')
                print(f"  ✅ New best model (accuracy: {val_acc:.1%})")
            else:
                patience_counter += 1
            
            # Early stopping
            if val_acc >= target_acc:
                print(f"\n🎯 TARGET ACCURACY {target_acc:.0%} REACHED!")
                break
            
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping (no improvement for {patience} epochs)")
                break
        
        self._plot_history(save_path)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Accuracy: {best_acc:.1%}")
        print(f"Target: {target_acc:.0%}")
        print(f"{'='*60}")
        
        return self.history, best_acc
    
    def _plot_history(self, save_dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['val_accuracy'])
        axes[1].axhline(y=0.6, color='r', linestyle='--', label='Target (60%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.history['val_f1'])
        axes[2].set_title('Validation F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sr_zone_training_history.png', dpi=150)
        plt.close()


def train_sr_zone_model(data_dir: str = 'data/sr_training_v2', epochs: int = 50):
    """Main training function."""
    
    train_dataset = SRZoneDataset(data_dir, split='train', augment=True)
    val_dataset = SRZoneDataset(data_dir, split='val', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = SRZoneModel(num_zones=NUM_ZONES)
    
    trainer = SRZoneTrainer(model, train_loader, val_loader)
    history, best_acc = trainer.train(epochs=epochs, target_acc=0.60)
    
    return model, history, best_acc


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train S/R Zone Detection Model V2')
    parser.add_argument('--mode', choices=['generate', 'train', 'both'], default='both')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    parser.add_argument('--samples', type=int, default=800, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--data-dir', type=str, default='data/sr_training_v2')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if args.mode in ['generate', 'both']:
        if not api_key:
            print("❌ Polygon API key required!")
            return
        
        print("\n" + "="*60)
        print("STEP 1: Generating Training Data (Zone-Based)")
        print("="*60)
        generate_sr_training_data(api_key, args.data_dir, args.samples)
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training S/R Zone Model")
        print("="*60)
        model, history, best_acc = train_sr_zone_model(args.data_dir, args.epochs)
        
        if best_acc >= 0.60:
            print("\n✅ SUCCESS: Model achieved 60%+ accuracy!")
        else:
            print(f"\n⚠️ Model accuracy: {best_acc:.1%} (below 60% target)")


if __name__ == "__main__":
    main()
