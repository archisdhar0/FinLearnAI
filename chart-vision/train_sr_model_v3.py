"""
Support/Resistance Model Training Pipeline V3
TARGET: 80%+ accuracy

Improvements over V2:
1. EfficientNet-B2 backbone (better than ResNet)
2. More aggressive data augmentation
3. Focal loss for better handling of class imbalance
4. Larger dataset (1500+ samples)
5. Mixup augmentation
6. Label smoothing
7. Cosine annealing with warm restarts
8. Gradient accumulation for effective larger batch size

Also includes visualization to draw lines on charts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from utils.chart_generator import PolygonDataFetcher, draw_candlestick_chart


NUM_ZONES = 8  # Fewer zones = easier to learn


# =============================================================================
# FOCAL LOSS (better for imbalanced data)
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# =============================================================================
# DATA GENERATION
# =============================================================================

def find_sr_zones(highs, lows, closes, num_zones=NUM_ZONES, window=5):
    """Find which zones contain S/R levels."""
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
        normalized = (price - price_min) / price_range
        zone = int(normalized * num_zones)
        return min(zone, num_zones - 1)
    
    # Find local minima (support) with strength weighting
    support_zones = [0.0] * num_zones
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            zone = price_to_zone(lows[i])
            # Count touches - more touches = stronger level
            support_zones[zone] += 1
    
    # Find local maxima (resistance)
    resistance_zones = [0.0] * num_zones
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            zone = price_to_zone(highs[i])
            resistance_zones[zone] += 1
    
    # Convert to binary (any touch = 1)
    support_binary = [1 if s > 0 else 0 for s in support_zones]
    resistance_binary = [1 if r > 0 else 0 for r in resistance_zones]
    
    return {
        'support_zones': support_binary,
        'resistance_zones': resistance_binary,
        'support_strength': support_zones,
        'resistance_strength': resistance_zones,
        'price_range': (float(price_min), float(price_max))
    }


def generate_training_data(
    api_key: str,
    output_dir: str = "data/sr_training_v3",
    num_samples: int = 1500,
    window_size: int = 60
):
    """Generate large training dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    # Many tickers for diversity
    tickers = [
        # Large caps
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'MA',
        'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'DIS', 'NFLX', 'ADBE', 'CRM', 'PYPL',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'GLD', 'SLV', 'TLT',
        # Mid caps
        'AMD', 'INTC', 'BA', 'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'MCD', 'SBUX',
        'COST', 'NKE', 'LOW', 'TGT', 'BKNG', 'ABNB', 'UBER', 'LYFT', 'SQ', 'ROKU'
    ]
    
    samples = []
    sample_idx = 0
    
    print(f"Generating {num_samples} samples with {NUM_ZONES} zones...")
    
    for ticker in tickers:
        if sample_idx >= num_samples:
            break
        
        print(f"\nFetching {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 4)  # 4 years of data
        
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
        
        # Slide with small step for more samples
        for start_idx in range(0, len(df) - window_size, window_size // 5):
            if sample_idx >= num_samples:
                break
            
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            
            sr_data = find_sr_zones(
                window_df['High'].values,
                window_df['Low'].values,
                window_df['Close'].values
            )
            
            # Require at least 1 S/R zone
            if sum(sr_data['support_zones']) == 0 and sum(sr_data['resistance_zones']) == 0:
                continue
            
            filename = f"sr_{sample_idx:05d}_{ticker}.png"
            filepath = output_path / filename
            
            if draw_candlestick_chart(window_df, str(filepath), figsize=(8, 6), dpi=100):
                sample = {
                    'filename': filename,
                    'ticker': ticker,
                    'support_zones': sr_data['support_zones'],
                    'resistance_zones': sr_data['resistance_zones'],
                    'price_range': sr_data['price_range'],
                }
                samples.append(sample)
                sample_idx += 1
                
                if sample_idx % 200 == 0:
                    print(f"  Generated {sample_idx}/{num_samples}")
    
    # Save
    with open(output_path / 'sr_labels.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✅ Generated {len(samples)} samples")
    return samples


# =============================================================================
# DATASET WITH MIXUP
# =============================================================================

class SRDatasetV3(Dataset):
    """Dataset with strong augmentation."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        with open(self.data_dir / 'sr_labels.json', 'r') as f:
            self.samples = json.load(f)
        
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.85)  # 85/15 split
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        # Strong augmentation for training
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
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
        
        # Label smoothing: 0 -> 0.05, 1 -> 0.95
        support = [0.95 if s == 1 else 0.05 for s in sample['support_zones']]
        resistance = [0.95 if r == 1 else 0.05 for r in sample['resistance_zones']]
        
        target = torch.tensor(support + resistance, dtype=torch.float32)
        
        return image, target


# =============================================================================
# MODEL
# =============================================================================

class SRModelV3(nn.Module):
    """EfficientNet-based S/R zone predictor."""
    
    def __init__(self, num_zones: int = NUM_ZONES):
        super().__init__()
        self.num_zones = num_zones
        
        # EfficientNet-B2 backbone
        try:
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
        except:
            # Fallback to ResNet50
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                if 'features.7' not in name and 'features.8' not in name and 'layer4' not in name:
                    param.requires_grad = False
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_zones * 2)
        )
        
        # Replace backbone classifier
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)
    
    def predict_zones(self, image, threshold=0.5):
        """Predict S/R zones from image."""
        self.eval()
        with torch.no_grad():
            logits = self(image)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            return {
                'support_zones': [int(p > threshold) for p in probs[:self.num_zones]],
                'resistance_zones': [int(p > threshold) for p in probs[self.num_zones:]],
                'support_probs': probs[:self.num_zones].tolist(),
                'resistance_probs': probs[self.num_zones:].tolist()
            }


# =============================================================================
# TRAINING
# =============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y


class SRTrainerV3:
    """Advanced trainer with mixup and gradient accumulation."""
    
    def __init__(self, model, train_loader, val_loader, device='auto'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accumulation_steps = 2  # Effective batch size = 32 * 2 = 64
        
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
        
        # Focal loss
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Optimizer with different LR for backbone vs classifier
        backbone_params = [p for n, p in model.named_parameters() 
                         if 'classifier' not in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() 
                           if 'classifier' in n and p.requires_grad]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': classifier_params, 'lr': 5e-4}
        ], weight_decay=0.02)
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self, use_mixup=True):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc='Training', leave=False)):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Mixup
            if use_mixup and np.random.random() > 0.5:
                images, targets = mixup_data(images, targets, alpha=0.2)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets) / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_correct = 0
        all_total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Accuracy (per zone)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Compare with original labels (not smoothed)
                targets_binary = (targets > 0.5).float()
                
                correct = (preds == targets_binary).sum().item()
                total = targets.numel()
                
                all_correct += correct
                all_total += total
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = all_correct / all_total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int = 80, save_dir: str = 'checkpoints', target_acc: float = 0.80):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Use mixup only after warmup
            use_mixup = epoch >= 5
            
            train_loss = self.train_epoch(use_mixup=use_mixup)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            lr = self.optimizer.param_groups[1]['lr']
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.1%} | LR: {lr:.2e}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                }, save_path / 'sr_model_v3_best.pt')
                print(f"  ✅ New best model (accuracy: {val_acc:.1%})")
            else:
                patience_counter += 1
            
            if val_acc >= target_acc:
                print(f"\n🎯 TARGET ACCURACY {target_acc:.0%} REACHED!")
                break
            
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping")
                break
        
        self._plot_history(save_path)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Accuracy: {best_acc:.1%}")
        print(f"{'='*60}")
        
        return self.history, best_acc
    
    def _plot_history(self, save_dir):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['val_accuracy'])
        axes[1].axhline(y=0.8, color='r', linestyle='--', label='Target (80%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sr_v3_training_history.png', dpi=150)
        plt.close()


# =============================================================================
# VISUALIZATION - DRAW LINES ON CHART
# =============================================================================

def visualize_sr_prediction(
    image_path: str,
    model: SRModelV3,
    price_range: tuple = None,
    save_path: str = None,
    device: str = 'cpu'
):
    """
    Draw S/R lines on a chart image based on model predictions.
    
    Args:
        image_path: Path to chart image
        model: Trained SRModelV3
        price_range: (min_price, max_price) for labeling
        save_path: Where to save visualization
        device: torch device
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    result = model.predict_zones(img_tensor)
    
    # Load original image for drawing
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    num_zones = model.num_zones
    zone_height = height / num_zones
    
    # Draw support zones (green)
    for i, (is_sr, prob) in enumerate(zip(result['support_zones'], result['support_probs'])):
        if is_sr or prob > 0.3:
            y = height - (i + 0.5) * zone_height  # Invert Y (image coords)
            alpha = min(prob, 0.9)
            ax.axhline(y=y, color='#22c55e', linestyle='--', linewidth=3, alpha=alpha)
            
            # Add label
            if price_range:
                price = price_range[0] + (i + 0.5) / num_zones * (price_range[1] - price_range[0])
                label = f"S: ${price:.2f} ({prob:.0%})"
            else:
                label = f"Support Zone {i+1} ({prob:.0%})"
            
            ax.text(10, y - 5, label, color='#22c55e', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Draw resistance zones (red)
    for i, (is_sr, prob) in enumerate(zip(result['resistance_zones'], result['resistance_probs'])):
        if is_sr or prob > 0.3:
            y = height - (i + 0.5) * zone_height
            alpha = min(prob, 0.9)
            ax.axhline(y=y, color='#ef4444', linestyle='--', linewidth=3, alpha=alpha)
            
            if price_range:
                price = price_range[0] + (i + 0.5) / num_zones * (price_range[1] - price_range[0])
                label = f"R: ${price:.2f} ({prob:.0%})"
            else:
                label = f"Resistance Zone {i+1} ({prob:.0%})"
            
            ax.text(width - 200, y - 5, label, color='#ef4444', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    ax.set_title('S/R Zone Prediction\n🟢 Support | 🔴 Resistance', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train S/R Model V3 (Target: 80%)')
    parser.add_argument('--mode', choices=['generate', 'train', 'both', 'visualize'], default='both')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    parser.add_argument('--samples', type=int, default=1500, help='Training samples')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--data-dir', type=str, default='data/sr_training_v3')
    parser.add_argument('--image', type=str, help='Image path for visualization')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if args.mode == 'visualize':
        if not args.image:
            print("❌ Provide --image path for visualization")
            return
        
        model = SRModelV3(num_zones=NUM_ZONES)
        checkpoint = torch.load('checkpoints/sr_model_v3_best.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        visualize_sr_prediction(args.image, model, save_path='sr_visualization.png')
        return
    
    if args.mode in ['generate', 'both']:
        if not api_key:
            print("❌ Polygon API key required!")
            return
        
        print("\n" + "="*60)
        print("STEP 1: Generating Training Data")
        print("="*60)
        generate_training_data(api_key, args.data_dir, args.samples)
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training S/R Model V3 (Target: 80%)")
        print("="*60)
        
        train_dataset = SRDatasetV3(args.data_dir, split='train')
        val_dataset = SRDatasetV3(args.data_dir, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        model = SRModelV3(num_zones=NUM_ZONES)
        
        trainer = SRTrainerV3(model, train_loader, val_loader)
        history, best_acc = trainer.train(epochs=args.epochs, target_acc=0.80)
        
        if best_acc >= 0.80:
            print("\n✅ SUCCESS: Model achieved 80%+ accuracy!")
        else:
            print(f"\n⚠️ Model accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
