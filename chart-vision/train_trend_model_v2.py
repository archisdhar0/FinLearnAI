"""
Trend Classification Model V2
TARGET: 80%+ accuracy for uptrend vs downtrend vs sideways

Features:
1. EfficientNet-B2 backbone
2. Strong data augmentation
3. Focal loss for class balance
4. Draws trend line on chart for visualization
5. Outputs slope direction and confidence

The trend line is drawn based on linear regression of the predicted trend direction.
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


CLASSES = ['downtrend', 'sideways', 'uptrend']


# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLossMultiClass(nn.Module):
    """Focal Loss for multi-class classification."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# TREND CALCULATION
# =============================================================================

def calculate_trend(closes: np.ndarray, threshold: float = 0.12) -> dict:
    """
    Calculate trend from closing prices using linear regression.
    
    Returns:
        label: 'uptrend', 'downtrend', or 'sideways'
        slope: normalized slope value
        confidence: R-squared (how linear the trend is)
        line_points: (start_y, end_y) normalized for drawing
    """
    if len(closes) < 10:
        return {'label': 'sideways', 'slope': 0, 'confidence': 0, 'line_points': (0.5, 0.5)}
    
    # Linear regression
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # Normalize slope by price range
    price_min = np.min(closes)
    price_max = np.max(closes)
    price_range = price_max - price_min
    
    if price_range == 0:
        return {'label': 'sideways', 'slope': 0, 'confidence': 0, 'line_points': (0.5, 0.5)}
    
    # Normalized slope (% change per bar)
    avg_price = np.mean(closes)
    norm_slope = (slope / avg_price) * 100
    
    # R-squared for confidence
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((closes - y_pred) ** 2)
    ss_tot = np.sum((closes - np.mean(closes)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate line endpoints (normalized 0-1)
    start_price = intercept
    end_price = slope * (len(closes) - 1) + intercept
    
    start_y = (start_price - price_min) / price_range
    end_y = (end_price - price_min) / price_range
    
    # Clamp to valid range
    start_y = max(0, min(1, start_y))
    end_y = max(0, min(1, end_y))
    
    # Classify trend
    # Require both slope AND linearity for trend classification
    if norm_slope > threshold and r_squared > 0.25:
        label = 'uptrend'
    elif norm_slope < -threshold and r_squared > 0.25:
        label = 'downtrend'
    else:
        label = 'sideways'
    
    return {
        'label': label,
        'slope': float(norm_slope),
        'confidence': float(r_squared),
        'line_points': (float(start_y), float(end_y))
    }


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_trend_data(
    api_key: str,
    output_dir: str = "data/trend_training_v2",
    samples_per_class: int = 400,
    window_size: int = 60
):
    """Generate balanced trend classification dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fetcher = PolygonDataFetcher(api_key=api_key)
    
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'MA',
        'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'DIS', 'NFLX', 'ADBE', 'CRM', 'PYPL',
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'GLD', 'SLV', 'TLT',
        'AMD', 'INTC', 'BA', 'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'MCD', 'SBUX',
        'COST', 'NKE', 'LOW', 'TGT', 'BKNG', 'ABNB', 'UBER', 'SQ', 'ROKU', 'SNAP'
    ]
    
    samples = {cls: [] for cls in CLASSES}
    
    print(f"Generating {samples_per_class} samples per class...")
    print(f"Target total: {samples_per_class * 3} samples")
    
    for ticker in tickers:
        if all(len(v) >= samples_per_class for v in samples.values()):
            break
        
        print(f"\nFetching {ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 4)
        
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
        
        for start_idx in range(0, len(df) - window_size, window_size // 5):
            if all(len(v) >= samples_per_class for v in samples.values()):
                break
            
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx].copy()
            closes = window_df['Close'].values
            
            trend_data = calculate_trend(closes)
            label = trend_data['label']
            
            # Skip low-confidence samples for up/down trends
            if label != 'sideways' and trend_data['confidence'] < 0.3:
                continue
            
            # Skip if we have enough of this class
            if len(samples[label]) >= samples_per_class:
                continue
            
            idx = len(samples[label])
            filename = f"trend_{label}_{idx:04d}_{ticker}.png"
            filepath = output_path / filename
            
            if draw_candlestick_chart(window_df, str(filepath), figsize=(8, 6), dpi=100):
                sample = {
                    'filename': filename,
                    'ticker': ticker,
                    'label': label,
                    'slope': trend_data['slope'],
                    'confidence': trend_data['confidence'],
                    'line_points': trend_data['line_points'],
                    'start_date': str(window_df.index[0].date()),
                    'end_date': str(window_df.index[-1].date())
                }
                samples[label].append(sample)
                
                total = sum(len(v) for v in samples.values())
                if total % 100 == 0:
                    print(f"  Progress: {total}/{samples_per_class * 3}")
                    for cls in CLASSES:
                        print(f"    {cls}: {len(samples[cls])}/{samples_per_class}")
    
    # Flatten
    all_samples = []
    for label, sample_list in samples.items():
        all_samples.extend(sample_list)
    
    # Save
    with open(output_path / 'trend_labels.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n✅ Generated {len(all_samples)} samples")
    for cls in CLASSES:
        print(f"   {cls}: {len(samples[cls])}")
    
    return all_samples


# =============================================================================
# DATASET
# =============================================================================

class TrendDatasetV2(Dataset):
    """Dataset with strong augmentation for trend classification."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        
        with open(self.data_dir / 'trend_labels.json', 'r') as f:
            self.samples = json.load(f)
        
        np.random.seed(42)
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * 0.85)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.08))
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
        
        label = CLASSES.index(sample['label'])
        
        return image, label


# =============================================================================
# MODEL
# =============================================================================

class TrendModelV2(nn.Module):
    """
    EfficientNet-based trend classifier.
    
    Outputs:
    - Class probabilities (uptrend/downtrend/sideways)
    - Slope regression (for drawing the trend line)
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        # EfficientNet-B2 backbone
        try:
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            in_features = self.backbone.classifier[1].in_features
        except:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                if 'features.7' not in name and 'features.8' not in name and 'layer4' not in name:
                    param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Slope regression head (for drawing line)
        self.slope_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # start_y, end_y (normalized 0-1)
            nn.Sigmoid()
        )
        
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        class_logits = self.classifier(features)
        slope_pred = self.slope_head(features)
        
        return class_logits, slope_pred
    
    def predict(self, image: torch.Tensor) -> dict:
        """Predict trend and slope from image."""
        self.eval()
        with torch.no_grad():
            class_logits, slope_pred = self(image)
            
            probs = F.softmax(class_logits, dim=1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            
            slope = slope_pred[0].cpu().numpy()
            
            return {
                'prediction': CLASSES[pred_idx],
                'confidence': float(probs[pred_idx]),
                'probabilities': {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
                'line_start_y': float(slope[0]),
                'line_end_y': float(slope[1])
            }


# =============================================================================
# TRAINING
# =============================================================================

class TrendTrainerV2:
    """Trainer for trend classification model."""
    
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
        
        # Focal loss for classification
        self.class_criterion = FocalLossMultiClass(alpha=1.0, gamma=2.0)
        
        # Different LR for backbone vs heads
        backbone_params = [p for n, p in model.named_parameters() 
                         if 'classifier' not in n and 'slope_head' not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() 
                      if ('classifier' in n or 'slope_head' in n) and p.requires_grad]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 5e-4}
        ], weight_decay=0.02)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.train_loader, desc='Training', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            class_logits, _ = self.model(images)
            loss = self.class_criterion(class_logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = {cls: 0 for cls in CLASSES}
        class_total = {cls: 0 for cls in CLASSES}
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating', leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                class_logits, _ = self.model(images)
                loss = self.class_criterion(class_logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(class_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    true_cls = CLASSES[labels[i].item()]
                    pred_cls = CLASSES[predicted[i].item()]
                    class_total[true_cls] += 1
                    if true_cls == pred_cls:
                        class_correct[true_cls] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, class_correct, class_total
    
    def train(self, epochs: int = 60, save_dir: str = 'checkpoints', target_acc: float = 0.80):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, class_correct, class_total = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")
            print(f"  Per-class:")
            for cls in CLASSES:
                if class_total[cls] > 0:
                    acc = class_correct[cls] / class_total[cls]
                    print(f"    {cls}: {acc:.1%} ({class_correct[cls]}/{class_total[cls]})")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                }, save_path / 'trend_model_v2_best.pt')
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
        plt.savefig(save_dir / 'trend_v2_training_history.png', dpi=150)
        plt.close()


# =============================================================================
# VISUALIZATION - DRAW TREND LINE
# =============================================================================

def visualize_trend_prediction(
    image_path: str,
    model: TrendModelV2,
    save_path: str = None,
    device: str = 'cpu'
):
    """
    Draw trend line on chart based on model prediction.
    
    Args:
        image_path: Path to chart image
        model: Trained TrendModelV2
        save_path: Where to save visualization
        device: torch device
    
    Returns:
        Prediction dictionary
    """
    # Load and preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.to(device)
    result = model.predict(img_tensor)
    
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # Determine trend line endpoints
    # Line goes from left to right of chart
    margin_x = width * 0.05
    margin_y = height * 0.1
    
    start_x = margin_x
    end_x = width - margin_x
    
    # Y coordinates (inverted because image coords)
    chart_height = height - 2 * margin_y
    start_y = margin_y + chart_height * (1 - result['line_start_y'])
    end_y = margin_y + chart_height * (1 - result['line_end_y'])
    
    # Color based on trend
    trend = result['prediction']
    if trend == 'uptrend':
        color = '#22c55e'  # Green
        arrow = '↗'
    elif trend == 'downtrend':
        color = '#ef4444'  # Red
        arrow = '↘'
    else:
        color = '#f59e0b'  # Orange/Yellow
        arrow = '→'
    
    # Draw trend line
    ax.plot([start_x, end_x], [start_y, end_y], 
            color=color, linewidth=4, linestyle='-', alpha=0.9)
    
    # Add arrow at the end
    arrow_size = 20
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/length * arrow_size, dy/length * arrow_size
    
    ax.annotate('', xy=(end_x, end_y), xytext=(end_x - dx*2, end_y - dy*2),
                arrowprops=dict(arrowstyle='->', color=color, lw=3))
    
    # Add label
    confidence = result['confidence']
    label = f"{trend.upper()} {arrow}\n{confidence:.0%} confidence"
    
    # Position label
    label_x = width * 0.02
    label_y = height * 0.05
    
    ax.text(label_x, label_y, label, fontsize=16, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7),
            verticalalignment='top')
    
    # Add probability bars
    bar_x = width * 0.02
    bar_y = height * 0.85
    bar_width = width * 0.15
    bar_height = 15
    
    for i, cls in enumerate(CLASSES):
        prob = result['probabilities'][cls]
        y = bar_y + i * (bar_height + 5)
        
        # Background bar
        ax.add_patch(plt.Rectangle((bar_x, y), bar_width, bar_height, 
                                   facecolor='gray', alpha=0.3))
        # Filled bar
        cls_color = {'uptrend': '#22c55e', 'downtrend': '#ef4444', 'sideways': '#f59e0b'}[cls]
        ax.add_patch(plt.Rectangle((bar_x, y), bar_width * prob, bar_height,
                                   facecolor=cls_color, alpha=0.8))
        # Label
        ax.text(bar_x + bar_width + 5, y + bar_height/2, f"{cls}: {prob:.0%}",
               fontsize=10, color='white', verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax.set_title(f'Trend Analysis: {trend.upper()}', fontsize=16, fontweight='bold', color=color)
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
    
    parser = argparse.ArgumentParser(description='Train Trend Model V2 (Target: 80%)')
    parser.add_argument('--mode', choices=['generate', 'train', 'both', 'visualize'], default='both')
    parser.add_argument('--api-key', type=str, help='Polygon API key')
    parser.add_argument('--samples', type=int, default=400, help='Samples per class')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--data-dir', type=str, default='data/trend_training_v2')
    parser.add_argument('--image', type=str, help='Image path for visualization')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    
    if args.mode == 'visualize':
        if not args.image:
            print("❌ Provide --image path")
            return
        
        model = TrendModelV2()
        checkpoint = torch.load('checkpoints/trend_model_v2_best.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        result = visualize_trend_prediction(args.image, model, save_path='trend_visualization.png')
        print(f"\nPrediction: {result['prediction']} ({result['confidence']:.0%})")
        return
    
    if args.mode in ['generate', 'both']:
        if not api_key:
            print("❌ Polygon API key required!")
            return
        
        print("\n" + "="*60)
        print("STEP 1: Generating Training Data")
        print("="*60)
        generate_trend_data(api_key, args.data_dir, args.samples)
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training Trend Model V2 (Target: 80%)")
        print("="*60)
        
        train_dataset = TrendDatasetV2(args.data_dir, split='train')
        val_dataset = TrendDatasetV2(args.data_dir, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        model = TrendModelV2()
        
        trainer = TrendTrainerV2(model, train_loader, val_loader)
        history, best_acc = trainer.train(epochs=args.epochs, target_acc=0.80)
        
        if best_acc >= 0.80:
            print("\n✅ SUCCESS: Model achieved 80%+ accuracy!")
        else:
            print(f"\n⚠️ Model accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
