"""
Model Evaluation & Inference Script
Evaluates both S/R and Trend models, reports accuracy metrics.
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from train_sr_model import SRDetectorModel, sr_accuracy
from train_trend_model import TrendClassifierModel, TrendDataset


class ChartAnalyzer:
    """
    Combined analyzer using both S/R and Trend models.
    """
    
    def __init__(
        self,
        sr_model_path: str = 'checkpoints/sr_model_best.pt',
        trend_model_path: str = 'checkpoints/trend_model_best.pt',
        device: str = 'auto'
    ):
        # Device
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
        
        # Load models
        self.sr_model = None
        self.trend_model = None
        
        if Path(sr_model_path).exists():
            self.sr_model = SRDetectorModel(max_levels=5)
            checkpoint = torch.load(sr_model_path, map_location=self.device)
            self.sr_model.load_state_dict(checkpoint['model_state_dict'])
            self.sr_model.to(self.device)
            self.sr_model.eval()
            print(f"✅ Loaded S/R model (accuracy: {checkpoint.get('val_accuracy', 'N/A')})")
        else:
            print(f"⚠️ S/R model not found at {sr_model_path}")
        
        if Path(trend_model_path).exists():
            self.trend_model = TrendClassifierModel(num_classes=3)
            checkpoint = torch.load(trend_model_path, map_location=self.device)
            self.trend_model.load_state_dict(checkpoint['model_state_dict'])
            self.trend_model.to(self.device)
            self.trend_model.eval()
            print(f"✅ Loaded Trend model (accuracy: {checkpoint.get('val_accuracy', 'N/A')})")
        else:
            print(f"⚠️ Trend model not found at {trend_model_path}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze(self, image_path: str) -> dict:
        """
        Analyze a chart image for S/R levels and trend.
        
        Args:
            image_path: Path to chart image
        
        Returns:
            Dictionary with S/R levels and trend prediction
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        results = {'image': image_path}
        
        # S/R Detection
        if self.sr_model:
            with torch.no_grad():
                sr_output = self.sr_model(img_tensor)[0].cpu().numpy()
                
                max_levels = self.sr_model.max_levels
                support = [float(s) for s in sr_output[:max_levels] if s > 0.05]
                resistance = [float(r) for r in sr_output[max_levels:] if r > 0.05]
                
                results['support'] = sorted(support)
                results['resistance'] = sorted(resistance, reverse=True)
        
        # Trend Classification
        if self.trend_model:
            with torch.no_grad():
                logits = self.trend_model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probs)
                
                results['trend'] = TrendDataset.CLASSES[pred_idx]
                results['trend_confidence'] = float(probs[pred_idx])
                results['trend_probabilities'] = {
                    cls: float(probs[i])
                    for i, cls in enumerate(TrendDataset.CLASSES)
                }
        
        return results
    
    def visualize(self, image_path: str, results: dict = None, save_path: str = None):
        """
        Visualize analysis results on the chart.
        """
        if results is None:
            results = self.analyze(image_path)
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        
        # Draw S/R levels
        if 'support' in results:
            for level in results['support'][:3]:
                y = int((1 - level) * height * 0.8 + height * 0.1)  # Approximate mapping
                ax.axhline(y=y, color='#22c55e', linestyle='--', linewidth=2, alpha=0.8)
                ax.text(10, y-5, f'Support: {level:.2f}', color='#22c55e', fontsize=10,
                       fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        if 'resistance' in results:
            for level in results['resistance'][:3]:
                y = int((1 - level) * height * 0.8 + height * 0.1)
                ax.axhline(y=y, color='#ef4444', linestyle='--', linewidth=2, alpha=0.8)
                ax.text(10, y-5, f'Resistance: {level:.2f}', color='#ef4444', fontsize=10,
                       fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Title with trend
        trend = results.get('trend', 'Unknown')
        confidence = results.get('trend_confidence', 0)
        trend_color = {'uptrend': '#22c55e', 'downtrend': '#ef4444', 'sideways': '#f59e0b'}.get(trend, 'white')
        
        ax.set_title(
            f"Trend: {trend.upper()} ({confidence:.0%} confidence)\n"
            f"Support: {len(results.get('support', []))} levels | "
            f"Resistance: {len(results.get('resistance', []))} levels",
            fontsize=14, fontweight='bold', color=trend_color
        )
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved to {save_path}")
        
        plt.show()
        plt.close()
        
        return results


def evaluate_sr_model(
    model_path: str = 'checkpoints/sr_model_best.pt',
    data_dir: str = 'data/sr_training',
    tolerance: float = 0.05
):
    """
    Evaluate S/R model accuracy on test set.
    """
    print("\n" + "="*60)
    print("S/R MODEL EVALUATION")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = SRDetectorModel(max_levels=5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    labels_path = Path(data_dir) / 'sr_labels.json'
    if not labels_path.exists():
        print(f"❌ Labels not found: {labels_path}")
        return None
    
    with open(labels_path, 'r') as f:
        samples = json.load(f)
    
    # Use last 20% as test set
    np.random.seed(42)
    np.random.shuffle(samples)
    test_samples = samples[int(len(samples) * 0.8):]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    accuracies = []
    support_accs = []
    resistance_accs = []
    
    print(f"\nEvaluating on {len(test_samples)} test samples...")
    
    for sample in test_samples:
        img_path = Path(data_dir) / sample['filename']
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)[0].cpu().numpy()
        
        pred_support = [p for p in output[:5] if p > 0.05]
        pred_resist = [p for p in output[5:] if p > 0.05]
        true_support = sample['support']
        true_resist = sample['resistance']
        
        acc_s = sr_accuracy(pred_support, true_support, tolerance)
        acc_r = sr_accuracy(pred_resist, true_resist, tolerance)
        
        support_accs.append(acc_s)
        resistance_accs.append(acc_r)
        accuracies.append((acc_s + acc_r) / 2)
    
    overall_acc = np.mean(accuracies)
    support_acc = np.mean(support_accs)
    resistance_acc = np.mean(resistance_accs)
    
    print(f"\n📊 RESULTS:")
    print(f"   Overall Accuracy: {overall_acc:.1%}")
    print(f"   Support Accuracy: {support_acc:.1%}")
    print(f"   Resistance Accuracy: {resistance_acc:.1%}")
    print(f"   Tolerance: ±{tolerance:.0%} of price range")
    
    if overall_acc >= 0.80:
        print(f"\n✅ PASSED: Model achieves 80%+ accuracy!")
    else:
        print(f"\n⚠️ BELOW TARGET: {overall_acc:.1%} < 80%")
    
    return {
        'overall_accuracy': overall_acc,
        'support_accuracy': support_acc,
        'resistance_accuracy': resistance_acc,
        'tolerance': tolerance,
        'num_samples': len(test_samples)
    }


def evaluate_trend_model(
    model_path: str = 'checkpoints/trend_model_best.pt',
    data_dir: str = 'data/trend_training'
):
    """
    Evaluate Trend model accuracy on test set.
    """
    print("\n" + "="*60)
    print("TREND MODEL EVALUATION")
    print("="*60)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = TrendClassifierModel(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    labels_path = Path(data_dir) / 'trend_labels.json'
    if not labels_path.exists():
        print(f"❌ Labels not found: {labels_path}")
        return None
    
    with open(labels_path, 'r') as f:
        samples = json.load(f)
    
    np.random.seed(42)
    np.random.shuffle(samples)
    test_samples = samples[int(len(samples) * 0.8):]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    correct = 0
    total = 0
    class_correct = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
    class_total = {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
    
    print(f"\nEvaluating on {len(test_samples)} test samples...")
    
    for sample in test_samples:
        img_path = Path(data_dir) / sample['filename']
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
        
        pred_label = TrendDataset.CLASSES[pred_idx]
        true_label = sample['label']
        
        class_total[true_label] += 1
        total += 1
        
        if pred_label == true_label:
            correct += 1
            class_correct[true_label] += 1
    
    overall_acc = correct / total if total > 0 else 0
    
    print(f"\n📊 RESULTS:")
    print(f"   Overall Accuracy: {overall_acc:.1%}")
    print(f"\n   Per-class accuracy:")
    for cls in TrendDataset.CLASSES:
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls]
            print(f"     {cls}: {acc:.1%} ({class_correct[cls]}/{class_total[cls]})")
    
    if overall_acc >= 0.80:
        print(f"\n✅ PASSED: Model achieves 80%+ accuracy!")
    else:
        print(f"\n⚠️ BELOW TARGET: {overall_acc:.1%} < 80%")
    
    return {
        'overall_accuracy': overall_acc,
        'class_accuracy': {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            for cls in TrendDataset.CLASSES
        },
        'num_samples': total
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Chart Vision Models')
    parser.add_argument('--mode', choices=['sr', 'trend', 'both', 'demo'], default='both')
    parser.add_argument('--image', type=str, help='Path to chart image for demo')
    parser.add_argument('--sr-model', type=str, default='checkpoints/sr_model_best.pt')
    parser.add_argument('--trend-model', type=str, default='checkpoints/trend_model_best.pt')
    parser.add_argument('--sr-data', type=str, default='data/sr_training')
    parser.add_argument('--trend-data', type=str, default='data/trend_training')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        if not args.image:
            print("❌ Please provide --image path for demo mode")
            return
        
        analyzer = ChartAnalyzer(args.sr_model, args.trend_model)
        results = analyzer.analyze(args.image)
        
        print("\n📊 Analysis Results:")
        print(json.dumps(results, indent=2))
        
        analyzer.visualize(args.image)
    
    else:
        results = {}
        
        if args.mode in ['sr', 'both']:
            sr_results = evaluate_sr_model(args.sr_model, args.sr_data)
            if sr_results:
                results['sr'] = sr_results
        
        if args.mode in ['trend', 'both']:
            trend_results = evaluate_trend_model(args.trend_model, args.trend_data)
            if trend_results:
                results['trend'] = trend_results
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if 'sr' in results:
            acc = results['sr']['overall_accuracy']
            status = "✅ PASS" if acc >= 0.80 else "❌ FAIL"
            print(f"  S/R Model: {acc:.1%} {status}")
        
        if 'trend' in results:
            acc = results['trend']['overall_accuracy']
            status = "✅ PASS" if acc >= 0.80 else "❌ FAIL"
            print(f"  Trend Model: {acc:.1%} {status}")


if __name__ == "__main__":
    main()
