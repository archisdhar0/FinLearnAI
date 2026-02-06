"""
Inference Script - Analyze chart images
Usage: python predict.py path/to/chart.png
"""

import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.trend_classifier import TrendClassifier, ChartDataset
from models.sr_detector import SupportResistanceDetector


class ChartAnalyzer:
    """
    Complete chart analysis pipeline.
    Combines trend classification + S/R detection.
    """
    
    def __init__(
        self,
        trend_model_path: str = 'checkpoints/best_model.pt',
        sr_method: str = 'classical'
    ):
        # Load trend classifier
        self.trend_model = TrendClassifier(num_classes=3, pretrained=False)
        
        if Path(trend_model_path).exists():
            checkpoint = torch.load(trend_model_path, map_location='cpu')
            self.trend_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trend model (acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
        else:
            print(f"⚠️ No trained model found at {trend_model_path}")
            print("   Using untrained model. Run training first!")
        
        self.trend_model.eval()
        
        # Load S/R detector
        self.sr_detector = SupportResistanceDetector(method=sr_method)
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def analyze(self, image_path: str) -> dict:
        """
        Analyze a chart image.
        
        Returns:
            Dictionary with trend and S/R analysis
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Trend classification
        img_tensor = self.transform(image).unsqueeze(0)
        trend_result = self.trend_model.predict(img_tensor)
        
        # S/R detection
        sr_result = self.sr_detector.detect(image_path)
        
        return {
            'trend': trend_result,
            'support_resistance': sr_result
        }
    
    def visualize(self, image_path: str, save_path: str = None):
        """Visualize analysis results."""
        results = self.analyze(image_path)
        
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img_array)
        
        # Draw S/R levels
        height = img_array.shape[0]
        for y in results['support_resistance'].get('support', []):
            ax.axhline(y=y, color='#22c55e', linewidth=3, linestyle='--', alpha=0.8)
        
        for y in results['support_resistance'].get('resistance', []):
            ax.axhline(y=y, color='#ef4444', linewidth=3, linestyle='--', alpha=0.8)
        
        # Add analysis panel
        trend = results['trend']
        trend_color = {
            'uptrend': '#22c55e',
            'downtrend': '#ef4444',
            'sideways': '#eab308'
        }.get(trend['prediction'], '#6366f1')
        
        info_text = f"""
        📊 CHART ANALYSIS
        ─────────────────────
        🎯 Trend: {trend['prediction'].upper()}
        📈 Confidence: {trend['confidence']*100:.1f}%
        
        📉 Probabilities:
           Uptrend: {trend['probabilities']['uptrend']*100:.1f}%
           Downtrend: {trend['probabilities']['downtrend']*100:.1f}%
           Sideways: {trend['probabilities']['sideways']*100:.1f}%
        
        🟢 Support levels: {len(results['support_resistance'].get('support', []))}
        🔴 Resistance levels: {len(results['support_resistance'].get('resistance', []))}
        """
        
        # Add text box
        props = dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            color='white',
            bbox=props
        )
        
        # Title with trend
        ax.set_title(
            f"Trend: {trend['prediction'].upper()} ({trend['confidence']*100:.0f}% confident)",
            fontsize=16,
            color=trend_color,
            fontweight='bold'
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved to {save_path}")
        
        plt.show()
        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze stock chart images')
    parser.add_argument('image', type=str, help='Path to chart image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                        help='Save visualization to this path')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the visualization')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ChartAnalyzer(trend_model_path=args.model)
    
    # Analyze
    print(f"\n📊 Analyzing: {args.image}")
    print("=" * 50)
    
    results = analyzer.analyze(args.image)
    
    # Print results
    trend = results['trend']
    print(f"\n🎯 TREND: {trend['prediction'].upper()}")
    print(f"   Confidence: {trend['confidence']*100:.1f}%")
    print(f"\n📊 Probabilities:")
    for cls, prob in trend['probabilities'].items():
        bar = '█' * int(prob * 20)
        print(f"   {cls:10s}: {bar} {prob*100:.1f}%")
    
    sr = results['support_resistance']
    print(f"\n🟢 Support levels found: {len(sr.get('support', []))}")
    print(f"🔴 Resistance levels found: {len(sr.get('resistance', []))}")
    
    # Visualize
    if not args.no_display or args.output:
        analyzer.visualize(args.image, save_path=args.output)


if __name__ == "__main__":
    main()
