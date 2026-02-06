"""
Support/Resistance Detector - Multiple Methods
1. Price-based (BEST): Uses actual OHLC data to find S/R levels
2. Image-based: Traces the price line in the image and finds clusters
"""

import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class SupportResistanceDetector:
    """
    Detect support and resistance levels.
    
    Methods:
    - 'price': Uses OHLC price data (most accurate)
    - 'image': Uses computer vision on chart image
    """
    
    def __init__(self, method: str = 'price'):
        self.method = method
    
    # ==================== PRICE-BASED METHOD (RECOMMENDED) ====================
    
    def detect_from_prices(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray,
        window: int = 5,
        num_levels: int = 3
    ) -> dict:
        """
        Find S/R from OHLC price data.
        
        This is the most accurate method!
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            window: Lookback for local min/max detection
            num_levels: Max number of S/R levels to return
        
        Returns:
            Dictionary with support and resistance price levels
        """
        # Find local maxima (resistance) - prices that were local highs
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
        
        # Find local minima (support) - prices that were local lows
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
        
        # Cluster nearby levels (within 2% of each other)
        def cluster_levels(levels, threshold_pct=0.02):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                # If this level is close to the cluster, add it
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                    current_cluster.append(level)
                else:
                    # Save cluster (average) and start new one
                    clusters.append({
                        'price': np.mean(current_cluster),
                        'touches': len(current_cluster),
                        'strength': len(current_cluster)  # More touches = stronger
                    })
                    current_cluster = [level]
            
            # Don't forget the last cluster
            clusters.append({
                'price': np.mean(current_cluster),
                'touches': len(current_cluster),
                'strength': len(current_cluster)
            })
            
            # Sort by strength (number of touches)
            clusters.sort(key=lambda x: x['strength'], reverse=True)
            
            return clusters[:num_levels]
        
        support_clusters = cluster_levels(support_levels)
        resistance_clusters = cluster_levels(resistance_levels)
        
        # Determine current price position relative to S/R
        current_price = closes[-1]
        
        return {
            'support': support_clusters,
            'resistance': resistance_clusters,
            'current_price': current_price,
            'nearest_support': min(support_clusters, key=lambda x: abs(x['price'] - current_price)) if support_clusters else None,
            'nearest_resistance': min(resistance_clusters, key=lambda x: abs(x['price'] - current_price)) if resistance_clusters else None,
        }
    
    # ==================== IMAGE-BASED METHOD ====================
    
    def detect_from_image(self, image_path: str) -> dict:
        """
        Detect S/R from chart image using computer vision.
        
        Approach:
        1. Convert to grayscale
        2. Find the price line (candlestick bodies/wicks)
        3. Create a horizontal histogram of price levels
        4. Find peaks in the histogram (frequently visited prices = S/R)
        
        Args:
            image_path: Path to chart image
        
        Returns:
            Dictionary with S/R levels as y-coordinates in image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image', 'support': [], 'resistance': []}
        
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find candlesticks (they're colored, background is dark)
        # This finds the green and red candles
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect green candles (up)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Detect red candles (down)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Combine masks
        candle_mask = green_mask | red_mask
        
        # Create horizontal histogram (sum pixels in each row)
        # This tells us which y-coordinates have the most price activity
        row_sums = np.sum(candle_mask, axis=1)
        
        # Smooth the histogram
        from scipy.ndimage import gaussian_filter1d
        try:
            row_sums_smooth = gaussian_filter1d(row_sums.astype(float), sigma=5)
        except:
            # Fallback without scipy
            kernel_size = 11
            kernel = np.ones(kernel_size) / kernel_size
            row_sums_smooth = np.convolve(row_sums.astype(float), kernel, mode='same')
        
        # Find peaks (y-coordinates with lots of price activity)
        peaks = self._find_peaks(row_sums_smooth, min_distance=height // 10)
        
        # Classify peaks as support or resistance based on position
        mid_y = height / 2
        support = [{'y': int(p), 'strength': row_sums_smooth[p]} for p in peaks if p > mid_y]
        resistance = [{'y': int(p), 'strength': row_sums_smooth[p]} for p in peaks if p <= mid_y]
        
        # Sort by strength
        support.sort(key=lambda x: x['strength'], reverse=True)
        resistance.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support': support[:3],
            'resistance': resistance[:3],
            'all_peaks': peaks.tolist() if hasattr(peaks, 'tolist') else list(peaks),
            'histogram': row_sums_smooth.tolist(),
            'image_shape': (height, width)
        }
    
    def _find_peaks(self, arr: np.ndarray, min_distance: int = 20) -> np.ndarray:
        """Find local maxima in array."""
        peaks = []
        for i in range(min_distance, len(arr) - min_distance):
            window = arr[i - min_distance:i + min_distance + 1]
            if arr[i] == max(window) and arr[i] > np.mean(arr):
                peaks.append(i)
        return np.array(peaks)
    
    # ==================== MAIN INTERFACE ====================
    
    def detect(self, image_path: str = None, price_data: dict = None) -> dict:
        """
        Main detection method.
        
        Args:
            image_path: Path to chart image (for image-based detection)
            price_data: Dictionary with 'High', 'Low', 'Close' arrays
        
        Returns:
            S/R detection results
        """
        if price_data is not None:
            return self.detect_from_prices(
                highs=np.array(price_data['High']),
                lows=np.array(price_data['Low']),
                closes=np.array(price_data['Close'])
            )
        elif image_path is not None:
            return self.detect_from_image(image_path)
        else:
            return {'error': 'Provide either image_path or price_data'}
    
    def visualize(
        self, 
        image_path: str, 
        results: dict = None,
        save_path: str = None,
        price_range: tuple = None
    ):
        """
        Visualize S/R levels on the chart.
        
        Args:
            image_path: Path to chart image
            results: Detection results (if None, will detect from image)
            save_path: Path to save visualization
            price_range: (min_price, max_price) for converting y to price
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Get detection results
        if results is None:
            results = self.detect_from_image(image_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.imshow(img)
        
        # Draw support lines (green, dashed)
        for level in results.get('support', []):
            y = level.get('y', level) if isinstance(level, dict) else level
            strength = level.get('strength', 1) if isinstance(level, dict) else 1
            alpha = min(0.9, 0.3 + strength / 1000)
            ax.axhline(y=y, color='#22c55e', linewidth=3, linestyle='--', alpha=alpha)
            ax.text(width - 10, y - 10, 'SUPPORT', color='#22c55e', fontsize=12, 
                   fontweight='bold', ha='right')
        
        # Draw resistance lines (red, dashed)
        for level in results.get('resistance', []):
            y = level.get('y', level) if isinstance(level, dict) else level
            strength = level.get('strength', 1) if isinstance(level, dict) else 1
            alpha = min(0.9, 0.3 + strength / 1000)
            ax.axhline(y=y, color='#ef4444', linewidth=3, linestyle='--', alpha=alpha)
            ax.text(width - 10, y - 10, 'RESISTANCE', color='#ef4444', fontsize=12,
                   fontweight='bold', ha='right')
        
        # Title
        n_support = len(results.get('support', []))
        n_resistance = len(results.get('resistance', []))
        ax.set_title(f'Support/Resistance Detection\n'
                    f'🟢 {n_support} Support | 🔴 {n_resistance} Resistance',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved to {save_path}")
        
        plt.show()
        plt.close()
        
        return results


def demo_with_price_data():
    """
    Demo using price data (the recommended approach).
    """
    print("=" * 60)
    print("📊 S/R DETECTION DEMO (Price-Based)")
    print("=" * 60)
    
    # Simulated price data (in real use, load from your data)
    np.random.seed(42)
    n = 60
    
    # Generate realistic-ish price movement
    base_price = 150
    trend = np.linspace(0, 20, n)  # Uptrend
    noise = np.cumsum(np.random.randn(n) * 2)
    closes = base_price + trend + noise
    
    # Add some consolidation periods (creates S/R)
    closes[15:25] = closes[15:25] * 0 + 160  # Consolidation at 160
    closes[40:50] = closes[40:50] * 0 + 175  # Consolidation at 175
    
    highs = closes + np.abs(np.random.randn(n) * 2)
    lows = closes - np.abs(np.random.randn(n) * 2)
    
    # Detect S/R
    detector = SupportResistanceDetector()
    results = detector.detect_from_prices(highs, lows, closes)
    
    print("\n🟢 SUPPORT LEVELS:")
    for level in results['support']:
        print(f"   ${level['price']:.2f} (touched {level['touches']}x)")
    
    print("\n🔴 RESISTANCE LEVELS:")
    for level in results['resistance']:
        print(f"   ${level['price']:.2f} (touched {level['touches']}x)")
    
    print(f"\n📍 Current price: ${results['current_price']:.2f}")
    
    if results['nearest_support']:
        print(f"   Nearest support: ${results['nearest_support']['price']:.2f}")
    if results['nearest_resistance']:
        print(f"   Nearest resistance: ${results['nearest_resistance']['price']:.2f}")
    
    return results


if __name__ == "__main__":
    demo_with_price_data()
