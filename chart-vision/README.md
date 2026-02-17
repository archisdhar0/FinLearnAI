# 📊 Chart Vision - Stock Chart Analysis with Deep Learning

Detect **trends** and **support/resistance levels** from stock chart images using CNN models.

## 🎯 Models & Accuracy Targets

| Model | Task | Target Accuracy |
|-------|------|-----------------|
| **S/R Detector** | Predict support/resistance price levels | **80%+** |
| **Trend Classifier** | Classify uptrend/downtrend/sideways | **80%+** |

Both models are trained on real stock data from Polygon.io and evaluated against ground truth derived from price data.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd chart-vision
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Your Polygon API Key

Get a free API key at [polygon.io](https://polygon.io/)

```bash
export POLYGON_API_KEY=your_api_key_here
```

Or pass it directly to commands with `--api-key YOUR_KEY`

---

## 🏋️ Training Models

### Train S/R Detection Model (Target: 80%+ Accuracy)

```bash
# Generate data + train (recommended)
python train_sr_model.py --mode both --samples 500 --epochs 30

# Or separately:
python train_sr_model.py --mode generate --samples 500
python train_sr_model.py --mode train --epochs 30
```

**What it does:**
1. Downloads stock data for 17+ tickers from Polygon
2. Generates chart images with ground truth S/R levels
3. Trains CNN to predict S/R from images
4. Stops early if 80% accuracy is reached

**Output:**
- `checkpoints/sr_model_best.pt` - Trained model
- `checkpoints/sr_training_history.png` - Training curves
- `data/sr_training/` - Training images and labels

### Train Trend Classifier (Target: 80%+ Accuracy)

```bash
# Generate data + train (recommended)
python train_trend_model.py --mode both --samples 200 --epochs 30

# Or separately:
python train_trend_model.py --mode generate --samples 200
python train_trend_model.py --mode train --epochs 30
```

**What it does:**
1. Downloads stock data and generates balanced dataset
2. Labels each chart with uptrend/downtrend/sideways using linear regression
3. Trains EfficientNet-B0 classifier
4. Stops early if 80% accuracy is reached

**Output:**
- `checkpoints/trend_model_best.pt` - Trained model
- `checkpoints/trend_training_history.png` - Training curves
- `checkpoints/trend_confusion_matrix.png` - Per-class accuracy
- `data/trend_training/` - Training images and labels

---

## 📊 Evaluate Models

```bash
# Evaluate both models
python evaluate_models.py --mode both

# Evaluate S/R model only
python evaluate_models.py --mode sr

# Evaluate Trend model only
python evaluate_models.py --mode trend
```

**Expected output:**
```
S/R MODEL EVALUATION
====================
Overall Accuracy: 82.3%
Support Accuracy: 85.1%
Resistance Accuracy: 79.5%
✅ PASSED: Model achieves 80%+ accuracy!

TREND MODEL EVALUATION
======================
Overall Accuracy: 84.7%
  uptrend: 88.2%
  downtrend: 85.3%
  sideways: 80.6%
✅ PASSED: Model achieves 80%+ accuracy!
```

---

## 🔮 Analyze New Charts

### Demo Mode

```bash
# Analyze a chart image
python evaluate_models.py --mode demo --image path/to/chart.png
```

### In Python

```python
from evaluate_models import ChartAnalyzer

analyzer = ChartAnalyzer()
results = analyzer.analyze('chart.png')

print(results)
# {
#   'support': [0.23, 0.31],      # Normalized 0-1
#   'resistance': [0.78, 0.85],
#   'trend': 'uptrend',
#   'trend_confidence': 0.92
# }

# Visualize with S/R lines and trend label
analyzer.visualize('chart.png', save_path='analysis.png')
```

---

## 📁 Project Structure

```
chart-vision/
├── train_sr_model.py       # S/R detection training pipeline
├── train_trend_model.py    # Trend classifier training pipeline
├── evaluate_models.py      # Evaluation & inference
├── demo.py                 # Compare price-based vs image-based S/R
├── models/
│   ├── sr_detector.py      # S/R detection (price-based + CV)
│   └── trend_classifier.py # Original trend classifier
├── utils/
│   └── chart_generator.py  # Polygon data fetcher + chart drawing
├── data/
│   ├── sr_training/        # S/R training data
│   ├── trend_training/     # Trend training data
│   └── demo/               # Demo images
├── checkpoints/            # Trained model weights
└── requirements.txt
```

---

## 🧠 How It Works

### S/R Detection Model

1. **Ground Truth**: Local minima (support) and maxima (resistance) from price data
2. **Architecture**: ResNet18 backbone → regression head
3. **Output**: 5 support + 5 resistance levels (normalized 0-1)
4. **Accuracy Metric**: % of predictions within 5% of a ground truth level

### Trend Classifier

1. **Ground Truth**: Linear regression slope of closing prices
   - Uptrend: positive slope > threshold
   - Downtrend: negative slope < -threshold
   - Sideways: slope near zero or high volatility
2. **Architecture**: EfficientNet-B0 → 3-class classifier
3. **Output**: Class probabilities + confidence

---

## ⚙️ Configuration

### Training Parameters

| Parameter | S/R Model | Trend Model |
|-----------|-----------|-------------|
| `--samples` | 500 | 200 per class |
| `--epochs` | 30 | 30 |
| `--data-dir` | data/sr_training | data/trend_training |
| Batch size | 32 | 32 |
| Learning rate | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |

### Accuracy Tolerance

- **S/R Model**: Prediction within 5% of price range counts as correct
- **Trend Model**: Exact class match required

---

## 🔧 Troubleshooting

### "Polygon API key required"
```bash
export POLYGON_API_KEY=your_key_here
# Or use --api-key YOUR_KEY
```

### "CUDA out of memory"
Reduce batch size in the training scripts (edit the DataLoader calls)

### Model accuracy below 80%
- Generate more training data (`--samples 1000`)
- Train for more epochs (`--epochs 50`)
- Check data quality (are labels correct?)

### "Module not found"
```bash
pip install -r requirements.txt
```

---

## 📈 Tips for Better Accuracy

1. **More diverse data**: Add more tickers to the training set
2. **Longer time windows**: Try 90-day windows instead of 60
3. **Data augmentation**: The trend model already uses augmentation
4. **Ensemble**: Train multiple models and average predictions
5. **Threshold tuning**: Adjust the slope threshold for trend classification

---

## 🚀 Next Steps

1. **Real-time analysis**: Integrate with live market data
2. **Pattern detection**: Add head & shoulders, double tops, flags
3. **API endpoint**: Wrap in FastAPI for web access
4. **QuantCademy integration**: Add to Technical Analysis module

---

Built for the QuantCademy project 🎓
