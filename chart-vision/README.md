# 📊 Chart Vision - Stock Chart Analysis with Computer Vision

Detect **trends** and **support/resistance levels** from stock chart images using deep learning.

## 🎯 What This Does

| Feature | Method | Accuracy Target |
|---------|--------|-----------------|
| **Trend Detection** (up/down/sideways) | CNN (ResNet18) | 75-85% |
| **Support/Resistance Zones** | Classical CV + YOLO | Visual detection |

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

### 3. Generate Training Data

This downloads stock data from Polygon.io and creates labeled chart images:

```bash
python utils/chart_generator.py --samples 100
```

This creates:
- `data/raw/*.png` - Chart images
- `data/raw/labels.json` - Trend labels (uptrend/downtrend/sideways)
- `data/raw/metadata.json` - S/R levels, price ranges

### 3. Train the Trend Classifier

```bash
python models/trend_classifier.py
```

Training takes ~30 minutes on a Mac M1/M2 or GPU.

Output:
- `checkpoints/best_model.pt` - Trained model
- `checkpoints/training_history.png` - Loss/accuracy curves
- `checkpoints/confusion_matrix.png` - Model performance

### 4. Analyze Charts!

```bash
# Analyze a chart image
python predict.py path/to/chart.png

# Save visualization
python predict.py path/to/chart.png --output results.png
```

## 📁 Project Structure

```
chart-vision/
├── data/
│   ├── raw/           # Generated chart images
│   ├── labeled/       # For YOLO training (manual labels)
│   └── processed/     # Processed datasets
├── models/
│   ├── trend_classifier.py   # CNN for trend detection
│   └── sr_detector.py        # S/R zone detection
├── utils/
│   └── chart_generator.py    # Auto-generate training data
├── checkpoints/       # Trained model weights
├── predict.py         # Inference script
└── requirements.txt
```

## 🧠 How It Works

### Trend Classification (CNN)

1. **ResNet18** pretrained on ImageNet
2. Fine-tuned on stock chart images
3. Outputs: `uptrend`, `downtrend`, `sideways` + confidence

```python
from models.trend_classifier import TrendClassifier
from PIL import Image
import torchvision.transforms as transforms

model = TrendClassifier()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])

image = Image.open('chart.png')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

result = model.predict(transform(image).unsqueeze(0))
print(result)
# {'prediction': 'uptrend', 'confidence': 0.87, 'probabilities': {...}}
```

### Support/Resistance Detection

**Classical Method** (no training needed):
1. Edge detection (Canny)
2. Hough line transform
3. Cluster horizontal lines
4. Filter by significance

**YOLO Method** (requires labeled data):
1. Label charts with S/R zones using LabelImg
2. Convert to YOLO format
3. Fine-tune YOLOv8

```python
from models.sr_detector import SupportResistanceDetector

detector = SupportResistanceDetector(method='classical')
results = detector.detect('chart.png')
print(results)
# {'support': [350.5, 342.1], 'resistance': [380.2, 395.0]}

# Visualize
detector.visualize('chart.png', save_path='sr_analysis.png')
```

## 🏋️ Training Your Own Model

### For Better Trend Detection

1. **More data**: Increase `num_samples_per_class` in `chart_generator.py`
2. **More epochs**: Increase epochs in `trend_classifier.py`
3. **Different tickers**: Add tickers to `ChartGenerator.TICKERS`

### For YOLO S/R Detection

1. **Label images manually** using LabelImg:
   ```bash
   pip install labelImg
   labelImg data/raw/
   ```
   
2. **Draw bounding boxes** around S/R zones

3. **Convert to YOLO format**:
   ```python
   from models.sr_detector import YOLODatasetConverter
   YOLODatasetConverter.convert_for_yolo(
       'data/raw/metadata.json',
       'data/raw',
       'data/yolo_sr'
   )
   ```

4. **Train YOLOv8**:
   ```bash
   yolo detect train data=data/yolo_sr/data.yaml model=yolov8n.pt epochs=50
   ```

## 📊 Expected Results

After training on 300+ images per class:

| Metric | Value |
|--------|-------|
| Trend Accuracy | 75-85% |
| Uptrend Recall | 80%+ |
| Downtrend Recall | 80%+ |
| Sideways Recall | 65-75% |

> **Note**: Sideways is hardest to detect because it's subjective.

## 🔧 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "CUDA out of memory"
Reduce batch size in `trend_classifier.py`:
```python
train_loader = DataLoader(..., batch_size=16)  # Lower from 32
```

### "mplfinance error"
```bash
pip install --upgrade mplfinance
```

### Classical S/R not detecting well
- Try different chart styles (candlestick vs line)
- Adjust Canny thresholds in `sr_detector.py`

## 🚀 Next Steps

1. **Integrate with QuantCademy**: Add real-time chart analysis to the Technical Analysis module
2. **Add pattern detection**: Detect head & shoulders, double tops, flags
3. **Paper trading**: Test signals on historical data
4. **API endpoint**: Wrap in FastAPI for web access

## 📚 References

- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [mplfinance](https://github.com/matplotlib/mplfinance)

---

Built for the QuantCademy project 🎓
