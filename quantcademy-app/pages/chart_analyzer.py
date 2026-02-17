"""
Chart Analyzer - Upload a chart image and get S/R levels + Trend analysis
Uses trained CNN models from chart-vision project
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add chart-vision to path
CHART_VISION_PATH = Path(__file__).parent.parent.parent / "chart-vision"
sys.path.insert(0, str(CHART_VISION_PATH))

# Try to import models
MODELS_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Import model classes
    from train_sr_model_v2 import SRZoneModel
    from train_trend_model_v2 import TrendModelV2, CLASSES as TREND_CLASSES
    
    MODELS_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)

# Number of zones (must match training)
NUM_ZONES = 10  # V2 uses 10 zones


def load_models():
    """Load trained models from checkpoints."""
    models = {}
    
    # S/R Model
    sr_path = CHART_VISION_PATH / "checkpoints" / "sr_zone_model_best.pt"
    if sr_path.exists():
        try:
            model = SRZoneModel(num_zones=NUM_ZONES)
            checkpoint = torch.load(sr_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models['sr'] = {
                'model': model,
                'accuracy': checkpoint.get('val_accuracy', 0)
            }
        except Exception as e:
            st.warning(f"Could not load S/R model: {e}")
    
    # Trend Model
    trend_path = CHART_VISION_PATH / "checkpoints" / "trend_model_v2_best.pt"
    if trend_path.exists():
        try:
            model = TrendModelV2()
            checkpoint = torch.load(trend_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models['trend'] = {
                'model': model,
                'accuracy': checkpoint.get('val_accuracy', 0)
            }
        except Exception as e:
            st.warning(f"Could not load Trend model: {e}")
    
    return models


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict_sr_zones(model, image_tensor) -> dict:
    """Predict S/R zones from image."""
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        num_zones = len(probs) // 2
        support_probs = probs[:num_zones]
        resistance_probs = probs[num_zones:]
        
        return {
            'support_zones': [int(p > 0.5) for p in support_probs],
            'resistance_zones': [int(p > 0.5) for p in resistance_probs],
            'support_probs': support_probs.tolist(),
            'resistance_probs': resistance_probs.tolist()
        }


def predict_trend(model, image_tensor) -> dict:
    """Predict trend from image."""
    with torch.no_grad():
        class_logits, slope_pred = model(image_tensor)
        
        probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)
        slope = slope_pred[0].cpu().numpy()
        
        return {
            'prediction': TREND_CLASSES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {cls: float(probs[i]) for i, cls in enumerate(TREND_CLASSES)},
            'line_start_y': float(slope[0]),
            'line_end_y': float(slope[1])
        }


def create_analysis_image(original_image: Image.Image, sr_result: dict, trend_result: dict) -> Image.Image:
    """Create annotated image with S/R lines and trend."""
    # Convert to numpy for matplotlib
    img_array = np.array(original_image)
    height, width = img_array.shape[:2]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array)
    
    # Draw S/R zones
    num_zones = len(sr_result['support_probs'])
    zone_height = height / num_zones
    
    # Support zones (green)
    for i, (is_sr, prob) in enumerate(zip(sr_result['support_zones'], sr_result['support_probs'])):
        if is_sr or prob > 0.3:
            y = height - (i + 0.5) * zone_height
            alpha = min(prob, 0.9)
            ax.axhline(y=y, color='#22c55e', linestyle='--', linewidth=2, alpha=alpha)
            ax.text(10, y - 5, f"Support {i+1} ({prob:.0%})", color='#22c55e', fontsize=9,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Resistance zones (red)
    for i, (is_sr, prob) in enumerate(zip(sr_result['resistance_zones'], sr_result['resistance_probs'])):
        if is_sr or prob > 0.3:
            y = height - (i + 0.5) * zone_height
            alpha = min(prob, 0.9)
            ax.axhline(y=y, color='#ef4444', linestyle='--', linewidth=2, alpha=alpha)
            ax.text(width - 150, y - 5, f"Resist {i+1} ({prob:.0%})", color='#ef4444', fontsize=9,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Draw trend line
    if trend_result:
        margin_x = width * 0.05
        margin_y = height * 0.1
        chart_height = height - 2 * margin_y
        
        start_x = margin_x
        end_x = width - margin_x
        start_y = margin_y + chart_height * (1 - trend_result['line_start_y'])
        end_y = margin_y + chart_height * (1 - trend_result['line_end_y'])
        
        trend = trend_result['prediction']
        if trend == 'uptrend':
            color = '#22c55e'
            arrow = '↗'
        elif trend == 'downtrend':
            color = '#ef4444'
            arrow = '↘'
        else:
            color = '#f59e0b'
            arrow = '→'
        
        ax.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=4, alpha=0.8)
        
        # Trend label
        confidence = trend_result['confidence']
        label = f"{trend.upper()} {arrow} ({confidence:.0%})"
        ax.text(width/2, 30, label, fontsize=16, fontweight='bold', color=color,
               ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def main():
    st.set_page_config(
        page_title="Chart Analyzer | QuantCademy",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0;">📊 Chart Analyzer</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Upload any candlestick chart image to detect Support/Resistance levels and Trend direction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODELS_AVAILABLE:
        st.error(f"""
        **Models not available.** 
        
        Make sure you have:
        1. Trained the models in `chart-vision/`
        2. Installed required packages: `pip install torch torchvision`
        
        Error: {IMPORT_ERROR if 'IMPORT_ERROR' in dir() else 'Unknown'}
        """)
        return
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if not models:
        st.warning("""
        **No trained models found.** 
        
        Train the models first:
        ```bash
        cd chart-vision
        python train_sr_model_v2.py --mode both --samples 800
        python train_trend_model_v2.py --mode both --samples 400
        ```
        """)
        return
    
    # Show model status
    col1, col2 = st.columns(2)
    with col1:
        if 'sr' in models:
            acc = models['sr']['accuracy']
            st.success(f"✅ S/R Model loaded ({acc:.1%} accuracy)")
        else:
            st.warning("⚠️ S/R Model not found")
    
    with col2:
        if 'trend' in models:
            acc = models['trend']['accuracy']
            st.success(f"✅ Trend Model loaded ({acc:.1%} accuracy)")
        else:
            st.warning("⚠️ Trend Model not found")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a candlestick chart image",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Works best with clean candlestick charts (dark background, green/red candles)"
    )
    
    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Chart")
            st.image(image, use_container_width=True)
        
        # Run analysis
        with st.spinner("Analyzing chart..."):
            img_tensor = preprocess_image(image)
            
            sr_result = None
            trend_result = None
            
            if 'sr' in models:
                sr_result = predict_sr_zones(models['sr']['model'], img_tensor)
            
            if 'trend' in models:
                trend_result = predict_trend(models['trend']['model'], img_tensor)
            
            # Create annotated image
            if sr_result or trend_result:
                annotated = create_analysis_image(
                    image, 
                    sr_result or {'support_zones': [], 'resistance_zones': [], 
                                 'support_probs': [], 'resistance_probs': []},
                    trend_result
                )
        
        with col2:
            st.markdown("### Analysis Results")
            if sr_result or trend_result:
                st.image(annotated, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Trend Analysis")
            if trend_result:
                trend = trend_result['prediction']
                conf = trend_result['confidence']
                
                if trend == 'uptrend':
                    st.success(f"**{trend.upper()}** ↗ ({conf:.0%} confidence)")
                elif trend == 'downtrend':
                    st.error(f"**{trend.upper()}** ↘ ({conf:.0%} confidence)")
                else:
                    st.warning(f"**{trend.upper()}** → ({conf:.0%} confidence)")
                
                st.markdown("**Probability breakdown:**")
                for cls, prob in trend_result['probabilities'].items():
                    color = {'uptrend': '🟢', 'downtrend': '🔴', 'sideways': '🟡'}[cls]
                    bar_width = int(prob * 100)
                    st.markdown(f"{color} {cls}: {'█' * (bar_width // 5)}{'░' * (20 - bar_width // 5)} {prob:.0%}")
            else:
                st.info("Trend model not available")
        
        with col2:
            st.markdown("### 🎯 Support & Resistance")
            if sr_result:
                # Support levels
                st.markdown("**Support Zones (🟢 buying pressure):**")
                support_found = False
                for i, (is_sr, prob) in enumerate(zip(sr_result['support_zones'], sr_result['support_probs'])):
                    if is_sr or prob > 0.4:
                        support_found = True
                        strength = "Strong" if prob > 0.6 else "Moderate" if prob > 0.4 else "Weak"
                        st.markdown(f"- Zone {i+1}: {prob:.0%} ({strength})")
                if not support_found:
                    st.markdown("- No significant support detected")
                
                st.markdown("")
                
                # Resistance levels
                st.markdown("**Resistance Zones (🔴 selling pressure):**")
                resist_found = False
                for i, (is_sr, prob) in enumerate(zip(sr_result['resistance_zones'], sr_result['resistance_probs'])):
                    if is_sr or prob > 0.4:
                        resist_found = True
                        strength = "Strong" if prob > 0.6 else "Moderate" if prob > 0.4 else "Weak"
                        st.markdown(f"- Zone {i+1}: {prob:.0%} ({strength})")
                if not resist_found:
                    st.markdown("- No significant resistance detected")
            else:
                st.info("S/R model not available")
        
        st.markdown("---")
        
        # Download button
        if sr_result or trend_result:
            buf = io.BytesIO()
            annotated.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Analyzed Chart",
                data=buf,
                file_name="chart_analysis.png",
                mime="image/png"
            )
        
        # Explanation
        with st.expander("ℹ️ How to interpret results"):
            st.markdown("""
            ### Trend Analysis
            - **Uptrend ↗**: Prices are generally rising. The green line shows the overall direction.
            - **Downtrend ↘**: Prices are generally falling. The red line shows the decline.
            - **Sideways →**: No clear direction. Price is consolidating.
            
            ### Support & Resistance
            - **Support (Green lines)**: Price levels where buying pressure tends to prevent further decline.
              Think of it as a "floor" where buyers step in.
            - **Resistance (Red lines)**: Price levels where selling pressure tends to prevent further rise.
              Think of it as a "ceiling" where sellers take profits.
            
            ### Confidence Scores
            - **>60%**: Strong signal - high confidence in the prediction
            - **40-60%**: Moderate signal - some evidence but not conclusive
            - **<40%**: Weak signal - low confidence, use with caution
            
            ### Tips
            - Works best with clean candlestick charts (dark background)
            - Higher timeframe charts (daily, weekly) tend to have clearer S/R levels
            - Always combine with other analysis - this is a tool, not financial advice!
            """)
    
    else:
        # Show example/instructions
        st.markdown("""
        ### How to use:
        1. **Upload** any candlestick chart image (screenshot from TradingView, etc.)
        2. **Wait** for the AI to analyze the chart
        3. **Review** the detected support/resistance levels and trend direction
        4. **Download** the annotated chart if needed
        
        ### Best results with:
        - Clean candlestick charts (not line charts)
        - Dark background with green/red candles
        - Charts showing 30-90 days of price action
        - No overlays or indicators cluttering the chart
        """)
        
        # Sample images if available
        sample_dir = CHART_VISION_PATH / "data" / "demo"
        if sample_dir.exists():
            samples = list(sample_dir.glob("*.png"))[:3]
            if samples:
                st.markdown("### Try with sample charts:")
                cols = st.columns(len(samples))
                for i, sample in enumerate(samples):
                    with cols[i]:
                        img = Image.open(sample)
                        st.image(img, caption=sample.name, use_container_width=True)
                        if st.button(f"Analyze", key=f"sample_{i}"):
                            st.session_state['sample_image'] = sample
                            st.rerun()


if __name__ == "__main__":
    main()
