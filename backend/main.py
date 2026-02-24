"""
FastAPI Backend for FinLearn AI
Connects React frontend to RAG, CV models, and Polygon API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import base64
import io

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "quantcademy-app"))
sys.path.insert(0, str(Path(__file__).parent.parent / "chart-vision"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(
    title="FinLearn AI API",
    description="Backend API for RAG chat, chart analysis, and stock screening",
    version="1.0.0"
)

# CORS - allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Forward declarations for prewarming (actual functions defined below)
_cv_loaded = False
_sr_model = None
_trend_model = None

def _prewarm_cv():
    """Helper to prewarm CV models - actual load_cv_models defined below."""
    global _cv_loaded, _sr_model, _trend_model
    # This will be replaced by the actual function below
    pass

# =============================================================================
# Startup Prewarming - Load all models on startup for faster first requests
# =============================================================================

@app.on_event("startup")
async def prewarm_models():
    """Prewarm all models on startup so first requests are fast."""
    global _cv_loaded, _sr_model, _trend_model
    
    print("\n" + "="*60)
    print("PREWARMING MODELS ON STARTUP...")
    print("="*60)
    
    # 1. Prewarm RAG (embedding model + reranker + knowledge base)
    print("\n[Prewarm] Loading RAG components...")
    try:
        from rag.retrieval import get_retriever
        retriever = get_retriever()
        # Do a dummy query to fully initialize
        retriever.retrieve("what is investing", top_k=1)
        print("[Prewarm] RAG ready")
    except Exception as e:
        print(f"[Prewarm] RAG failed: {e}")
    
    # 2. Prewarm CV models (call the actual function defined below)
    print("\n[Prewarm] Loading CV models...")
    try:
        # Import and call the actual load function
        load_cv_models()
        if _sr_model and _trend_model:
            print("[Prewarm] CV models ready")
        else:
            print("[Prewarm] CV models partially loaded")
    except Exception as e:
        print(f"[Prewarm] CV models failed: {e}")
    
    # 3. Check LLM connection
    print("\n[Prewarm] Checking LLM connection...")
    try:
        from rag.llm_provider import check_llm_status
        status = check_llm_status()
        if status.get('status') == 'online':
            print(f"[Prewarm] LLM ready ({status.get('provider')})")
        else:
            print(f"[Prewarm] LLM not available: {status}")
    except Exception as e:
        print(f"[Prewarm] LLM check failed: {e}")
    
    print("\n" + "="*60)
    print("SERVER READY - All models prewarmed!")
    print("="*60 + "\n")

# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    lesson_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []

class StockRequest(BaseModel):
    tickers: List[str]

class ChartAnalysisResponse(BaseModel):
    trend: str
    trend_confidence: float
    trend_probabilities: Dict[str, float]
    support_zones: List[Dict[str, Any]]
    resistance_zones: List[Dict[str, Any]]
    annotated_image: Optional[str] = None  # Base64 encoded PNG with lines drawn

# =============================================================================
# RAG Chat Endpoint
# =============================================================================

# Lazy load RAG components
_rag_loaded = False
_retriever = None
_llm = None

def load_rag():
    global _rag_loaded, _retriever, _llm
    if _rag_loaded:
        return
    
    try:
        from rag.retrieval import retrieve_with_citations
        from rag.llm_provider import chat_with_llm, check_llm_status
        
        # Check LLM status
        status = check_llm_status()
        if status.get('status') != 'online':
            print(f"[RAG] LLM not available: {status}")
        else:
            print(f"[RAG] LLM ready: {status.get('provider')}")
        
        _rag_loaded = True
        print("[RAG] Components loaded successfully")
    except Exception as e:
        print(f"[RAG] Failed to load: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG-powered chat endpoint."""
    load_rag()
    
    try:
        from rag.retrieval import retrieve_with_citations, format_context_with_citations
        from rag.llm_provider import chat_with_llm
        
        # Retrieve relevant context - returns RetrievalResponse object
        # Use lower confidence threshold (0.15) to allow basic questions
        retrieval_response = retrieve_with_citations(
            query=request.message,
            top_k=5,
            min_confidence=0.15,  # Lower threshold for basic questions
            current_lesson_id=request.lesson_id
        )
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        # Access results from the RetrievalResponse object
        for i, result in enumerate(retrieval_response.results[:5]):
            # result is a RetrievalResult object with a .chunk attribute
            chunk = result.chunk
            # Don't include source labels in context - just the content
            context_parts.append(chunk.content[:1000])
            sources.append({
                'title': chunk.source,
                'snippet': chunk.content[:200],
                'score': result.final_score
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Only refuse if we have NO results at all
        if not retrieval_response.results or len(retrieval_response.results) == 0:
            return ChatResponse(
                response="I don't have information about that topic in my knowledge base. Try asking about investing basics, stocks, bonds, ETFs, or portfolio management.",
                sources=[]
            )
        
        # Generate response with LLM
        prompt = f"""You are a helpful investing tutor for beginners. Answer the user's question based on the provided context.
Be conversational, clear, and educational. Use the context to inform your answer.

CRITICAL RULES:
- Do NOT include any source references like [Source 1], [Source 2], etc.
- Do NOT write "Sources:" or list any sources at the end
- Do NOT mention citations or references
- Just provide a clean, helpful response

Context:
{context}

User Question: {request.message}

Provide a helpful answer:"""
        
        response = chat_with_llm(prompt, stream=False)
        
        # If still a generator, consume it
        if hasattr(response, '__iter__') and not isinstance(response, str):
            response = ''.join(response)
        
        # Clean up response - remove any source references the LLM might have added
        import re
        # Remove patterns like "Sources:", "Source:", "[Source 1]", etc.
        response = re.sub(r'\n*Sources?:.*$', '', response, flags=re.IGNORECASE | re.DOTALL)
        response = re.sub(r'\[Source \d+\]', '', response)
        response = re.sub(r'\(Source \d+\)', '', response)
        response = re.sub(r'Source \d+:', '', response)
        response = response.strip()
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        print(f"[Chat Error] {e}")
        import traceback
        traceback.print_exc()
        # Fallback response
        return ChatResponse(
            response=f"I'm having trouble connecting to my knowledge base right now. Your question was about: {request.message}. Please try again in a moment.",
            sources=[]
        )

# =============================================================================
# Chart Analysis Endpoint
# =============================================================================

# CV model variables are declared at the top of the file (forward declarations)

def load_cv_models():
    global _cv_loaded, _sr_model, _trend_model
    if _cv_loaded:
        return
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import models
        
        # Define model architectures locally (EXACTLY matching training scripts)
        class SRZoneModel(nn.Module):
            """S/R Zone Detection Model - matches train_sr_model_v2.py exactly"""
            def __init__(self, num_zones=10):
                super().__init__()
                self.num_zones = num_zones
                
                # Use ResNet34 for better capacity
                backbone = models.resnet34(weights=None)
                
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
        
        class TrendModelV2(nn.Module):
            """Trend Classification Model - matches train_trend_model_v2.py exactly"""
            def __init__(self, num_classes=3):
                super().__init__()
                
                # EfficientNet-B2 backbone
                try:
                    self.backbone = models.efficientnet_b2(weights=None)
                    in_features = self.backbone.classifier[1].in_features
                except:
                    self.backbone = models.resnet50(weights=None)
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Identity()
                
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
                slope = self.slope_head(features)
                return class_logits, slope
        
        device = torch.device('cpu')
        checkpoint_dir = Path(__file__).parent.parent / "chart-vision" / "checkpoints"
        
        print(f"[CV] Looking for models in: {checkpoint_dir}")
        
        # Load S/R model
        sr_path = checkpoint_dir / "sr_zone_model_best.pt"
        if sr_path.exists():
            _sr_model = SRZoneModel(num_zones=10)
            checkpoint = torch.load(sr_path, map_location=device, weights_only=False)
            _sr_model.load_state_dict(checkpoint['model_state_dict'])
            _sr_model.eval()
            print("[CV] S/R model loaded successfully")
        else:
            print(f"[CV] S/R model not found at {sr_path}")
        
        # Load Trend model
        trend_path = checkpoint_dir / "trend_model_v2_best.pt"
        if trend_path.exists():
            _trend_model = TrendModelV2()
            checkpoint = torch.load(trend_path, map_location=device, weights_only=False)
            _trend_model.load_state_dict(checkpoint['model_state_dict'])
            _trend_model.eval()
            print("[CV] Trend model loaded successfully")
        else:
            print(f"[CV] Trend model not found at {trend_path}")
        
        _cv_loaded = True
        
    except Exception as e:
        print(f"[CV] Failed to load models: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/analyze-chart", response_model=ChartAnalysisResponse)
async def analyze_chart(file: UploadFile = File(...)):
    """Analyze uploaded chart image with CV models."""
    load_cv_models()
    
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import numpy as np
        
        # Read and preprocess image
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(original_image).unsqueeze(0)
        
        result = {
            'trend': 'sideways',
            'trend_confidence': 0.5,
            'trend_probabilities': {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34},
            'support_zones': [],
            'resistance_zones': [],
            'signal': 'HOLD'
        }
        
        # Trend prediction
        if _trend_model is not None:
            with torch.no_grad():
                class_logits, slope = _trend_model(img_tensor)
                probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
                
                classes = ['downtrend', 'sideways', 'uptrend']
                pred_idx = np.argmax(probs)
                
                result['trend'] = classes[pred_idx]
                result['trend_confidence'] = float(probs[pred_idx])
                result['trend_probabilities'] = {
                    cls: float(probs[i]) for i, cls in enumerate(classes)
                }
        
        # S/R prediction - use normalized price range (0-100) for uploaded images
        if _sr_model is not None:
            with torch.no_grad():
                logits = _sr_model(img_tensor)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                
                num_zones = len(probs) // 2
                support_probs = probs[:num_zones]
                resistance_probs = probs[num_zones:]
                
                # Use percentage-based zones for uploaded images
                for i, prob in enumerate(support_probs):
                    if prob > 0.4:
                        # Convert zone to approximate price level (0-100 scale)
                        zone_price = (i + 0.5) / num_zones * 100
                        result['support_zones'].append({
                            'zone': i + 1,
                            'price': round(zone_price, 1),
                            'confidence': int(prob * 100)  # Convert to percentage
                        })
                
                for i, prob in enumerate(resistance_probs):
                    if prob > 0.4:
                        zone_price = (i + 0.5) / num_zones * 100
                        result['resistance_zones'].append({
                            'zone': i + 1,
                            'price': round(zone_price, 1),
                            'confidence': int(prob * 100)  # Convert to percentage
                        })
        
        # Calculate signal
        if result['trend'] == 'uptrend' and result['trend_confidence'] > 0.6:
            result['signal'] = 'BUY'
        elif result['trend'] == 'downtrend' and result['trend_confidence'] > 0.6:
            result['signal'] = 'SELL'
        else:
            result['signal'] = 'HOLD'
        
        # Draw analysis on the original image
        annotated_image = draw_analysis_on_chart(
            original_image,
            result,
            (0, 100),  # Normalized price range for uploaded images
            num_bars=30
        )
        
        # Convert annotated image to base64
        buf = io.BytesIO()
        annotated_image.save(buf, format='PNG')
        buf.seek(0)
        result['annotated_image'] = base64.b64encode(buf.read()).decode('utf-8')
        
        return ChartAnalysisResponse(**result)
        
    except Exception as e:
        print(f"[Chart Analysis Error] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Stock Screener Endpoint - Real CV Model Analysis
# =============================================================================

def generate_chart_image_from_data(bars, figsize=(6, 4), dpi=100):
    """Generate a candlestick chart image from Polygon bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    
    dates = range(len(bars))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dark background
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Draw candlesticks
    width = 0.6
    up_color = '#22c55e'
    down_color = '#ef4444'
    
    for i, bar in enumerate(bars):
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        is_up = c >= o
        color = up_color if is_up else down_color
        
        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o) if abs(c - o) > 0 else 0.01
        ax.add_patch(Rectangle((i - width/2, body_bottom), width, body_height,
                               facecolor=color, edgecolor=color))
        # Wicks
        ax.plot([i, i], [l, body_bottom], color=color, linewidth=1)
        ax.plot([i, i], [body_bottom + body_height, h], color=color, linewidth=1)
    
    ax.set_xlim(-1, len(bars))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor='#1a1a2e',
                bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)


def analyze_chart_with_models(image, price_range):
    """Run CV models on chart image."""
    import torch
    import torchvision.transforms as transforms
    import numpy as np
    
    results = {
        'trend': 'sideways',
        'trend_confidence': 0.5,
        'support_zones': [],
        'resistance_zones': [],
        'signal': 'HOLD',
        'signal_strength': 50
    }
    
    if not _cv_loaded:
        return results
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image.convert('RGB')).unsqueeze(0)
    
    # Trend prediction
    if _trend_model is not None:
        with torch.no_grad():
            class_logits, slope_pred = _trend_model(img_tensor)
            probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
            
            classes = ['downtrend', 'sideways', 'uptrend']
            pred_idx = np.argmax(probs)
            
            results['trend'] = classes[pred_idx]
            results['trend_confidence'] = float(probs[pred_idx])
    
    # S/R prediction
    if _sr_model is not None:
        with torch.no_grad():
            logits = _sr_model(img_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            num_zones = len(probs) // 2
            support_probs = probs[:num_zones]
            resistance_probs = probs[num_zones:]
            
            price_min, price_max = price_range
            price_step = (price_max - price_min) / num_zones
            
            for i, prob in enumerate(support_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    results['support_zones'].append({
                        'zone': i + 1,
                        'price': round(price, 2),
                        'confidence': int(prob * 100)  # Convert to percentage
                    })
            
            for i, prob in enumerate(resistance_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    results['resistance_zones'].append({
                        'zone': i + 1,
                        'price': round(price, 2),
                        'confidence': int(prob * 100)  # Convert to percentage
                    })
    
    # Calculate signal
    signal_score = 0
    
    if results['trend'] == 'uptrend':
        signal_score += results['trend_confidence'] * 40
    elif results['trend'] == 'downtrend':
        signal_score -= results['trend_confidence'] * 40
    
    if results['support_zones']:
        signal_score += 15
    if results['resistance_zones']:
        signal_score -= 15
    
    if signal_score > 20:
        results['signal'] = 'BUY'
        results['signal_strength'] = min(signal_score, 100)
    elif signal_score < -20:
        results['signal'] = 'SELL'
        results['signal_strength'] = min(abs(signal_score), 100)
    else:
        results['signal'] = 'HOLD'
        results['signal_strength'] = 50
    
    return results


def draw_analysis_on_chart(image, analysis, price_range, num_bars=30):
    """Draw S/R lines and trend line on the chart image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    import numpy as np
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(img_array)
    
    img_height, img_width = img_array.shape[:2]
    price_min, price_max = price_range
    price_range_val = price_max - price_min
    
    def price_to_y(price):
        """Convert price to y-coordinate (inverted because image y=0 is top)"""
        if price_range_val == 0:
            return img_height / 2
        normalized = (price - price_min) / price_range_val
        # Invert and add padding (chart doesn't go to edges)
        return img_height * (1 - normalized * 0.85 - 0.075)
    
    # Draw Support lines (green)
    for zone in analysis.get('support_zones', []):
        y = price_to_y(zone.get('price', 50))
        # Confidence might be 0-100 (int) or 0-1 (float), normalize to 0-1
        confidence = zone.get('confidence', 50)
        if confidence > 1:
            confidence = confidence / 100.0
        alpha = min(0.4 + confidence * 0.5, 0.95)  # Clamp to valid range
        ax.axhline(y=y, color='#22c55e', linestyle='--', linewidth=2, alpha=alpha)
        price_label = zone.get('price', 0)
        ax.text(img_width - 5, y - 5, f"S: ${price_label:.2f}" if price_label > 1 else f"S: Zone {zone.get('zone', '?')}", 
                color='#22c55e', fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8))
    
    # Draw Resistance lines (red)
    for zone in analysis.get('resistance_zones', []):
        y = price_to_y(zone.get('price', 50))
        # Confidence might be 0-100 (int) or 0-1 (float), normalize to 0-1
        confidence = zone.get('confidence', 50)
        if confidence > 1:
            confidence = confidence / 100.0
        alpha = min(0.4 + confidence * 0.5, 0.95)  # Clamp to valid range
        ax.axhline(y=y, color='#ef4444', linestyle='--', linewidth=2, alpha=alpha)
        price_label = zone.get('price', 0)
        ax.text(img_width - 5, y + 12, f"R: ${price_label:.2f}" if price_label > 1 else f"R: Zone {zone.get('zone', '?')}", 
                color='#ef4444', fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8))
    
    # Draw Trend line
    trend = analysis.get('trend', 'sideways')
    trend_conf = analysis.get('trend_confidence', 0.5)
    
    if trend == 'uptrend':
        # Draw diagonal line from bottom-left to top-right
        ax.plot([0, img_width], [img_height * 0.7, img_height * 0.3], 
                color='#22c55e', linewidth=3, alpha=0.7, linestyle='-')
        ax.text(10, 20, f"↗ UPTREND ({trend_conf*100:.0f}%)", 
                color='#22c55e', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))
    elif trend == 'downtrend':
        # Draw diagonal line from top-left to bottom-right
        ax.plot([0, img_width], [img_height * 0.3, img_height * 0.7], 
                color='#ef4444', linewidth=3, alpha=0.7, linestyle='-')
        ax.text(10, 20, f"↘ DOWNTREND ({trend_conf*100:.0f}%)", 
                color='#ef4444', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))
    else:
        # Draw horizontal line for sideways
        ax.plot([0, img_width], [img_height * 0.5, img_height * 0.5], 
                color='#f59e0b', linewidth=3, alpha=0.7, linestyle='-')
        ax.text(10, 20, f"→ SIDEWAYS ({trend_conf*100:.0f}%)", 
                color='#f59e0b', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9))
    
    # Add signal badge
    signal = analysis.get('signal', 'HOLD')
    signal_colors = {'BUY': '#22c55e', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}
    signal_color = signal_colors.get(signal, '#f59e0b')
    ax.text(img_width - 10, 20, signal, 
            color=signal_color, fontsize=12, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', 
                     edgecolor=signal_color, linewidth=2, alpha=0.9))
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convert back to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#1a1a2e',
                bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    return PILImage.open(buf)


class StockAnalysisResponse(BaseModel):
    ticker: str
    price: float
    change: float
    change_pct: float
    trend: str
    trend_confidence: float
    signal: str
    signal_strength: float
    support: Optional[float] = None
    resistance: Optional[float] = None
    support_zones: List[Dict[str, Any]] = []
    resistance_zones: List[Dict[str, Any]] = []
    chart_image: str  # Base64 encoded PNG


@app.post("/api/stocks", response_model=List[StockAnalysisResponse])
async def get_stocks(request: StockRequest):
    """Get stock data with real CV model analysis."""
    
    polygon_key = os.environ.get('POLYGON_API_KEY')
    if not polygon_key:
        raise HTTPException(status_code=500, detail="Polygon API key not configured")
    
    # Load CV models if not already loaded
    load_cv_models()
    
    try:
        from polygon import RESTClient
        import numpy as np
        
        client = RESTClient(polygon_key)
        results = []
        
        for ticker in request.tickers:
            try:
                # Get 30+ days of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=45)
                
                bars = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    limit=50
                )
                
                if not bars or len(bars) < 10:
                    continue
                
                # Get last 30 days
                bars = bars[-30:] if len(bars) > 30 else bars
                
                latest = bars[-1]
                prev = bars[-2] if len(bars) > 1 else bars[-1]
                
                change = latest.close - prev.close
                change_pct = (change / prev.close) * 100 if prev.close > 0 else 0
                
                # Generate chart image
                chart_image = generate_chart_image_from_data(bars)
                
                # Get price range for S/R calculation
                price_min = min(b.low for b in bars)
                price_max = max(b.high for b in bars)
                
                # Run CV model analysis
                analysis = analyze_chart_with_models(chart_image, (price_min, price_max))
                
                # Draw analysis lines on the chart
                annotated_chart = draw_analysis_on_chart(
                    chart_image, 
                    analysis, 
                    (price_min, price_max),
                    num_bars=len(bars)
                )
                
                # Convert annotated image to base64
                buf = io.BytesIO()
                annotated_chart.save(buf, format='PNG')
                buf.seek(0)
                chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Get support/resistance prices
                support_price = analysis['support_zones'][0]['price'] if analysis['support_zones'] else round(price_min, 2)
                resistance_price = analysis['resistance_zones'][0]['price'] if analysis['resistance_zones'] else round(price_max, 2)
                
                results.append(StockAnalysisResponse(
                    ticker=ticker,
                    price=round(latest.close, 2),
                    change=round(change, 2),
                    change_pct=round(change_pct, 2),
                    trend=analysis['trend'],
                    trend_confidence=round(analysis['trend_confidence'] * 100, 1),
                    signal=analysis['signal'],
                    signal_strength=round(analysis['signal_strength'], 1),
                    support=support_price,
                    resistance=resistance_price,
                    support_zones=analysis['support_zones'],
                    resistance_zones=analysis['resistance_zones'],
                    chart_image=chart_base64
                ))
                
            except Exception as e:
                print(f"[Stock Error] {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
        
    except Exception as e:
        print(f"[Stocks Error] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health Check
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Check API health and component status."""
    
    status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "rag": False,
            "cv_models": False,
            "polygon": False
        }
    }
    
    # Check RAG
    try:
        from rag.llm_provider import check_llm_status
        llm_status = check_llm_status()
        status["components"]["rag"] = llm_status.get('status') == 'online'
    except:
        pass
    
    # Check CV models
    status["components"]["cv_models"] = _cv_loaded and (_sr_model is not None or _trend_model is not None)
    
    # Check Polygon
    status["components"]["polygon"] = bool(os.environ.get('POLYGON_API_KEY'))
    
    return status

# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
