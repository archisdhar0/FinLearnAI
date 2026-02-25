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
        
        # Draw analysis on the original image (no price labels for uploaded images)
        annotated_image = draw_analysis_on_chart(
            original_image,
            result,
            (0, 100),  # Normalized price range for uploaded images
            num_bars=30,
            show_prices=False  # Don't show price numbers for uploaded images
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
# XAI (Explainable AI) Endpoint - Grad-CAM Visualization
# =============================================================================

class XAIResponse(BaseModel):
    sr_heatmap: Optional[str] = None  # Base64 encoded heatmap image
    trend_heatmap: Optional[str] = None
    explanation: str = ""

@app.post("/api/xai", response_model=XAIResponse)
async def generate_xai(file: UploadFile = File(...)):
    """Generate Grad-CAM heatmap showing what the model focuses on."""
    load_cv_models()
    
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from PIL import Image as PILImage
        import numpy as np
        import cv2
        
        # Read and preprocess image
        contents = await file.read()
        image = PILImage.open(io.BytesIO(contents)).convert('RGB')
        original_array = np.array(image)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        result = XAIResponse(explanation="")
        explanations = []
        
        # Generate Grad-CAM for Trend model
        if _trend_model is not None:
            try:
                _trend_model.eval()
                
                # Get the target layer (last conv layer of backbone)
                if hasattr(_trend_model, 'backbone') and hasattr(_trend_model.backbone, 'features'):
                    target_layer = _trend_model.backbone.features[-1]
                else:
                    target_layer = None
                
                if target_layer:
                    activations = None
                    gradients = None
                    
                    def forward_hook(module, input, output):
                        nonlocal activations
                        activations = output.detach()
                    
                    def backward_hook(module, grad_input, grad_output):
                        nonlocal gradients
                        gradients = grad_output[0].detach()
                    
                    fh = target_layer.register_forward_hook(forward_hook)
                    bh = target_layer.register_full_backward_hook(backward_hook)
                    
                    # Forward pass
                    output, _ = _trend_model(input_tensor)
                    pred_class = output.argmax(dim=1).item()
                    class_names = ['downtrend', 'sideways', 'uptrend']
                    
                    # Backward pass
                    _trend_model.zero_grad()
                    one_hot = torch.zeros_like(output)
                    one_hot[0, pred_class] = 1
                    output.backward(gradient=one_hot)
                    
                    # Generate heatmap
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam).squeeze().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    # Resize and colorize
                    heatmap = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))
                    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    
                    # Overlay
                    overlay = (0.6 * original_array + 0.4 * heatmap_colored).astype(np.uint8)
                    
                    # Convert to base64
                    overlay_img = PILImage.fromarray(overlay)
                    buf = io.BytesIO()
                    overlay_img.save(buf, format='PNG')
                    buf.seek(0)
                    result.trend_heatmap = base64.b64encode(buf.read()).decode('utf-8')
                    
                    explanations.append(f"Trend: Model predicted {class_names[pred_class].upper()}")
                    
                    fh.remove()
                    bh.remove()
                    
            except Exception as e:
                print(f"[XAI Trend Error] {e}")
        
        # Generate Grad-CAM for S/R model
        if _sr_model is not None:
            try:
                _sr_model.eval()
                
                if hasattr(_sr_model, 'backbone') and hasattr(_sr_model.backbone, 'features'):
                    target_layer = _sr_model.backbone.features[-1]
                else:
                    target_layer = None
                
                if target_layer:
                    activations = None
                    gradients = None
                    
                    def forward_hook(module, input, output):
                        nonlocal activations
                        activations = output.detach()
                    
                    def backward_hook(module, grad_input, grad_output):
                        nonlocal gradients
                        gradients = grad_output[0].detach()
                    
                    fh = target_layer.register_forward_hook(forward_hook)
                    bh = target_layer.register_full_backward_hook(backward_hook)
                    
                    # Forward pass
                    output = _sr_model(input_tensor)
                    probs = torch.sigmoid(output)[0]
                    
                    # Find strongest zone
                    max_idx = probs.argmax().item()
                    
                    # Backward pass for that zone
                    _sr_model.zero_grad()
                    output[0, max_idx].backward()
                    
                    # Generate heatmap
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam).squeeze().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    # Resize and colorize
                    heatmap = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))
                    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    
                    # Overlay
                    overlay = (0.6 * original_array + 0.4 * heatmap_colored).astype(np.uint8)
                    
                    # Convert to base64
                    overlay_img = PILImage.fromarray(overlay)
                    buf = io.BytesIO()
                    overlay_img.save(buf, format='PNG')
                    buf.seek(0)
                    result.sr_heatmap = base64.b64encode(buf.read()).decode('utf-8')
                    
                    num_zones = len(probs) // 2
                    zone_type = "Support" if max_idx < num_zones else "Resistance"
                    explanations.append(f"S/R: Model focused on {zone_type} detection")
                    
                    fh.remove()
                    bh.remove()
                    
            except Exception as e:
                print(f"[XAI S/R Error] {e}")
        
        result.explanation = " | ".join(explanations) if explanations else "XAI visualization generated"
        return result
        
    except Exception as e:
        print(f"[XAI Error] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Stock Screener Endpoint - Real CV Model Analysis
# =============================================================================

def generate_chart_image_from_data(bars, figsize=(8, 4), dpi=120):
    """Generate a clean, minimal candlestick chart image from Polygon bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Clean dark background
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    # Draw candlesticks - cleaner style
    width = 0.7
    up_color = '#10b981'    # Softer green
    down_color = '#f43f5e'  # Softer red
    
    prices = []
    for i, bar in enumerate(bars):
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        prices.extend([h, l])
        is_up = c >= o
        color = up_color if is_up else down_color
        
        # Body - slightly rounded look with edge
        body_bottom = min(o, c)
        body_height = abs(c - o) if abs(c - o) > 0.001 else 0.01
        ax.add_patch(Rectangle((i - width/2, body_bottom), width, body_height,
                               facecolor=color, edgecolor=color, linewidth=0.5))
        # Wicks - thinner
        ax.plot([i, i], [l, body_bottom], color=color, linewidth=0.8)
        ax.plot([i, i], [body_bottom + body_height, h], color=color, linewidth=0.8)
    
    # Add subtle grid
    ax.grid(True, axis='y', color='#2a2a4a', linewidth=0.3, alpha=0.5)
    
    # Set axis limits with padding
    price_min, price_max = min(prices), max(prices)
    price_padding = (price_max - price_min) * 0.1
    ax.set_ylim(price_min - price_padding, price_max + price_padding)
    ax.set_xlim(-0.5, len(bars) - 0.5)
    
    # Clean axis styling
    ax.set_xticks([])
    ax.tick_params(axis='y', colors='#6b7280', labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
    
    # Remove spines except left
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#2a2a4a')
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor='#0f0f1a',
                bbox_inches='tight', pad_inches=0.1)
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
            
            # Collect all zones above threshold, then sort by confidence
            support_candidates = []
            resistance_candidates = []
            
            for i, prob in enumerate(support_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    support_candidates.append({
                        'zone': i + 1,
                        'price': round(price, 2),
                        'confidence': int(prob * 100)
                    })
            
            for i, prob in enumerate(resistance_probs):
                if prob > 0.4:
                    price = price_min + (i + 0.5) * price_step
                    resistance_candidates.append({
                        'zone': i + 1,
                        'price': round(price, 2),
                        'confidence': int(prob * 100)
                    })
            
            # Sort by confidence (highest first) and take top zones
            support_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            resistance_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            results['support_zones'] = support_candidates
            results['resistance_zones'] = resistance_candidates
            
            # Debug: print all zone probabilities
            print(f"[S/R Debug] Price range: ${price_min:.2f} - ${price_max:.2f}")
            print(f"[S/R Debug] Support probs: {[f'{p:.2f}' for p in support_probs]}")
            print(f"[S/R Debug] Resistance probs: {[f'{p:.2f}' for p in resistance_probs]}")
    
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


def draw_analysis_on_chart(image, analysis, price_range, num_bars=30, show_prices=True):
    """Draw clean, minimal S/R lines and trend indicator on the chart image.
    
    Args:
        image: PIL Image
        analysis: dict with trend, support_zones, resistance_zones
        price_range: (min, max) price tuple
        num_bars: number of bars in chart
        show_prices: whether to show price labels (False for uploaded images without real prices)
    """
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
    
    # Check if we have real prices (not normalized 0-100 range)
    has_real_prices = show_prices and price_max > 100
    
    def price_to_y(price):
        """Convert price to y-coordinate (inverted because image y=0 is top)"""
        if price_range_val == 0:
            return img_height / 2
        normalized = (price - price_min) / price_range_val
        return img_height * (1 - normalized * 0.8 - 0.1)
    
    # Draw Support lines (green) - pick the LOWEST price zones (bottom of chart)
    support_zones = analysis.get('support_zones', [])
    # Sort by price ascending (lowest first) for support, take top 2
    support_to_draw = sorted(support_zones, key=lambda x: x.get('price', 0))[:2]
    for i, zone in enumerate(support_to_draw):
        y = price_to_y(zone.get('price', 50))
        ax.axhline(y=y, color='#10b981', linestyle='-', linewidth=1.5, alpha=0.8)
        if has_real_prices:
            price_label = zone.get('price', 0)
            ax.text(img_width - 8, y - 3, f'${price_label:.0f}', 
                    color='#10b981', fontsize=7, ha='right', va='bottom', fontweight='bold')
        else:
            ax.text(img_width - 8, y - 3, 'S', 
                    color='#10b981', fontsize=8, ha='right', va='bottom', fontweight='bold')
    
    # Draw Resistance lines (red) - pick the HIGHEST price zones (top of chart)
    resistance_zones = analysis.get('resistance_zones', [])
    # Sort by price descending (highest first) for resistance, take top 2
    resistance_to_draw = sorted(resistance_zones, key=lambda x: x.get('price', 0), reverse=True)[:2]
    for i, zone in enumerate(resistance_to_draw):
        y = price_to_y(zone.get('price', 50))
        ax.axhline(y=y, color='#f43f5e', linestyle='-', linewidth=1.5, alpha=0.8)
        if has_real_prices:
            price_label = zone.get('price', 0)
            ax.text(img_width - 8, y + 10, f'${price_label:.0f}', 
                    color='#f43f5e', fontsize=7, ha='right', va='top', fontweight='bold')
        else:
            ax.text(img_width - 8, y + 10, 'R', 
                    color='#f43f5e', fontsize=8, ha='right', va='top', fontweight='bold')
    
    # Draw subtle trend line
    trend = analysis.get('trend', 'sideways')
    trend_conf = analysis.get('trend_confidence', 0.5)
    
    trend_colors = {'uptrend': '#10b981', 'downtrend': '#f43f5e', 'sideways': '#f59e0b'}
    trend_color = trend_colors.get(trend, '#f59e0b')
    
    if trend == 'uptrend':
        ax.plot([img_width * 0.1, img_width * 0.9], [img_height * 0.65, img_height * 0.35], 
                color=trend_color, linewidth=2, alpha=0.6, linestyle='--')
    elif trend == 'downtrend':
        ax.plot([img_width * 0.1, img_width * 0.9], [img_height * 0.35, img_height * 0.65], 
                color=trend_color, linewidth=2, alpha=0.6, linestyle='--')
    
    # Minimal signal badge - top right corner
    signal = analysis.get('signal', 'HOLD')
    signal_colors = {'BUY': '#10b981', 'SELL': '#f43f5e', 'HOLD': '#f59e0b'}
    signal_color = signal_colors.get(signal, '#f59e0b')
    
    # Simple badge
    ax.text(img_width - 10, 15, signal, 
            color='white', fontsize=9, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=signal_color, 
                     edgecolor='none', alpha=0.9))
    
    # Trend indicator - small text below signal
    trend_label = trend.upper()
    conf_pct = int(trend_conf * 100) if trend_conf <= 1 else int(trend_conf)
    ax.text(img_width - 10, 35, f'{trend_label} {conf_pct}%', 
            color='#9ca3af', fontsize=7, ha='right', va='top')
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convert back to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0f0f1a',
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
