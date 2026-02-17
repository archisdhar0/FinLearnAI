"""
Explainable AI (XAI) for Chart Vision Models
Uses Grad-CAM to visualize what the model is "looking at"

Grad-CAM = Gradient-weighted Class Activation Mapping
- Shows which regions of the image influenced the model's decision
- Highlights important areas with a heatmap overlay
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


class GradCAM:
    """
    Grad-CAM implementation for CNN models.
    
    Works by:
    1. Forward pass to get predictions
    2. Backward pass to get gradients
    3. Weight feature maps by gradient importance
    4. Create heatmap showing important regions
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The CNN model
            target_layer: The layer to compute Grad-CAM for (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class index to explain (None = predicted class)
        
        Returns:
            heatmap: Numpy array (H, W) with values 0-1
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle models with multiple outputs (like TrendModelV2)
        if isinstance(output, tuple):
            output = output[0]  # Take classification output
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam.squeeze()
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
        cam = np.array(cam, dtype=np.float32)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class


def get_target_layer(model, model_type='sr'):
    """
    Get the target layer for Grad-CAM based on model architecture.
    
    Args:
        model: The model
        model_type: 'sr' or 'trend'
    
    Returns:
        The target layer (usually last conv layer of backbone)
    """
    # For EfficientNet-based models
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # EfficientNet
        if hasattr(backbone, 'features'):
            return backbone.features[-1]
        
        # ResNet
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1]
    
    # For models with 'features' attribute
    if hasattr(model, 'features'):
        return model.features[-1]
    
    raise ValueError("Could not find target layer for Grad-CAM")


def visualize_gradcam(
    image_path: str,
    model,
    model_type: str = 'sr',
    target_class: int = None,
    save_path: str = None,
    show_original: bool = True
):
    """
    Generate and visualize Grad-CAM for a chart image.
    
    Args:
        image_path: Path to chart image
        model: Trained model
        model_type: 'sr' for S/R model, 'trend' for trend model
        target_class: Class to explain (None = predicted class)
        save_path: Where to save visualization
        show_original: Whether to show original image alongside
    
    Returns:
        Dictionary with prediction and heatmap
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Get target layer
    target_layer = get_target_layer(model, model_type)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap, predicted_class = gradcam.generate(input_tensor, target_class)
    
    # Resize heatmap to image size
    original_size = image.size  # (W, H)
    heatmap_resized = cv2.resize(heatmap, original_size)
    
    # Create visualization
    img_array = np.array(image)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = (0.6 * img_array + 0.4 * heatmap_colored).astype(np.uint8)
    
    # Get class name
    if model_type == 'trend':
        class_names = ['downtrend', 'sideways', 'uptrend']
        class_name = class_names[predicted_class]
    else:
        class_name = f"Zone {predicted_class}"
    
    # Create figure
    if show_original:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Chart', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Model Focus: {class_name}', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(overlay)
        ax.set_title(f'Grad-CAM: Why model predicted "{class_name}"', fontsize=16, fontweight='bold')
        ax.axis('off')
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes if show_original else ax, shrink=0.6, pad=0.02)
    cbar.set_label('Importance', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_name,
        'heatmap': heatmap_resized,
        'overlay': overlay
    }


def explain_sr_prediction(
    image_path: str,
    model,
    save_path: str = None
):
    """
    Explain S/R model prediction with Grad-CAM for each detected zone.
    
    Shows which parts of the chart led to support/resistance detection.
    """
    from train_sr_model_v2 import SRZoneModel
    
    # Load and preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    img_array = np.array(image)
    original_size = image.size
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
    
    num_zones = len(probs) // 2
    support_probs = probs[:num_zones]
    resistance_probs = probs[num_zones:]
    
    # Find significant zones
    significant_support = [(i, p) for i, p in enumerate(support_probs) if p > 0.4]
    significant_resistance = [(i, p) for i, p in enumerate(resistance_probs) if p > 0.4]
    
    # Create multi-panel visualization
    n_panels = 1 + len(significant_support) + len(significant_resistance)
    n_panels = min(n_panels, 5)  # Max 5 panels
    
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Chart', fontsize=12)
    axes[0].axis('off')
    
    # Get target layer
    target_layer = get_target_layer(model, 'sr')
    
    panel_idx = 1
    
    # Explain support zones
    for zone_idx, prob in significant_support[:2]:  # Max 2 support
        if panel_idx >= n_panels:
            break
        
        # Create Grad-CAM for this zone
        gradcam = GradCAM(model, target_layer)
        
        # We need to modify for multi-label - use zone-specific gradient
        model.zero_grad()
        output = model(input_tensor)
        
        # Backprop for specific zone
        loss = output[0, zone_idx]  # Support zone
        loss.backward(retain_graph=True)
        
        gradients = gradcam.gradients
        activations = gradcam.activations
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        heatmap = cv2.resize(cam, original_size)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_GREENS)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * img_array + 0.5 * heatmap_colored).astype(np.uint8)
        
        axes[panel_idx].imshow(overlay)
        axes[panel_idx].set_title(f'Support Zone {zone_idx+1}\n({prob:.0%} confidence)', 
                                  fontsize=12, color='green')
        axes[panel_idx].axis('off')
        panel_idx += 1
    
    # Explain resistance zones
    for zone_idx, prob in significant_resistance[:2]:  # Max 2 resistance
        if panel_idx >= n_panels:
            break
        
        gradcam = GradCAM(model, target_layer)
        model.zero_grad()
        output = model(input_tensor)
        
        loss = output[0, num_zones + zone_idx]  # Resistance zone
        loss.backward(retain_graph=True)
        
        gradients = gradcam.gradients
        activations = gradcam.activations
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        heatmap = cv2.resize(cam, original_size)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_REDS)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * img_array + 0.5 * heatmap_colored).astype(np.uint8)
        
        axes[panel_idx].imshow(overlay)
        axes[panel_idx].set_title(f'Resistance Zone {zone_idx+1}\n({prob:.0%} confidence)', 
                                  fontsize=12, color='red')
        axes[panel_idx].axis('off')
        panel_idx += 1
    
    plt.suptitle('Explainable AI: What the Model Sees', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'support_zones': significant_support,
        'resistance_zones': significant_resistance
    }


def explain_trend_prediction(
    image_path: str,
    model,
    save_path: str = None
):
    """
    Explain trend model prediction with Grad-CAM.
    
    Shows which parts of the chart led to uptrend/downtrend/sideways prediction.
    """
    # Load and preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    img_array = np.array(image)
    original_size = image.size
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output, slope = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        predicted_class = probs.argmax()
    
    class_names = ['downtrend', 'sideways', 'uptrend']
    class_colors = ['red', 'orange', 'green']
    
    # Create Grad-CAM for predicted class
    target_layer = get_target_layer(model, 'trend')
    gradcam = GradCAM(model, target_layer)
    heatmap, _ = gradcam.generate(input_tensor, predicted_class)
    
    heatmap_resized = cv2.resize(heatmap, original_size)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(img_array)
    axes[0].set_title('Original Chart', fontsize=14)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Model Attention', fontsize=14)
    axes[1].axis('off')
    
    # Overlay
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_array + 0.4 * heatmap_colored).astype(np.uint8)
    
    axes[2].imshow(overlay)
    pred_name = class_names[predicted_class]
    pred_color = class_colors[predicted_class]
    axes[2].set_title(f'Prediction: {pred_name.upper()} ({probs[predicted_class]:.0%})', 
                      fontsize=14, fontweight='bold', color=pred_color)
    axes[2].axis('off')
    
    # Add probability bars
    fig.text(0.5, 0.02, 
             f"Probabilities: ↘ Downtrend {probs[0]:.0%} | → Sideways {probs[1]:.0%} | ↗ Uptrend {probs[2]:.0%}",
             ha='center', fontsize=12)
    
    plt.suptitle('Explainable AI: Why the Model Predicted This Trend', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return {
        'prediction': pred_name,
        'confidence': probs[predicted_class],
        'probabilities': {name: prob for name, prob in zip(class_names, probs)},
        'heatmap': heatmap_resized
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Explainable AI for Chart Vision')
    parser.add_argument('--image', type=str, required=True, help='Path to chart image')
    parser.add_argument('--model', choices=['sr', 'trend', 'both'], default='both',
                        help='Which model to explain')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    
    args = parser.parse_args()
    
    if args.model in ['sr', 'both']:
        print("\n" + "="*60)
        print("Explaining S/R Model Prediction")
        print("="*60)
        
        try:
            from train_sr_model_v2 import SRZoneModel
            
            model = SRZoneModel(num_zones=10)
            checkpoint = torch.load('checkpoints/sr_zone_model_best.pt', 
                                   map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            output_path = args.output or 'sr_explanation.png'
            explain_sr_prediction(args.image, model, save_path=output_path)
            
        except Exception as e:
            print(f"Error explaining S/R model: {e}")
    
    if args.model in ['trend', 'both']:
        print("\n" + "="*60)
        print("Explaining Trend Model Prediction")
        print("="*60)
        
        try:
            from train_trend_model_v2 import TrendModelV2
            
            model = TrendModelV2()
            checkpoint = torch.load('checkpoints/trend_model_v2_best.pt',
                                   map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            output_path = args.output or 'trend_explanation.png'
            if args.model == 'both':
                output_path = 'trend_explanation.png'
            explain_trend_prediction(args.image, model, save_path=output_path)
            
        except Exception as e:
            print(f"Error explaining Trend model: {e}")


if __name__ == "__main__":
    main()
