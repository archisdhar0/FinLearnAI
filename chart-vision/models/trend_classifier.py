"""
Trend Classifier - CNN model to classify chart trends
Architecture: ResNet18 fine-tuned for chart images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class ChartDataset(Dataset):
    """Dataset for loading chart images with trend labels."""
    
    CLASSES = ['downtrend', 'sideways', 'uptrend']
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Args:
            data_dir: Directory containing images and labels.json
            split: 'train' or 'val' (80/20 split)
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self.default_transform()
        
        # Load labels
        labels_path = self.data_dir / 'labels.json'
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Create samples list: [(image_path, label_idx), ...]
        self.samples = []
        for class_name, filepaths in labels.items():
            label_idx = self.CLASSES.index(class_name)
            for filepath in filepaths:
                self.samples.append((filepath, label_idx))
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * 0.8)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def default_transform(self):
        """Default image transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TrendClassifier(nn.Module):
    """
    CNN for classifying stock chart trends.
    Uses ResNet18 backbone with custom classifier head.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        # Replace final layer for our 3 classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image: torch.Tensor) -> dict:
        """
        Make prediction on a single image.
        
        Args:
            image: Preprocessed image tensor (1, 3, 224, 224)
        
        Returns:
            Dictionary with predicted class and probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(image)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            return {
                'prediction': ChartDataset.CLASSES[pred_idx],
                'confidence': probs[0, pred_idx].item(),
                'probabilities': {
                    cls: probs[0, i].item() 
                    for i, cls in enumerate(ChartDataset.CLASSES)
                }
            }


class TrendTrainer:
    """Training pipeline for the trend classifier."""
    
    def __init__(
        self,
        model: TrendClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'auto'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Device selection
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )
        
        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, epochs: int = 20, save_dir: str = 'checkpoints'):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print('='*50)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc, preds, labels = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path / 'best_model.pt')
                print(f"✅ Saved new best model (acc: {val_acc:.4f})")
        
        # Final evaluation
        self.plot_history(save_path)
        self.plot_confusion_matrix(preds, labels, save_path)
        
        return self.history
    
    def plot_history(self, save_dir: Path):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        ax2.plot(self.history['val_acc'])
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=150)
        plt.close()
        print(f"Saved training history to {save_dir / 'training_history.png'}")
    
    def plot_confusion_matrix(self, preds, labels, save_dir: Path):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ChartDataset.CLASSES,
            yticklabels=ChartDataset.CLASSES
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=ChartDataset.CLASSES))


def train_trend_classifier(data_dir: str = 'data/raw', epochs: int = 20):
    """Main training function."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),  # Flipping changes trend!
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ChartDataset(data_dir, split='train', transform=train_transform)
    val_dataset = ChartDataset(data_dir, split='val', transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    model = TrendClassifier(num_classes=3, pretrained=True)
    
    # Create trainer
    trainer = TrendTrainer(model, train_loader, val_loader)
    
    # Train
    history = trainer.train(epochs=epochs)
    
    return model, history


if __name__ == "__main__":
    model, history = train_trend_classifier()
