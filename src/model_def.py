# model_def.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)




class MICAttention(nn.Module):
    """基于MIC值的注意力机制模块"""

    def __init__(self, n_features, n_classes, mic_weights=None, temperature=1.0, learnable=True):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.temperature = temperature

        if mic_weights is not None:
            self.attention_weights = nn.Parameter(
                torch.FloatTensor(mic_weights),
                requires_grad=learnable
            )
        else:
            self.attention_weights = nn.Parameter(
                torch.randn(n_classes, n_features),
                requires_grad=learnable
            )

        if not learnable:
            self.attention_weights.requires_grad = False

    def forward(self, x, label_indices=None):
        batch_size = x.size(0)

        attention_scores = self.attention_weights.unsqueeze(0) * x.unsqueeze(1)
        attention_probs = F.softmax(attention_scores / max(self.temperature, 1e-8), dim=2)

        if label_indices is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            label_probs = attention_probs[batch_indices, label_indices, :]
            weighted_features = x * label_probs
        else:
            x_expanded = x.unsqueeze(1).expand(-1, self.n_classes, -1)
            weighted_features = x_expanded * attention_probs

        return weighted_features, attention_probs

    def get_attention_matrix(self):
        return self.attention_weights.detach().cpu().numpy()

class BalancedTraceabilityNN(nn.Module):
    """针对不平衡数据的改进溯源神经网络"""

    def __init__(
        self, n_features, n_classes, mic_weights=None,
        hidden_dims=None, dropout_rate=0.35,
        use_attention=True, attention_learnable=True,
        use_focal_loss=False, gamma=2.5, alpha=None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_features = n_features
        self.n_classes = n_classes
        self.use_attention = use_attention
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = None

        if use_attention:
            self.attention = MICAttention(
                n_features, n_classes, mic_weights,
                learnable=attention_learnable
            )
            attention_output_dim = n_classes * n_features
        else:
            attention_output_dim = n_features

        layers = []
        input_dim = attention_output_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (i + 1) / len(hidden_dims))
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(input_dim, n_classes)
        self.feature_reconstructor = nn.Linear(input_dim, n_features)
        self.embedding_layer = nn.Linear(input_dim, 32)

        print(f"\n模型架构:")
        print(f"  输入维度: {n_features}")
        print(f"  注意力输出: {attention_output_dim}")
        print(f"  隐藏层: {hidden_dims}")
        print(f"  输出维度: {n_classes}")
        if use_attention:
            print(f"  使用MIC注意力机制 (可学习={attention_learnable})")
        if use_focal_loss:
            print(f"  使用Focal Loss (gamma={gamma})")

    def forward(self, x, return_embeddings=False):
        if self.use_attention:
            weighted_features, attention_probs = self.attention(x)
            x = weighted_features.view(x.size(0), -1)
        else:
            attention_probs = None

        features = self.feature_extractor(x)
        logits = self.classifier(features)
        reconstructed_features = self.feature_reconstructor(features)
        embeddings = self.embedding_layer(features) if return_embeddings else None

        outputs = {
            'logits': logits,
            'reconstructed_features': reconstructed_features,
            'attention_probs': attention_probs
        }

        if return_embeddings:
            outputs['embeddings'] = embeddings

        return outputs

    def compute_loss(self, outputs, targets, features_original, reconstruction_weight=0.05):
        logits = outputs['logits']
        reconstructed_features = outputs['reconstructed_features']

        if self.use_focal_loss:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                focal_loss = alpha[targets] * focal_loss

            classification_loss = focal_loss.mean()
        else:
            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                classification_loss = F.cross_entropy(logits, targets, weight=alpha)
            else:
                classification_loss = F.cross_entropy(logits, targets)

        reconstruction_loss = F.mse_loss(reconstructed_features, features_original)
        total_loss = classification_loss + reconstruction_weight * reconstruction_loss

        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'reconstruction_loss': reconstruction_loss
        }

    def predict_with_explanation(self, x, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(device)
        else:
            x = x.to(device)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['logits']
            attention_probs = outputs['attention_probs']

            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            batch_size = x.size(0)
            batch_indices = torch.arange(batch_size, device=x.device)

            if attention_probs is not None:
                predicted_attention = attention_probs[batch_indices, predictions, :]
                feature_importance = x * predicted_attention
            else:
                feature_importance = None

            return {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'attention_weights': attention_probs.cpu().numpy() if attention_probs is not None else None,
                'feature_importance': feature_importance.cpu().numpy() if feature_importance is not None else None
            }


class ModelEnsemble:
    """模型集成"""

    def __init__(self, models, output_manager, ensemble_method='soft_voting'):
        self.models = models
        self.output_manager = output_manager
        self.ensemble_method = ensemble_method
        self.n_models = len(models)

    def predict(self, X, device='cuda' if torch.cuda.is_available() else 'cpu'):
        all_predictions = []
        all_probabilities = []
        all_attention_weights = []

        X_tensor = torch.FloatTensor(X).to(device)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                result = model.predict_with_explanation(X_tensor, device)
                all_predictions.append(result['predictions'])
                all_probabilities.append(result['probabilities'])
                if result['attention_weights'] is not None:
                    all_attention_weights.append(result['attention_weights'])

        avg_probabilities = None

        if self.ensemble_method == 'hard_voting':
            predictions_array = np.array(all_predictions)
            ensemble_predictions = []
            for i in range(predictions_array.shape[1]):
                votes = predictions_array[:, i]
                ensemble_predictions.append(np.bincount(votes).argmax())
            ensemble_predictions = np.array(ensemble_predictions)

        else:
            probabilities_array = np.array(all_probabilities)
            avg_probabilities = np.mean(probabilities_array, axis=0)
            ensemble_predictions = np.argmax(avg_probabilities, axis=1)

        avg_attention = np.mean(all_attention_weights, axis=0) if all_attention_weights else None

        return {
            'predictions': ensemble_predictions,
            'probabilities': avg_probabilities,
            'attention_weights': avg_attention,
            'individual_predictions': all_predictions,
            'individual_probabilities': all_probabilities
        }

    def evaluate(self, X, y, device='cuda' if torch.cuda.is_available() else 'cpu'):
        results = self.predict(X, device)
        predictions = results['predictions']

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)

        print(f"\n集成模型性能 ({self.ensemble_method}):")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")

        counter = Counter(y)
        minority_classes = [cls for cls, count in counter.items() if count < np.median(list(counter.values())) * 0.5]

        if minority_classes:
            print(f"\n少数类性能分析:")
            for cls in minority_classes:
                mask = y == cls
                if np.any(mask):
                    cls_accuracy = accuracy_score(y[mask], predictions[mask])
                    cls_samples = np.sum(mask)
                    print(f"  类别 {cls}: {cls_accuracy:.4f} ({cls_samples}个样本)")

        return results
