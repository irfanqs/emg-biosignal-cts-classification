"""
Focal Loss Implementation for Imbalanced Classification
Helps model focus on hard-to-classify samples and minority classes
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class FocalLoss:
    """
    Focal Loss untuk menangani class imbalance
    
    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Parameters:
        alpha (float or list): Weighting factor for classes (default: 0.25)
                              Can be a list of weights per class
        gamma (float): Focusing parameter (default: 2.0)
                      Higher gamma = more focus on hard examples
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False):
        """
        Initialize Focal Loss
        
        Args:
            alpha (float or list): Class weighting factor(s)
            gamma (float): Focusing parameter (2.0 recommended)
            from_logits (bool): Whether predictions are logits or probabilities
        """
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def __call__(self, y_true, y_pred):
        """
        Compute focal loss
        
        Args:
            y_true: Ground truth labels (sparse format: [batch_size])
            y_pred: Predicted probabilities (shape: [batch_size, num_classes])
        
        Returns:
            Focal loss value
        """
        # Convert sparse labels to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss components
        # p_t: probability of true class
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        # Cross entropy: -log(p_t)
        ce_loss = -tf.math.log(p_t)
        
        # Alpha weighting
        if isinstance(self.alpha, (list, tuple)):
            # Per-class alpha weights
            alpha_t = tf.reduce_sum(
                y_true_one_hot * tf.constant(self.alpha, dtype=tf.float32),
                axis=-1
            )
        else:
            # Single alpha value
            alpha_t = self.alpha
        
        # Combine all components: FL = alpha * (1-p_t)^gamma * CE
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Return mean loss across batch
        return tf.reduce_mean(focal_loss)


def get_focal_loss(num_classes=4, alpha=None, gamma=2.0, class_counts=None):
    """
    Get configured Focal Loss function
    
    Args:
        num_classes (int): Number of classes
        alpha (float or list): Alpha parameter(s)
                              If None, will calculate from class_counts
        gamma (float): Gamma parameter (default: 2.0)
        class_counts (dict): Class distribution for automatic alpha calculation
                           Example: {0: 102, 1: 138, 2: 264, 3: 179}
    
    Returns:
        FocalLoss instance
    """
    if alpha is None and class_counts is not None:
        # Calculate inverse frequency weights
        total = sum(class_counts.values())
        alpha = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            # Inverse frequency normalized
            weight = total / (num_classes * count)
            alpha.append(weight)
        
        print(f"\nFocal Loss Auto-calculated Alpha Weights:")
        for i, w in enumerate(alpha):
            print(f"  Class {i}: {w:.4f}")
    
    elif alpha is None:
        # Default balanced alpha
        alpha = 0.25
    
    return FocalLoss(alpha=alpha, gamma=gamma, from_logits=False)


# Keras-compatible wrapper
class KerasFocalLoss(tf.keras.losses.Loss):
    """
    Keras-compatible Focal Loss wrapper
    Can be used directly in model.compile(loss=KerasFocalLoss(...))
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    
    def call(self, y_true, y_pred):
        return self.focal_loss_fn(y_true, y_pred)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config
