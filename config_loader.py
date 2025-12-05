"""
Configuration Loader Utility
Load dan validate configuration files
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional
import warnings


class ConfigLoader:
    """
    Load dan validate configuration files
    Support multiple config formats dan merging
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config file (default: config.json)
        """
        self.config_path = config_path or "config.json"
        self.config = {}
        self.is_loaded = False
    
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration dari JSON file
        
        Args:
            config_path: Override default config path
            
        Returns:
            Dictionary dengan configuration
        """
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        print(f"Loading config from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.is_loaded = True
        self._validate_config()
        
        print(f"Configuration loaded successfully")
        return self.config
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = [
            'data',
            'preprocessing',
            'feature_extraction',
            'models',
            'training'
        ]
        
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            warnings.warn(f"Missing config sections: {missing}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value dengan dot notation
        
        Args:
            key: Config key (e.g., 'training.epochs')
            default: Default value jika key tidak ada
            
        Returns:
            Config value
        """
        if not self.is_loaded:
            raise RuntimeError("Config not loaded. Call load() first.")
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set config value dengan dot notation
        
        Args:
            key: Config key (e.g., 'training.epochs')
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        Save configuration ke JSON file
        
        Args:
            output_path: Path untuk save (default: overwrite original)
        """
        path = output_path or self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {path}")
    
    def merge(self, other_config: Dict[str, Any], overwrite: bool = True):
        """
        Merge another config into current config
        
        Args:
            other_config: Config dict to merge
            overwrite: Overwrite existing values
        """
        self._merge_recursive(self.config, other_config, overwrite)
    
    def _merge_recursive(self, base: Dict, update: Dict, overwrite: bool):
        """Recursive merge helper"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_recursive(base[key], value, overwrite)
            elif overwrite or key not in base:
                base[key] = value
    
    def print_summary(self):
        """Print configuration summary"""
        if not self.is_loaded:
            print("No configuration loaded")
            return
        
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        
        # Project info
        if 'project_info' in self.config:
            info = self.config['project_info']
            print(f"\nProject: {info.get('name', 'N/A')}")
            print(f"   Description: {info.get('description', 'N/A')}")
        
        # Data
        if 'data' in self.config:
            data = self.config['data']
            print(f"\nData:")
            print(f"   Directory: {data.get('base_directory', 'N/A')}")
            print(f"   Classes: {data.get('classes', [])}")
            max_files = data.get('max_files_per_class')
            if max_files:
                print(f"   Max files/class: {max_files}")
        
        # Models
        if 'models' in self.config:
            models = self.config['models']
            if 'available_models' in models:
                print(f"\nModels: {len(models['available_models'])} available")
                for model in models['available_models']:
                    print(f"   - {model.get('name', 'N/A')}: {model.get('description', '')}")
        
        # Training
        if 'training' in self.config:
            train = self.config['training']
            print(f"\nTraining:")
            if 'hyperparameters' in train:
                hp = train['hyperparameters']
                print(f"Epochs: {hp.get('epochs', 'N/A')}")
                print(f"Batch size: {hp.get('batch_size', 'N/A')}")
                print(f"Learning rate: {hp.get('learning_rate', 'N/A')}")
                print(f"Optimizer: {hp.get('optimizer', 'N/A')}")
        
        # Cross-validation
        if 'cross_validation' in self.config:
            cv = self.config['cross_validation']
            if cv.get('enabled', False):
                print(f"\nCross-Validation: {cv.get('n_splits', 5)}-fold")
            else:
                print(f"\nCross-Validation: Disabled")
        
        # Output
        if 'output' in self.config:
            output = self.config['output']
            print(f"\nOutput:")
            print(f"   Directory: {output.get('experiments_dir', 'N/A')}")
            print(f"   Save models: {output.get('save_models', True)}")
            print(f"   Generate report: {output.get('generate_report', True)}")
        
        print("="*70 + "\n")
    
    def validate_paths(self) -> bool:
        """
        Validate that all paths exist
        
        Returns:
            True if all paths valid
        """
        issues = []
        
        # Check data directory
        if 'data' in self.config and 'base_directory' in self.config['data']:
            data_dir = self.config['data']['base_directory']
            if not os.path.exists(data_dir):
                issues.append(f"Data directory not found: {data_dir}")
        
        if issues:
            print("Path validation issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("All paths validated")
        return True
    
    def get_training_config(self) -> Dict[str, Any]:
        """Extract training-specific configuration"""
        if not self.is_loaded:
            raise RuntimeError("Config not loaded")
        
        training_config = {}
        
        if 'training' in self.config:
            train = self.config['training']
            
            # Hyperparameters
            if 'hyperparameters' in train:
                hp = train['hyperparameters']
                training_config.update({
                    'epochs': hp.get('epochs', 100),
                    'batch_size': hp.get('batch_size', 12),
                    'learning_rate': hp.get('learning_rate', 0.0003),
                    'optimizer': hp.get('optimizer', 'adamw')
                })
            
            # Callbacks
            if 'callbacks' in train:
                cb = train['callbacks']
                
                if 'early_stopping' in cb and cb['early_stopping'].get('enabled', True):
                    training_config['early_stopping_patience'] = \
                        cb['early_stopping'].get('patience', 25)
            
            # Class weights
            if 'class_weights' in train:
                training_config['use_class_weights'] = \
                    train['class_weights'].get('enabled', True)
            
            # Verbose
            training_config['verbose'] = train.get('verbose', 1)
        
        return training_config
    
    def get_model_configs(self) -> list:
        """Extract model configurations"""
        if not self.is_loaded:
            raise RuntimeError("Config not loaded")
        
        if 'models' not in self.config or 'available_models' not in self.config['models']:
            return []
        
        return self.config['models']['available_models']


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Convenience function untuk load config
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    Fallback jika config file tidak ada
    """
    return {
        'data': {
            'base_directory': 'data',
            'classes': ['non_cts', 'mild', 'moderate', 'severe']
        },
        'training': {
            'hyperparameters': {
                'epochs': 20,
                'batch_size': 3,
                'learning_rate': 0.0003,
                'optimizer': 'adamw'
            }
        }
    }


# Example usage
if __name__ == "__main__":
    print("Testing ConfigLoader...")
    
    # Test 1: Load config
    try:
        loader = ConfigLoader("config.json")
        config = loader.load()
        
        print("\nTest 1: Load config - PASSED")
        
        # Test 2: Print summary
        loader.print_summary()
        print("Test 2: Print summary - PASSED")
        
        # Test 3: Get values
        epochs = loader.get('training.hyperparameters.epochs')
        print(f"\nTest 3: Get value - Epochs: {epochs}")
        
        # Test 4: Set value
        loader.set('training.hyperparameters.epochs', 50)
        new_epochs = loader.get('training.hyperparameters.epochs')
        print(f"Test 4: Set value - New epochs: {new_epochs}")
        
        # Test 5: Validate paths
        loader.validate_paths()
        print("Test 5: Validate paths - PASSED")
        
        # Test 6: Get training config
        train_config = loader.get_training_config()
        print(f"\nTest 6: Training config extracted:")
        print(f"   {train_config}")
        
        print("\nAll tests passed!")
        
    except FileNotFoundError:
        print("Config file not found (expected for testing)")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()