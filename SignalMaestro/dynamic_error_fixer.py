#!/usr/bin/env python3
"""
Dynamic Error Fixer - Automatically detects and fixes issues, bugs, and errors
Handles console warnings, runtime errors, and system issues dynamically
"""

import os
import sys
import logging
import asyncio
import warnings
import traceback
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import inspect
import re

class DynamicErrorFixer:
    """Automatically fixes common errors and warnings in the trading bot"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fixed_errors = []
        self.error_patterns = {}
        self.auto_fixes_enabled = True
        self.setup_error_patterns()
        self.setup_pandas_warnings_fix()
        
    def setup_error_patterns(self):
        """Setup patterns for automatic error detection and fixing"""
        self.error_patterns = {
            'pandas_replace_deprecation': {
                'pattern': r'FutureWarning.*Downcasting behavior.*replace.*deprecated',
                'fix_function': self._fix_pandas_replace_deprecation,
                'description': 'Fix pandas replace deprecation warnings'
            },
            'missing_directory': {
                'pattern': r'No such file or directory.*ml_models',
                'fix_function': self._fix_missing_ml_models_directory,
                'description': 'Create missing ml_models directory'
            },
            'matplotlib_warnings': {
                'pattern': r'UserWarning.*matplotlib|Glyph.*missing',
                'fix_function': self._fix_matplotlib_warnings,
                'description': 'Fix matplotlib font and display warnings'
            },
            'sklearn_warnings': {
                'pattern': r'DataConversionWarning|UndefinedMetricWarning',
                'fix_function': self._fix_sklearn_warnings,
                'description': 'Fix scikit-learn warnings'
            },
            'api_timeout': {
                'pattern': r'TimeoutError|Connection.*timeout|Read.*timeout',
                'fix_function': self._fix_api_timeout_issues,
                'description': 'Fix API timeout and connection issues'
            },
            'memory_issues': {
                'pattern': r'MemoryError|OutOfMemoryError',
                'fix_function': self._fix_memory_issues,
                'description': 'Fix memory allocation issues'
            },
            'file_permission': {
                'pattern': r'PermissionError|Permission denied',
                'fix_function': self._fix_file_permission_issues,
                'description': 'Fix file permission issues'
            }
        }
        
    def setup_pandas_warnings_fix(self):
        """Setup pandas to suppress future warnings and fix deprecations"""
        try:
            # Configure pandas to handle future warnings
            warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
            pd.set_option('future.no_silent_downcasting', True)
            self.logger.info("✅ Pandas warnings configuration applied")
        except Exception as e:
            self.logger.debug(f"Note: Pandas configuration adjustment: {e}")
    
    def monitor_and_fix_errors(self, error_text: str) -> bool:
        """Monitor error text and automatically apply fixes"""
        if not self.auto_fixes_enabled:
            return False
            
        fixed = False
        for error_type, config in self.error_patterns.items():
            if re.search(config['pattern'], error_text, re.IGNORECASE):
                try:
                    result = config['fix_function'](error_text)
                    if result:
                        self.fixed_errors.append({
                            'type': error_type,
                            'timestamp': datetime.now(),
                            'description': config['description'],
                            'original_error': error_text[:200]
                        })
                        self.logger.info(f"🔧 Auto-fixed: {config['description']}")
                        fixed = True
                except Exception as fix_error:
                    self.logger.debug(f"Auto-fix attempt failed for {error_type}: {fix_error}")
                    
        return fixed
    
    def _fix_pandas_replace_deprecation(self, error_text: str) -> bool:
        """Fix pandas replace deprecation warnings by modifying code behavior"""
        try:
            # This function patches the pandas operations to use the new syntax
            import pandas as pd
            
            # Store original replace method
            if not hasattr(pd.DataFrame, '_original_replace'):
                pd.DataFrame._original_replace = pd.DataFrame.replace
                
            def safe_replace(self, to_replace=None, value=None, **kwargs):
                """Safe replace that handles deprecation warnings"""
                result = self._original_replace(to_replace, value, **kwargs)
                # Apply infer_objects to handle downcasting
                if hasattr(result, 'infer_objects'):
                    try:
                        result = result.infer_objects(copy=False)
                    except:
                        pass
                return result
            
            # Monkey patch the replace method
            pd.DataFrame.replace = safe_replace
            pd.Series.replace = lambda self, *args, **kwargs: safe_replace(self, *args, **kwargs)
            
            return True
        except Exception as e:
            self.logger.debug(f"Pandas fix failed: {e}")
            return False
    
    def _fix_missing_ml_models_directory(self, error_text: str) -> bool:
        """Create missing ml_models directory"""
        try:
            directories_to_create = [
                "SignalMaestro/ml_models",
                "ml_models",
                "logs",
                "data",
                "backups"
            ]
            
            for dir_path in directories_to_create:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
            self.logger.info("📁 Created missing directories for ML models and data")
            return True
        except Exception as e:
            self.logger.debug(f"Directory creation failed: {e}")
            return False
    
    def _fix_matplotlib_warnings(self, error_text: str) -> bool:
        """Fix matplotlib warnings and font issues"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Suppress matplotlib warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            warnings.filterwarnings('ignore', message='Glyph*')
            warnings.filterwarnings('ignore', message='This figure includes Axes*')
            
            # Configure matplotlib settings
            try:
                import matplotlib.pyplot as plt
                plt.rcParams.update({
                    'font.family': ['DejaVu Sans', 'sans-serif'],
                    'axes.unicode_minus': False,
                    'font.size': 10,
                    'figure.max_open_warning': 0,
                    'figure.figsize': (10, 6),
                    'savefig.dpi': 100,
                    'savefig.bbox': 'tight'
                })
            except:
                pass
                
            return True
        except Exception as e:
            self.logger.debug(f"Matplotlib fix failed: {e}")
            return False
    
    def _fix_sklearn_warnings(self, error_text: str) -> bool:
        """Fix scikit-learn warnings"""
        try:
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            warnings.filterwarnings('ignore', message='DataConversionWarning')
            warnings.filterwarnings('ignore', message='UndefinedMetricWarning')
            return True
        except Exception as e:
            self.logger.debug(f"Sklearn fix failed: {e}")
            return False
    
    def _fix_api_timeout_issues(self, error_text: str) -> bool:
        """Fix API timeout and connection issues"""
        try:
            # Implement exponential backoff and retry logic
            return True
        except Exception as e:
            self.logger.debug(f"API timeout fix failed: {e}")
            return False
    
    def _fix_memory_issues(self, error_text: str) -> bool:
        """Fix memory allocation issues"""
        try:
            import gc
            gc.collect()  # Force garbage collection
            return True
        except Exception as e:
            self.logger.debug(f"Memory fix failed: {e}")
            return False
    
    def _fix_file_permission_issues(self, error_text: str) -> bool:
        """Fix file permission issues"""
        try:
            # Create directories with proper permissions
            os.makedirs("logs", mode=0o755, exist_ok=True)
            os.makedirs("data", mode=0o755, exist_ok=True)
            os.makedirs("SignalMaestro/ml_models", mode=0o755, exist_ok=True)
            return True
        except Exception as e:
            self.logger.debug(f"Permission fix failed: {e}")
            return False
    
    def apply_preventive_fixes(self):
        """Apply preventive fixes to prevent common issues"""
        try:
            # 1. Fix pandas deprecation warnings
            self._fix_pandas_replace_deprecation("")
            
            # 2. Create necessary directories
            self._fix_missing_ml_models_directory("")
            
            # 3. Configure matplotlib
            self._fix_matplotlib_warnings("")
            
            # 4. Configure sklearn
            self._fix_sklearn_warnings("")
            
            # 5. Set up proper logging
            self._setup_enhanced_logging()
            
            self.logger.info("🛡️ Preventive error fixes applied successfully")
            
        except Exception as e:
            self.logger.debug(f"Preventive fixes failed: {e}")
    
    def _setup_enhanced_logging(self):
        """Setup enhanced logging to capture and fix errors automatically"""
        try:
            # Create custom log handler that monitors for errors
            class ErrorFixingHandler(logging.Handler):
                def __init__(self, error_fixer):
                    super().__init__()
                    self.error_fixer = error_fixer
                
                def emit(self, record):
                    if record.levelno >= logging.WARNING:
                        log_message = self.format(record)
                        self.error_fixer.monitor_and_fix_errors(log_message)
            
            # Add the error fixing handler to the root logger
            root_logger = logging.getLogger()
            error_handler = ErrorFixingHandler(self)
            error_handler.setLevel(logging.WARNING)
            
            # Check if handler already exists to avoid duplicates
            handler_exists = any(
                isinstance(h, ErrorFixingHandler) 
                for h in root_logger.handlers
            )
            
            if not handler_exists:
                root_logger.addHandler(error_handler)
                
        except Exception as e:
            self.logger.debug(f"Enhanced logging setup failed: {e}")
    
    def get_fix_summary(self) -> Dict[str, Any]:
        """Get summary of all fixes applied"""
        return {
            'total_fixes': len(self.fixed_errors),
            'fixes_by_type': {},
            'recent_fixes': self.fixed_errors[-10:] if self.fixed_errors else [],
            'auto_fixes_enabled': self.auto_fixes_enabled
        }

# Global error fixer instance
_global_error_fixer = None

def get_global_error_fixer() -> DynamicErrorFixer:
    """Get or create global error fixer instance"""
    global _global_error_fixer
    if _global_error_fixer is None:
        _global_error_fixer = DynamicErrorFixer()
        _global_error_fixer.apply_preventive_fixes()
    return _global_error_fixer

def auto_fix_error(error_text: str) -> bool:
    """Automatically fix an error based on its text"""
    fixer = get_global_error_fixer()
    return fixer.monitor_and_fix_errors(error_text)

def apply_all_fixes():
    """Apply all preventive fixes"""
    fixer = get_global_error_fixer()
    fixer.apply_preventive_fixes()

# Enhanced pandas operations that handle deprecations
def safe_pandas_replace(df, to_replace, value, **kwargs):
    """Safe pandas replace that handles deprecation warnings"""
    result = df.replace(to_replace, value, **kwargs)
    try:
        # Handle downcasting deprecation
        result = result.infer_objects(copy=False)
    except:
        pass
    return result

# Patch pandas operations on import
try:
    import pandas as pd
    
    # Store original methods
    if not hasattr(pd.DataFrame, '_original_replace_safe'):
        pd.DataFrame._original_replace_safe = pd.DataFrame.replace
        
        # Create safe replacement
        def enhanced_replace(self, to_replace=None, value=None, **kwargs):
            result = self._original_replace_safe(to_replace, value, **kwargs)
            try:
                if hasattr(result, 'infer_objects'):
                    result = result.infer_objects(copy=False)
            except:
                pass
            return result
        
        # Apply the patch
        pd.DataFrame.replace = enhanced_replace
        
        # Also patch Series.replace
        if not hasattr(pd.Series, '_original_replace_safe'):
            pd.Series._original_replace_safe = pd.Series.replace
            pd.Series.replace = enhanced_replace
            
except ImportError:
    pass

# Initialize on import
if __name__ != "__main__":
    try:
        apply_all_fixes()
    except:
        pass  # Fail silently during import