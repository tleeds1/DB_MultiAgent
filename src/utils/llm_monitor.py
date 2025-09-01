"""
LLM Monitor for quota management and model switching
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
import requests
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelType(Enum):
    """Types of LLM models"""
    GEMINI = "gemini"
    OPENAI = "openai"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    model_type: ModelType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    rate_limit_per_minute: int = 60
    is_available: bool = True


@dataclass
class UsageStats:
    """Usage statistics for a model"""
    model_name: str
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    last_used: Optional[str] = None
    average_response_time: float = 0.0


class LLMMonitor:
    """Monitor and manage LLM usage and model switching"""
    
    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.models: Dict[str, ModelConfig] = {}
        self.usage_stats: Dict[str, UsageStats] = {}
        self.current_model = None
        self.fallback_chain = []
        self.quota_limits = {}
        self.logger = self._setup_logger()
        
        # Load configuration
        self._load_config()
        self._initialize_models()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("LLMMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load model configurations
                for model_config in config.get('models', []):
                    model = ModelConfig(
                        name=model_config['name'],
                        model_type=ModelType(model_config['type']),
                        api_key=model_config.get('api_key'),
                        base_url=model_config.get('base_url'),
                        max_tokens=model_config.get('max_tokens', 4096),
                        temperature=model_config.get('temperature', 0.7),
                        cost_per_1k_tokens=model_config.get('cost_per_1k_tokens', 0.0),
                        rate_limit_per_minute=model_config.get('rate_limit_per_minute', 60),
                        is_available=model_config.get('is_available', True)
                    )
                    self.models[model.name] = model
                
                # Load quota limits
                self.quota_limits = config.get('quota_limits', {})
                
                # Load fallback chain
                self.fallback_chain = config.get('fallback_chain', [])
                
                # Set current model
                self.current_model = config.get('current_model')
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        default_config = {
            'models': [
                {
                    'name': 'gemini-2.0-flash-exp',
                    'type': 'gemini',
                    'api_key': os.getenv('GEMINI_API'),
                    'max_tokens': 8192,
                    'temperature': 0.7,
                    'cost_per_1k_tokens': 0.00015,
                    'rate_limit_per_minute': 60,
                    'is_available': True
                },
                {
                    'name': 'gemini-1.5-flash',
                    'type': 'gemini',
                    'api_key': os.getenv('GEMINI_API'),
                    'max_tokens': 8192,
                    'temperature': 0.7,
                    'cost_per_1k_tokens': 0.000075,
                    'rate_limit_per_minute': 60,
                    'is_available': True
                },
                {
                    'name': 'llama-3.1-8b',
                    'type': 'llama',
                    'base_url': 'http://localhost:11434',
                    'max_tokens': 4096,
                    'temperature': 0.7,
                    'cost_per_1k_tokens': 0.0,
                    'rate_limit_per_minute': 120,
                    'is_available': True
                }
            ],
            'quota_limits': {
                'daily_cost_limit': 10.0,
                'daily_token_limit': 1000000,
                'monthly_cost_limit': 100.0,
                'monthly_token_limit': 10000000
            },
            'fallback_chain': [
                'gemini-2.0-flash-exp',
                'gemini-1.5-flash',
                'llama-3.1-8b'
            ],
            'current_model': 'gemini-2.0-flash-exp'
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            self._load_config()
        except Exception as e:
            self.logger.error(f"Failed to create default config: {e}")
    
    def _initialize_models(self):
        """Initialize model clients"""
        for model_name, model_config in self.models.items():
            # Initialize usage stats
            if model_name not in self.usage_stats:
                self.usage_stats[model_name] = UsageStats(model_name=model_name)
            
            # Check model availability
            if model_config.model_type == ModelType.GEMINI:
                if not model_config.api_key:
                    model_config.is_available = False
                    self.logger.warning(f"Gemini model {model_name} not available: no API key")
            
            elif model_config.model_type == ModelType.LLAMA:
                # Check if local Llama server is running
                if not self._check_llama_server(model_config.base_url):
                    model_config.is_available = False
                    self.logger.warning(f"Llama model {model_name} not available: server not running")
    
    def _check_llama_server(self, base_url: str) -> bool:
        """Check if Llama server is running"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [
            name for name, config in self.models.items()
            if config.is_available
        ]
    
    def get_current_model(self) -> Optional[str]:
        """Get current active model"""
        if self.current_model and self.models[self.current_model].is_available:
            return self.current_model
        
        # Try to find first available model
        available_models = self.get_available_models()
        if available_models:
            self.current_model = available_models[0]
            return self.current_model
        
        return None
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return False
        
        if not self.models[model_name].is_available:
            self.logger.error(f"Model {model_name} is not available")
            return False
        
        self.current_model = model_name
        self.logger.info(f"Switched to model: {model_name}")
        
        # Save configuration
        self._save_config()
        return True
    
    def get_next_fallback_model(self) -> Optional[str]:
        """Get next model in fallback chain"""
        if not self.fallback_chain:
            return None
        
        current_index = -1
        if self.current_model in self.fallback_chain:
            current_index = self.fallback_chain.index(self.current_model)
        
        # Try next model in chain
        for i in range(current_index + 1, len(self.fallback_chain)):
            model_name = self.fallback_chain[i]
            if model_name in self.models and self.models[model_name].is_available:
                return model_name
        
        return None
    
    def check_quota_limits(self, model_name: str, estimated_tokens: int = 0) -> Tuple[bool, str]:
        """Check if usage is within quota limits"""
        if model_name not in self.usage_stats:
            return True, "No usage data available"
        
        stats = self.usage_stats[model_name]
        model_config = self.models[model_name]
        
        # Calculate estimated cost
        estimated_cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
        
        # Check daily limits
        today = datetime.now().date().isoformat()
        daily_stats = self._get_daily_stats(model_name, today)
        
        daily_cost_limit = self.quota_limits.get('daily_cost_limit', float('inf'))
        daily_token_limit = self.quota_limits.get('daily_token_limit', float('inf'))
        
        if daily_stats['cost'] + estimated_cost > daily_cost_limit:
            return False, f"Daily cost limit exceeded: {daily_stats['cost']:.2f}/{daily_cost_limit:.2f}"
        
        if daily_stats['tokens'] + estimated_tokens > daily_token_limit:
            return False, f"Daily token limit exceeded: {daily_stats['tokens']}/{daily_token_limit}"
        
        # Check rate limiting
        if not self._check_rate_limit(model_name):
            return False, "Rate limit exceeded"
        
        return True, "Quota check passed"
    
    def _get_daily_stats(self, model_name: str, date: str) -> Dict[str, Any]:
        """Get daily usage statistics"""
        # This is a simplified implementation
        # In production, you'd want to store this in a database
        stats = self.usage_stats.get(model_name, UsageStats(model_name=model_name))
        
        # For now, return current stats (assuming they're from today)
        return {
            'cost': stats.total_cost,
            'tokens': stats.total_tokens,
            'requests': stats.total_requests
        }
    
    def _check_rate_limit(self, model_name: str) -> bool:
        """Check rate limiting"""
        model_config = self.models[model_name]
        stats = self.usage_stats[model_name]
        
        # Simple rate limiting based on requests per minute
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Count requests in last minute (simplified)
        recent_requests = 0
        if stats.last_used:
            try:
                last_used_time = datetime.fromisoformat(stats.last_used).timestamp()
                if last_used_time > minute_ago:
                    recent_requests = 1  # Simplified
            except:
                pass
        
        return recent_requests < model_config.rate_limit_per_minute
    
    def record_usage(self, model_name: str, tokens_used: int, cost: float, 
                    response_time: float, success: bool = True):
        """Record usage statistics"""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = UsageStats(model_name=model_name)
        
        stats = self.usage_stats[model_name]
        
        # Update statistics
        stats.total_requests += 1
        stats.total_tokens += tokens_used
        stats.total_cost += cost
        stats.last_used = datetime.now().isoformat()
        
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        # Update average response time
        if stats.total_requests > 1:
            stats.average_response_time = (
                (stats.average_response_time * (stats.total_requests - 1) + response_time) /
                stats.total_requests
            )
        else:
            stats.average_response_time = response_time
        
        self.logger.info(f"Usage recorded for {model_name}: {tokens_used} tokens, ${cost:.4f}, {response_time:.2f}s")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary across all models"""
        summary = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'success_rate': 0.0,
            'models': {}
        }
        
        for model_name, stats in self.usage_stats.items():
            summary['total_requests'] += stats.total_requests
            summary['total_tokens'] += stats.total_tokens
            summary['total_cost'] += stats.total_cost
            
            success_rate = 0.0
            if stats.total_requests > 0:
                success_rate = stats.successful_requests / stats.total_requests
            
            summary['models'][model_name] = {
                'requests': stats.total_requests,
                'tokens': stats.total_tokens,
                'cost': stats.total_cost,
                'success_rate': success_rate,
                'avg_response_time': stats.average_response_time,
                'last_used': stats.last_used
            }
        
        if summary['total_requests'] > 0:
            summary['success_rate'] = sum(
                stats.successful_requests for stats in self.usage_stats.values()
            ) / summary['total_requests']
        
        return summary
    
    def estimate_tokens(self, text: str, model_name: str) -> int:
        """Estimate token count for text"""
        if model_name not in self.models:
            return len(text.split()) * 1.3  # Rough estimate
        
        model_config = self.models[model_name]
        
        if model_config.model_type == ModelType.GEMINI:
            # Gemini token estimation (approximate)
            return len(text.split()) * 1.3
        
        elif model_config.model_type == ModelType.LLAMA:
            # Llama token estimation
            try:
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
                tokens = tokenizer.encode(text)
                return len(tokens)
            except:
                return len(text.split()) * 1.3
        
        return len(text.split()) * 1.3
    
    def monitor_call(self, func: Callable) -> Callable:
        """Decorator to monitor LLM calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            model_name = kwargs.get('model_name') or self.get_current_model()
            
            if not model_name:
                raise ValueError("No available model found")
            
            # Estimate tokens
            prompt = kwargs.get('prompt', '')
            estimated_tokens = self.estimate_tokens(prompt, model_name)
            
            # Check quota
            quota_ok, quota_message = self.check_quota_limits(model_name, estimated_tokens)
            if not quota_ok:
                self.logger.warning(f"Quota check failed: {quota_message}")
                
                # Try fallback model
                fallback_model = self.get_next_fallback_model()
                if fallback_model:
                    self.logger.info(f"Switching to fallback model: {fallback_model}")
                    kwargs['model_name'] = fallback_model
                    model_name = fallback_model
                else:
                    raise Exception(f"Quota exceeded and no fallback available: {quota_message}")
            
            try:
                # Make the call
                result = func(*args, **kwargs)
                
                # Record successful usage
                response_time = time.time() - start_time
                actual_tokens = self.estimate_tokens(str(result), model_name)
                model_config = self.models[model_name]
                cost = (actual_tokens / 1000) * model_config.cost_per_1k_tokens
                
                self.record_usage(model_name, actual_tokens, cost, response_time, True)
                
                return result
                
            except Exception as e:
                # Record failed usage
                response_time = time.time() - start_time
                self.record_usage(model_name, estimated_tokens, 0.0, response_time, False)
                
                # Try fallback if available
                fallback_model = self.get_next_fallback_model()
                if fallback_model and fallback_model != model_name:
                    self.logger.info(f"Retrying with fallback model: {fallback_model}")
                    kwargs['model_name'] = fallback_model
                    return func(*args, **kwargs)
                
                raise e
        
        return wrapper
    
    def _save_config(self):
        """Save current configuration"""
        config = {
            'models': [asdict(model) for model in self.models.values()],
            'quota_limits': self.quota_limits,
            'fallback_chain': self.fallback_chain,
            'current_model': self.current_model
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for model usage optimization"""
        recommendations = []
        summary = self.get_usage_summary()
        
        # Check for high costs
        if summary['total_cost'] > 5.0:
            recommendations.append("Consider using local models (Llama) to reduce costs")
        
        # Check for low success rates
        for model_name, stats in summary['models'].items():
            if stats['success_rate'] < 0.8:
                recommendations.append(f"Model {model_name} has low success rate ({stats['success_rate']:.1%})")
        
        # Check for slow response times
        for model_name, stats in summary['models'].items():
            if stats['avg_response_time'] > 5.0:
                recommendations.append(f"Model {model_name} is slow ({stats['avg_response_time']:.1f}s avg)")
        
        # Check for quota usage
        daily_cost_limit = self.quota_limits.get('daily_cost_limit', float('inf'))
        if summary['total_cost'] > daily_cost_limit * 0.8:
            recommendations.append("Approaching daily cost limit - consider cost optimization")
        
        return recommendations
