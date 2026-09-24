"""
Configuration Manager for LLM Providers
Supports: Vertex AI, OpenAI, Anthropic Claude
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProviderConfig(BaseModel):
    """Base configuration for LLM providers"""
    provider: str
    model_name: str
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class VertexAIConfig(LLMProviderConfig):
    """Configuration for Google Vertex AI"""
    provider: str = "vertexai"
    project_id: str
    location: str = "us-central1"
    
    @classmethod
    def from_env(cls):
        return cls(
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            model_name=os.getenv("VERTEXAI_MODEL", "gemini-pro"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048"))
        )


class OpenAIConfig(LLMProviderConfig):
    """Configuration for OpenAI"""
    provider: str = "openai"
    api_key: str
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048"))
        )


class AnthropicConfig(LLMProviderConfig):
    """Configuration for Anthropic Claude"""
    provider: str = "anthropic"
    api_key: str
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048"))
        )


class ConfigManager:
    """Manages LLM configurations and provides easy switching"""
    
    # Predefined configurations for different use cases
    PRESETS = {
        "vertexai_default": {
            "provider": "vertexai",
            "model_name": "gemini-2.5-pro",
            "temperature": 0.2
        },
        "vertexai_creative": {
            "provider": "vertexai",
            "model_name": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "openai_precise": {
            "provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        "openai_balanced": {
            "provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 3000,
        },
        "openai_fast": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        "claude_precise": {
            "provider": "anthropic",
            "model_name": "claude-3-opus-20240229",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        "claude_balanced": {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        "claude_fast": {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
            "temperature": 0.3,
            "max_tokens": 2048,
        },
    }
    
    def __init__(self):
        self.current_config: Optional[Dict[str, Any]] = None
        self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration from environment"""
        provider = os.getenv("DEFAULT_PROVIDER", "vertexai")
        
        if provider == "vertexai":
            config = VertexAIConfig.from_env()
        elif provider == "openai":
            config = OpenAIConfig.from_env()
        elif provider == "anthropic":
            config = AnthropicConfig.from_env()
        else:
            # Default to Vertex AI
            config = VertexAIConfig.from_env()
        
        self.current_config = config.dict()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.current_config or {}
    
    def set_config(self, config: Dict[str, Any]):
        """Set custom configuration"""
        self.current_config = config
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a predefined configuration preset"""
        if preset_name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.PRESETS.keys())}")
        
        preset = self.PRESETS[preset_name].copy()
        
        # Add provider-specific credentials from environment
        provider = preset["provider"]
        if provider == "vertexai":
            preset["project_id"] = os.getenv("GCP_PROJECT_ID", "")
            preset["location"] = os.getenv("GCP_LOCATION", "us-central1")
        elif provider in ["openai", "anthropic"]:
            api_key_env = f"{provider.upper()}_API_KEY"
            preset["api_key"] = os.getenv(api_key_env, "")
        
        self.current_config = preset
        return preset
    
    def get_available_presets(self) -> list:
        """Get list of available configuration presets"""
        return list(self.PRESETS.keys())
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate configuration"""
        required_fields = ["provider", "model_name"]
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate provider-specific requirements
        provider = config["provider"]
        
        if provider == "vertexai":
            if not config.get("project_id"):
                return False, "Vertex AI requires project_id"
        elif provider in ["openai", "anthropic"]:
            if not config.get("api_key"):
                return False, f"{provider} requires api_key"
        else:
            return False, f"Unknown provider: {provider}"
        
        # Validate temperature
        temp = config.get("temperature", 0.3)
        if not 0 <= temp <= 1:
            return False, "Temperature must be between 0 and 1"
        
        return True, None
    
    def export_config(self, filepath: str = "llm_config.json"):
        """Export current configuration to JSON file"""
        import json
        
        with open(filepath, "w") as f:
            json.dump(self.current_config, f, indent=2)
        
        return filepath
    
    def import_config(self, filepath: str):
        """Import configuration from JSON file"""
        import json
        
        with open(filepath, "r") as f:
            config = json.load(f)
        
        is_valid, error = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")
        
        self.current_config = config
        return config


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager


# Example usage
if __name__ == "__main__":
    # Create config manager
    manager = ConfigManager()
    
    # Show available presets
    print("Available presets:")
    for preset in manager.get_available_presets():
        print(f"  - {preset}")
    
    # Load a preset
    print("\nLoading 'vertexai_default' preset...")
    config = manager.load_preset("vertexai_default")
    print(f"Current config: {config}")
    
    # Validate configuration
    is_valid, error = manager.validate_config(config)
    print(f"\nValidation: {'✓ Valid' if is_valid else f'✗ Invalid: {error}'}")
    
    # Export configuration
    exported_file = manager.export_config("my_config.json")
    print(f"\nConfiguration exported to: {exported_file}")