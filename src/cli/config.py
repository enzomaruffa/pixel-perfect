"""Configuration management for the pixel-perfect CLI."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuration for a processing pipeline."""
    
    input: Optional[str] = Field(None, description="Input image path")
    output: Optional[str] = Field(None, description="Output image path")
    cache_dir: Optional[str] = Field(None, description="Cache directory path")
    debug: bool = Field(False, description="Enable debug output")
    
    operations: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of operations to apply"
    )


class CacheConfig(BaseModel):
    """Cache configuration options."""
    
    enabled: bool = Field(True, description="Enable caching")
    max_size_mb: Optional[int] = Field(None, description="Maximum cache size in MB")
    max_age_days: int = Field(7, description="Maximum age of cache entries in days")
    auto_cleanup: bool = Field(True, description="Automatically clean up old entries")


class GlobalConfig(BaseModel):
    """Global configuration for pixel-perfect."""
    
    cache: CacheConfig = Field(default_factory=CacheConfig)
    default_cache_dir: Optional[str] = Field(None, description="Default cache directory")
    presets_dir: Optional[str] = Field(None, description="User presets directory")


class ConfigManager:
    """Manages configuration files and user settings."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.config_dir = Path.home() / ".pixel-perfect"
        self.config_file = self.config_dir / "config.yaml"
        self.presets_dir = self.config_dir / "presets"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.presets_dir.mkdir(exist_ok=True)
    
    def load_global_config(self) -> GlobalConfig:
        """Load global configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config_data = yaml.safe_load(f) or {}
                return GlobalConfig(**config_data)
            except Exception:
                # If loading fails, return default config
                pass
        
        return GlobalConfig()
    
    def save_global_config(self, config: GlobalConfig) -> None:
        """Save global configuration."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
    
    def load_pipeline_config(self, config_path: Path) -> PipelineConfig:
        """Load pipeline configuration from file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path) as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    config_data = json.load(f)
                else:
                    raise ValueError("Config file must be YAML or JSON")
            
            return PipelineConfig(**config_data)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def save_pipeline_config(self, config: PipelineConfig, config_path: Path) -> None:
        """Save pipeline configuration to file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config.model_dump(exclude_none=True), f, default_flow_style=False)
                elif config_path.suffix.lower() == ".json":
                    json.dump(config.model_dump(exclude_none=True), f, indent=2)
                else:
                    raise ValueError("Config file must be YAML or JSON")
        
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
    
    def list_user_presets(self) -> list[str]:
        """List user-defined presets."""
        preset_files = []
        for file_path in self.presets_dir.glob("*.yaml"):
            preset_files.append(file_path.stem)
        for file_path in self.presets_dir.glob("*.yml"):
            preset_files.append(file_path.stem)
        return sorted(preset_files)
    
    def load_user_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a user-defined preset."""
        preset_file = None
        
        # Try different extensions
        for ext in [".yaml", ".yml"]:
            candidate = self.presets_dir / f"{preset_name}{ext}"
            if candidate.exists():
                preset_file = candidate
                break
        
        if not preset_file:
            raise FileNotFoundError(f"User preset '{preset_name}' not found")
        
        try:
            with open(preset_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load preset: {e}")
    
    def save_user_preset(self, preset_name: str, preset_data: Dict[str, Any]) -> None:
        """Save a user-defined preset."""
        preset_file = self.presets_dir / f"{preset_name}.yaml"
        
        try:
            with open(preset_file, "w") as f:
                yaml.dump(preset_data, f, default_flow_style=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save preset: {e}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from file (convenience function)."""
    manager = ConfigManager()
    config = manager.load_pipeline_config(config_path)
    return config.model_dump()


def save_config(config_data: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to file (convenience function)."""
    manager = ConfigManager()
    config = PipelineConfig(**config_data)
    manager.save_pipeline_config(config, config_path)


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration file."""
    return {
        "pipeline": {
            "input": "input.jpg",
            "output": "output.jpg",
            "cache_dir": "./cache",
            "debug": True
        },
        "operations": [
            {
                "type": "PixelFilter",
                "params": {
                    "condition": "prime",
                    "fill_color": [255, 0, 0, 255]
                }
            },
            {
                "type": "RowShift",
                "params": {
                    "selection": "odd",
                    "shift_amount": 3,
                    "wrap": True
                }
            },
            {
                "type": "PixelMath",
                "params": {
                    "expression": "r * 1.2",
                    "channels": ["r"],
                    "clamp": True
                }
            }
        ]
    }