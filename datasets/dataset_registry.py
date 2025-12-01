from typing import Dict, Type
from omegaconf import DictConfig
from .base_dataset import BaseDataModule


class DatasetRegistry:
    _datasets: Dict[str, Type[BaseDataModule]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(dataset_cls: Type[BaseDataModule]) -> Type[BaseDataModule]:
            cls._datasets[name] = dataset_cls
            return dataset_cls
        return decorator
    
    @classmethod
    def get_dataset(cls, name: str, config: DictConfig) -> BaseDataModule:
        if name not in cls._datasets:
            raise ValueError(f"Dataset {name} not found in registry. Available datasets: {list(cls._datasets.keys())}")
        
        return cls._datasets[name](config)
    
    @classmethod
    def list_datasets(cls) -> list:
        return list(cls._datasets.keys())