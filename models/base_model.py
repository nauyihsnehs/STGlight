import pytorch_lightning as pl
import torch


class BaseModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def configure_optimizers(self):
        lr = self.config.get('optimizer', {}).get('lr', 1e-3)
        weight_decay = self.config.get('optimizer', {}).get('weight_decay', 0)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return optimizer

        scheduler_type = scheduler_config.get('type', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 10)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_config.get('monitor', 'val_loss'),
                "interval": scheduler_config.get('interval', 'epoch'),
                "frequency": scheduler_config.get('frequency', 1)
            }
        }


class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._models[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get_model(cls, name, config):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry. Available models: {list(cls._models.keys())}")
        return cls._models[name](config)

    @classmethod
    def list_models(cls):
        return list(cls._models.keys())
