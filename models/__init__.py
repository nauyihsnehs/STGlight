from .vlsnet import VLSNet
from .vlsnet_tf import VLSNetTF
from .nfnrnet import NFNet, NRNet
from .nfnrnet_tf import NFNetTF, NRNetTF
from .alnet import ALNet
from .alnet_tf import ALNetTF
from .pointnext import PointNet
from .pointnext_tf import PointNetTF

from .base_model import ModelRegistry

__all__ = ['VLSNet', 'NFNet', 'ALNet', 'PointNet', 'NRNet', 'NFNetTF', 'NRNetTF',
           'ALNetTF', 'VLSNetTF', 'ModelRegistry']

AVAILABLE_MODELS = ModelRegistry.list_models()
