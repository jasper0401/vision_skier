from torch import nn
import timm

from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.models.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from src.models.shufflenet import ShuffleNetG2, ShuffleNetG3
from src.models.shufflenetv2 import shufflenet_v2
from src.models.mobilenet import MobileNet
from src.models.mobilenetv2 import MobileNetV2
from src.models.pnasnet import PNASNetA, PNASNetB
from src.models.efficientnet import EfficientNetB0
from src.models.dla import DLA
from src.models.dpn import DPN26, DPN92

from src.models.nano_clip_models import ImageEncoder, TextEncoder

def create_model(model_name: str, **model_args):
    assert model_name in globals() and callable(globals()[model_name]), "The model name is unknown!"
    model_func = globals()[model_name]
    model = model_func(**model_args)
    print (f"Initialized Model {model_name}.")
    return model 

def create_nano_clip(device, dtype):
    image_encoder = ImageEncoder().to(device=device, dtype=dtype)
    text_encoder = TextEncoder().to(device=device, dtype=dtype)
    return image_encoder, text_encoder

def timm_model(model_name: str, num_classes: int, drop_rate:float=.3, drop_path_rate:float=.1) -> nn.Module:
    model = timm.create_model(
        model_name, 
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    print (type(model))
    return model