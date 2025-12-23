
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


def create_model(model_name: str, **model_args):
    assert model_name in globals() and callable(globals()[model_name]), "The model name is unknown!"
    model_func = globals()[model_name]
    model = model_func(**model_args)
    return model 