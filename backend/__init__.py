from .vgg import vgg16, vgg16_bn
from .resnet_cifar import resnet50 as resnet50_cifar10
from .resnet_imagenet import resnet50 as resnet50_imagenet

def Model(backend, device, cfg=None):
    if backend =='vgg16':
        print('Backend: VGG16')
        model = vgg16() if cfg is None else vgg16(cfg=cfg)
    elif backend =='vgg16_bn':
        print('Backend: VGG16_BN')
        model = vgg16_bn() if cfg is None else vgg16_bn(cfg=cfg)
    elif backend == 'resnet50_cifar10':
        print('Backend: ResNet50_cifar10')
        model = resnet50_cifar10() if cfg is None else resnet50_cifar10(cfg=cfg)
    elif backend == 'resnet50_imagenet':
        print('Backend: resnet50_imagenet')
        model = resnet50_imagenet() if cfg is None else resnet50_imagenet(cfg=cfg)
    else:
        raise ValueError
    
    model = model.to(device)
    return model