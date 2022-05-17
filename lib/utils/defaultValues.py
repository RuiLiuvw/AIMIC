# Wheight = 800
# Wwidth = 680
TestMode = 0

ModelList = {
                'AlexNet': ['alexnet'],
                'Inception': ['inception3'],
                'MobileNet': ['mobilenet', 'mobilenet_v2', 'mobilenet_v3_large'],
                'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                'ResNeXt': ['resnext50_32x4d','resnext101_32x8d'],
                'VGG': ['vgg11', 'vgg11bn', 'vgg13', 'vgg13bn', 'vgg16', 'vgg16bn', 'vgg19', 'vgg19bn'],
                'ShuffleNet': ["shufflenet_v2_x0_5",'shufflenet_v2_x1_0','shufflenet_v2_x1_5','shufflenet_v2_x2_0'],
                'GoogleNet': ['googlenet'],
                'EfficientNet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                                 'efficientnet_b6','efficientnet_b7'],
                'ViT': ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14']
             }

OptimList = {'adam': ['adam', 'adamw', 'adamax'],
        'prop': ['rmsprop', 'rprop'],
        'adadelta': ['adadelta'], 'lbfgs': ['lbfgs'], 'adagrad': ['adagrad']}

DefaultPath = ['', '', '']
DefaultName = ['alexnet', 'adam', 'Mydataset']

DefaultValue = [0, 0, 0, 0, 0, 0.0000,0] if TestMode == 0 else [3, 1, 224, 50, 2, 0.0001,0]

DefaultPath_infer = ['','']

DefaultValue_infer = [8]



