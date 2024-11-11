import torch
from models.template_model import inception_v3, MLP, ConvergeEnd2EndModel, DisperseEnd2EndModel
from torchvision.models import resnet18, resnet50, resnet101, ResNet101_Weights

class DCBM(torch.nn.Module):
    def __init__(self, backbone, pretrained, freeze, num_classes, n_attributes, implicit_dim, use_aux, expand_dim, n_class_attr, 
                       use_relu=False, use_sigmoid=False):
        super(DCBM, self).__init__()
        three_class = n_class_attr == 3
        torch.hub.set_dir('~/.cache/torch/hub/')
        if backbone == "inceptionV3":
            self.cnn_Layer = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                                    n_attributes=n_attributes+implicit_dim, bottleneck=True, expand_dim=expand_dim,
                                    three_class=three_class)
        elif backbone == 'resnet50':
            self.cnn_Layer = resnet50(pretrained=pretrained)
        elif backbone == 'resnet18':
            self.cnn_Layer = resnet18(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.cnn_Layer = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2) # in line with ECBM: https://github.com/xmed-lab/ECBM
        self.explicit_MLP_Layer = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
        self.implicit_MLP_Layer = MLP(input_dim=implicit_dim, num_classes=num_classes, expand_dim=expand_dim)
        self.embedding_Layer = DisperseEnd2EndModel(self.cnn_Layer, self.explicit_MLP_Layer, self.implicit_MLP_Layer, 0, 
                            n_attributes, n_attributes, n_attributes+implicit_dim, use_relu, use_sigmoid, use_aux)

        ### the outputs of Converge_MLP_Layer are not used in the paper.
        self.converge_MLP_Layer = MLP(input_dim=num_classes*2, num_classes=num_classes, expand_dim=expand_dim)
        self.model = ConvergeEnd2EndModel(self.embedding_Layer, self.converge_MLP_Layer, 0, 0, use_relu, use_sigmoid, use_aux)

    def forward(self, x):
        return self.model(x)
