from .mnist import mnist_conv, mnist_mlp
from .cifar import cifar10_conv, cifar10_conv_anal
from .vae_mnist_mlp import vae_mnist_mlp
from .vae_mnist_conv import vae_conv
from .vision_transformer import vision_transformer_imagenet_1k
try:
    from .vit_imagenet_parallel import vision_transformer_imagenet_1k_parallel
except Exception:
    pass