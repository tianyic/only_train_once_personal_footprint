from .transforms import Rename, ConvBNFuse

# PyTorch Graph Transforms
FRAMEWORK_TRANSFORMS = [
    Rename(op=r"onnx::(.*)", to=r"\1"),
    Rename(op=r"gemm", to=r"linear"),
    Rename(op=r"batchnormalization", to="batchnorm"),
]

CONV_BN_FUSE = ConvBNFuse("conv > batchnorm", "convbn")
