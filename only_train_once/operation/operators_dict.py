OPERATOR_DICT = {
    "Conv": {
        "Zero-Invariant": "TRUE",
        "Type": "Stem",
        "Shape-Dependent": "N/A"
    },
    "Gemm": {
        "Zero-Invariant": "TRUE",
        "Type": "Stem",
        "Shape-Dependent": "N/A"
    },
    "Linear": {
        "Zero-Invariant": "TRUE",
        "Type": "Stem",
        "Shape-Dependent": "N/A"
    },
    "BatchNormalization": {
        "Zero-Invariant": "TRUE",
        "Type": "Auxilary",
        "Shape-Dependent": "N/A"
    },
    "BatchNorm": {
        "Zero-Invariant": "TRUE",
        "Type": "Auxilary",
        "Shape-Dependent": "N/A"
    },
    "Relu": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "LeakyRelu": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "Abs": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "Sin": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "Sign": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "Sqrt": {
        "Zero-Invariant": "TRUE",
        "Type": "Activation",
        "Shape-Dependent": "N/A"
    },
    "Pad": {
        "Zero-Invariant": "TRUE",
        "Type": "Auxiliary",
        "Shape-Dependent": "N/A"
    },
    "Add": {
        "Zero-Invariant": "TRUE",
        "Type": "Joint",
        "Shape-Dependent": "TRUE"
    },
    "Mul": {
        "Zero-Invariant": "TRUE",
        "Type": "Joint",
        "Shape-Dependent": "TRUE"
    },
    "Concat": {
        "Zero-Invariant": "TRUE",
        "Type": "Joint",
        "Shape-Dependent": "FALSE"
    },
    "MaxPool": {
        "Zero-Invariant": "TRUE",
        "Type": "Pool",
        "Shape-Dependent": "FALSE"
    },
    "AveragePool": {
        "Zero-Invariant": "TRUE",
        "Type": "Pool",
        "Shape-Dependent": "FALSE"
    },
    "Dropout": {
        "Zero-Invariant": "TRUE",
        "Type": "Dropout",
        "Shape-Dependent": "FALSE"
    },
    "Shape": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": "FALSE"
    },
    "Reshape": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": "FALSE"
    },
    "Transpose": {
        "Zero-Invariant": "FALSE",
        "Type": "Others",
        "Shape-Dependent": "FALSE"
    },
    "Constant": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": ""
    },   
    "Upsample": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": ""
    }, 
    "Gather": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": "",
        "NOT-SURE": "TRUE"
    },   
    "Unsqueeze": {
        "Zero-Invariant": "TRUE",
        "Type": "Others",
        "Shape-Dependent": "",
        "NOT-SURE": "TRUE"
    },   
    "GlobalAveragePool": {
        "Zero-Invariant": "TRUE",
        "Type": "Pool",
        "Shape-Dependent": "",
        "NOT-SURE": "TRUE"
    },   
    "": {
        "Zero-Invariant": "",
        "Type": "",
        "Shape-Dependent": ""
    }

}