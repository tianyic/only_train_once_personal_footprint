DEFAULT_OPT_PARAMS = {
    "sgd": {
        "first_momentum": 0.0,
        "second_momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "lmbda": 1e-3,
        "lmbda_amplify": 2,
        "hat_lmbda_coeff": 10
    }
    ,
    "adam": {
        "lr": 1e-3,
        "first_momentum": 0.9,
        "second_momentum": 0.999,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "lmbda": 1e-2,
        "lmbda_amplify": 20,
        "hat_lmbda_coeff": 1e3
    }
    ,
    "adamw": {
        "lr": 1e-3,
        "first_momentum": 0.9,
        "second_momentum": 0.999,
        "dampening": 0.0,
        "weight_decay": 1e-2,
        "lmbda": 1e-2,
        "lmbda_amplify": 20,
        "hat_lmbda_coeff": 1e3
    }
}
