class Hyperparams():
    # Training
    epochs = 10
    max_steps = 1024

    # Architecture
    layers = 6
    n_heads = 8
    mlp_hidden = embedding_size * 2
    pe = ['sinusoidal', 'learnable', 'rope', 'alibi']

    # input
    embedding_size = 512
    max_length = 1024
    batch_size = 64

    # Training
    lr = 1e-4
    weight_decay = 0.001
    