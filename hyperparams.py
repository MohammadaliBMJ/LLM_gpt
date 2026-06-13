class Hyperparams():
    # input
    embedding_size = 512
    max_length = 1024 #  Change based on tokenizer
    batch_size = 64
    vocab_size = 32000

    # Architecture
    layers = 6
    n_heads = 8
    mlp_hidden = embedding_size * 4
    pe = 'rope' #  Or 'ALiBi', 'learnable', 'sinusoidal'
    rope_base = 100000
    dropout = 0.1
    tokenizer = 'byte' #  Or 'BPE', 'unigram'

    # Training
    epochs = 10
    max_steps = 2000
    lr = 1e-4
    weight_decay = 0.001
    warmup_steps = 200


configs = Hyperparams()
