# Model Hyperparameters
vocab_size = 155            # This shouldn't be changed unless training data changes and more chars appear
block_size = 128            # How many characters the model processes at one time
d_model = 512               # Each character is turned into a d_model dimensional vector
n_heads = 4                 # Number of heads
n_layers = 4                # Amount of times the transformer block is run
dropout = 0.1               # random dropout
batch_size = 64             # batch size for training
learning_rate = 6e-4        # how much gradients change the model in training

# Training
max_steps = 20_000            # How many steps in total the model will be trained
eval_interval = 200         # The model will be evaluated every eval_interval

# Use "cuda" for Nvidia gpu, "mps" for Apple Silicon, or "cpu" for CPU
device = "cuda"