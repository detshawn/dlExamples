from model import *

# config
_detailed = False
_epochs = 20
_batch_size = 100

# 1. Dataset (Import)
m = 11000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
# print(dataset[:10])

# 2. Pre-processing
Tx = 30
Ty = 10

X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape: ", X.shape) # (m, Tx)
print("X.shape: ", Y.shape) # (m, Ty)
print("Xoh.shape: ", Xoh.shape) # (m, Tx, len(human_vocab))
print("Yoh.shape: ", Yoh.shape) # (m, Ty, len(machine_vocab))

if _detailed:
    index = 0
    print("Source date:", dataset[index][0])
    print("Target date:", dataset[index][1])
    print()
    print("Source after preprocessing (indices):", X[index])
    print("Target after preprocessing (indices):", Y[index])
    print()
    print("Source after preprocessing (one-hot):", Xoh[index])
    print("Target after preprocessing (one-hot):", Yoh[index])

n_a = 32  # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64  # number of units for the post-attention LSTM's hidden state "s"

# 3. Create model
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

# 4. Compile model
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Fit model
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

hist = model.fit([Xoh, s0, c0], outputs, validation_split=1/11,
                 epochs=_epochs, batch_size=_batch_size)

plot_hist(hist)

ATTENTION_WEIGHTS_LAYER_IDX = 8
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab,
                                   "Tuesday 09 Oct 1993",
                                   num=ATTENTION_WEIGHTS_LAYER_IDX, n_s=n_s)
