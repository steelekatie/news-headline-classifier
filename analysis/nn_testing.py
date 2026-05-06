# =============================================================================
# 0. NEURAL MODEL TESTING - learned embeddings + MLP w/ PyTorch
# =============================================================================
# SUMMARY: Builds/trains a lightweight neural classifier from scratch.
#       No pretrained models used.
#       Built directly from the training vocabulary.
#
# Structure: token embedding lookup --> mean pooling --> linear layer
#       Two builds compared:
#       Neural_Text:   text embeddings only
#       Neural_Hybrid: text embeddings + 12 style features (like Hybrid_v2_style)
#
# RESULTS: Does not manage to outperform our ensembles.

# -- Imports -------------------------------------------------------------
import os
import re
import unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_data, clean_hed, extract_style, STYLE_FEATURE_NAMES


RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. MODEL
# =============================================================================

EMBED_DIM  = 128 # embedding dimension — small since vocab is not massive
STYLE_DIM  = 12  # matches extract_style() output width
HIDDEN_DIM = 256 # MLP hidden layer width
DROPOUT    = 0.3

class NeuralText(nn.Module):
    """
    Text-only model
    Look up embedding, pool mean over token dim, fit linear classifier
    PAD tokens excluded from the mean to avoid dilution. PAD tokens = index 0
    """

    def __init__(self, vocab_size, embed_dim = EMBED_DIM, 
                 hidden_dim = HIDDEN_DIM, dropout = DROPOUT):
        super().__init__()
        # zero out PAD to exclude from gradients
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, token_ids, style=None):
        # token_ids are (batch, seq_len)
        emb = self.embedding(token_ids)
        # masknon-padding
        mask = (token_ids != 0).unsqueeze(-1) 
        # get mean over non-pad
        emb = (emb * mask).sum(dim = 1) / mask.sum(dim = 1).clamp(min = 1) 
        return self.classifier(emb)


class NeuralHybrid(nn.Module):
    """
    Hybrid model: text embeddings + style features
    Same embedding + pooling as NeuralText, then concatenates style features
    before the classifier, similar to hybrid_v2_style
    """

    def __init__(self, vocab_size, embed_dim = EMBED_DIM, style_dim = STYLE_DIM,
                 hidden_dim = HIDDEN_DIM, dropout = DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + style_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, token_ids, style):
        emb = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        emb = (emb * mask).sum(dim = 1) / mask.sum(dim = 1).clamp(min = 1)
        x = torch.cat([emb, style], dim = 1) # add pooled text and style
        return self.classifier(x)

# =============================================================================
# 2. LOAD DATA
# =============================================================================

X, y = prepare_data("expanded_headlines.csv", verbose = True)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X.tolist(), y, 
    test_size = 0.2, 
    random_state = RANDOM_STATE, 
    stratify = y
)
    
# =============================================================================
# 3. SETUP - vocabulary, encoding, dataset loaders
# =============================================================================

# -- Helper functions -------------------------------------------------------------
PAD_TOKEN = "<PAD>"   # index 0 - pad sequences to equal length
UNK_TOKEN = "<UNK>"   # index 1 - tokens not seen in training

def build_vocab(texts, min_freq = 2):
    """
    Index mapping from training headlines. tokens -> index dictionary
    Tokens extracted from whitespace split on clean_hed() output;
    tokens appearing fewer than min_freq times are mapped to UNK.
    """
    counts = Counter()
    for text in texts:
        counts.update(tokenize(clean_hed(text)))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def tokenize(text):
    """Splits cleaned text on whitespace into list of token strings."""
    return text.split()


def encode(texts, vocab, max_len = 30):
    """
    Encodes list of raw headline strings to padded integer array.
    Unknown tokens mapped to UNK index.
    max_len: tokens per hed. We see max = 25. Gives 5-word buffer for unseen heds.
    Shape: (n_samples, max_len)
    """
    unk_idx = vocab[UNK_TOKEN]
    encoded = np.zeros((len(texts), max_len), dtype = np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize(clean_hed(text))[:max_len]
        for j, token in enumerate(tokens):
            encoded[i, j] = vocab.get(token, unk_idx)
    return encoded

# -- NN additional preprocessing -----------------------------------------------
# vocabulary from training data only
vocab = build_vocab(X_train_raw, min_freq = 2)
print(f"Vocabulary size: {len(vocab)}")

# tokens
X_train_enc = encode(X_train_raw, vocab)
X_test_enc  = encode(X_test_raw,  vocab)

# style features - need to convert to series for extract_style() from preprocess.py
style_train_raw = extract_style(pd.Series(X_train_raw)) 
style_test_raw  = extract_style(pd.Series(X_test_raw))

# standardize style features
style_scaler = StandardScaler()
style_train  = style_scaler.fit_transform(style_train_raw).astype(np.float32)
style_test   = style_scaler.transform(style_test_raw).astype(np.float32)

# -- Loaders -------------------------------------------------------------------

def make_loader(enc, style, labels, batch_size = 256, shuffle = False):
    dataset = TensorDataset(
        torch.tensor(enc,    dtype = torch.long),
        torch.tensor(style,  dtype = torch.float32),
        torch.tensor(labels, dtype = torch.long),
    )
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

train_loader = make_loader(X_train_enc, style_train, y_train, shuffle = True)
test_loader  = make_loader(X_test_enc,  style_test,  y_test)

# =============================================================================
# 4. Training
# =============================================================================

def train_model(model, train_loader, test_loader, 
                epochs = 30, lr = 1e-3, patience = 5):
    """
    Trains with Adam, cross-entropy loss, and early stopping on val macro F1.
    Returns best model state and a DataFrame of per-epoch metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, patience_ctr = 0.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):

        # Train -------
        model.train()
        train_loss = 0.0
        for token_ids, style_batch, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(token_ids, style_batch), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate ------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for token_ids, style_batch, labels in test_loader:
                preds = model(token_ids, style_batch).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        f1  = f1_score(all_labels, all_preds, average="macro")
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        history.append({"epoch": epoch, "loss": avg_loss, "f1_mean": f1, "acc_mean": acc})
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val F1 {f1:.4f} | val acc {acc:.4f}")

        if f1 > best_f1:
            best_f1, best_state, patience_ctr = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} (best val F1: {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# -- Train both builds (text, text+ style)----------------------------------------

VOCAB_SIZE = len(vocab)

print("\n" + "=" * 60)
print("Training Neural_Text (embeddings only)")
print("=" * 60)
model_text = NeuralText(VOCAB_SIZE)
model_text, history_text = train_model(model_text, 
                                       train_loader, 
                                       test_loader)

# ============================================================
# Training Neural_Text (embeddings only)
# ============================================================
# Epoch   1 | loss 0.6473 | val F1 0.6557 | val acc 0.6576
# Epoch   2 | loss 0.5700 | val F1 0.7108 | val acc 0.7118
# Epoch   3 | loss 0.4962 | val F1 0.7262 | val acc 0.7276
# Epoch   4 | loss 0.4308 | val F1 0.7433 | val acc 0.7434
# Epoch   5 | loss 0.3803 | val F1 0.7529 | val acc 0.7540
# Epoch   6 | loss 0.3330 | val F1 0.7565 | val acc 0.7570
# Epoch   7 | loss 0.2906 | val F1 0.7645 | val acc 0.7654
# Epoch   8 | loss 0.2533 | val F1 0.7689 | val acc 0.7704
# Epoch   9 | loss 0.2211 | val F1 0.7684 | val acc 0.7687
# Epoch  10 | loss 0.1960 | val F1 0.7648 | val acc 0.7656
# Epoch  11 | loss 0.1692 | val F1 0.7631 | val acc 0.7648
# Epoch  12 | loss 0.1486 | val F1 0.7663 | val acc 0.7672
# Epoch  13 | loss 0.1312 | val F1 0.7624 | val acc 0.7633
#   Early stopping at epoch 13 (best val F1: 0.7689)

print("\n" + "=" * 60)
print("Training Neural_Hybrid (embeddings + style features)")
print("=" * 60)
model_hybrid = NeuralHybrid(VOCAB_SIZE)
model_hybrid, history_hybrid = train_model(model_hybrid, 
                                           train_loader, 
                                           test_loader)

# ============================================================
# Training Neural_Hybrid (embeddings + style features)
# ============================================================
# Epoch   1 | loss 0.5768 | val F1 0.7269 | val acc 0.7284
# Epoch   2 | loss 0.5052 | val F1 0.7442 | val acc 0.7479
# Epoch   3 | loss 0.4516 | val F1 0.7601 | val acc 0.7620
# Epoch   4 | loss 0.3953 | val F1 0.7733 | val acc 0.7752
# Epoch   5 | loss 0.3428 | val F1 0.7734 | val acc 0.7743
# Epoch   6 | loss 0.2952 | val F1 0.7773 | val acc 0.7780
# Epoch   7 | loss 0.2542 | val F1 0.7714 | val acc 0.7717
# Epoch   8 | loss 0.2169 | val F1 0.7791 | val acc 0.7801
# Epoch   9 | loss 0.1829 | val F1 0.7731 | val acc 0.7736
# Epoch  10 | loss 0.1563 | val F1 0.7803 | val acc 0.7817
# Epoch  11 | loss 0.1209 | val F1 0.7745 | val acc 0.7752
# Epoch  12 | loss 0.1009 | val F1 0.7749 | val acc 0.7756
# Epoch  13 | loss 0.0821 | val F1 0.7810 | val acc 0.7827
# Epoch  14 | loss 0.0680 | val F1 0.7739 | val acc 0.7752
# Epoch  15 | loss 0.0559 | val F1 0.7782 | val acc 0.7797
# Epoch  16 | loss 0.0541 | val F1 0.7767 | val acc 0.7777
# Epoch  17 | loss 0.0537 | val F1 0.7765 | val acc 0.7775
# Epoch  18 | loss 0.0384 | val F1 0.7742 | val acc 0.7752
#   Early stopping at epoch 18 (best val F1: 0.7810)

# =============================================================================
# 5. Evaluation
# =============================================================================

def evaluate_model(model, loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for token_ids, style_batch, labels in loader:
            preds = model(token_ids, style_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    final_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Final pipeline: {name}")
    print(f"Test accuracy: {final_accuracy:.4f}")
    print(
        "Classification Report:\n",
        classification_report(all_labels, all_preds, target_names = ["NBC", "FoxNews"]),
    )
    return all_preds, all_labels


preds_text,   labels_text   = evaluate_model(model_text,   test_loader, "Neural_Text")
# Final pipeline: Neural_Text
# Test accuracy: 0.7704
# Classification Report:
#                precision    recall  f1-score   support

#          NBC       0.79      0.79      0.79      2911
#      FoxNews       0.75      0.75      0.75      2468

#     accuracy                           0.77      5379
#    macro avg       0.77      0.77      0.77      5379
# weighted avg       0.77      0.77      0.77      5379

preds_hybrid, labels_hybrid = evaluate_model(model_hybrid, test_loader, "Neural_Hybrid")
# Final pipeline: Neural_Hybrid
# Test accuracy: 0.7827
# Classification Report:
#                precision    recall  f1-score   support

#          NBC       0.80      0.80      0.80      2911
#      FoxNews       0.77      0.76      0.76      2468

#     accuracy                           0.78      5379
#    macro avg       0.78      0.78      0.78      5379
# weighted avg       0.78      0.78      0.78      5379


neural_results = pd.DataFrame(
    [
        {
            "pipeline": "Neural_Text",
            "f1_mean":  round(f1_score(labels_text,   preds_text,   average="macro"), 4),
            "f1_std":   None,
            "acc_mean": round(accuracy_score(labels_text,   preds_text),   4),
            "acc_std":  None,
        },
        {
            "pipeline": "Neural_Hybrid",
            "f1_mean":  round(f1_score(labels_hybrid, preds_hybrid, average="macro"), 4),
            "f1_std":   None,
            "acc_mean": round(accuracy_score(labels_hybrid, preds_hybrid), 4),
            "acc_std":  None,
        },
    ]
)
print("\nNeural model comparison:")
print(neural_results.to_string(index=False))
# Neural model comparison:
#      pipeline  f1_mean f1_std  acc_mean acc_std
#   Neural_Text   0.7689   None    0.7704    None
# Neural_Hybrid   0.7810   None    0.7827    None

# -- Training curves -----------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, history, name in zip(
    axes,
    [history_text, history_hybrid],
    ["Neural_Text", "Neural_Hybrid"],
):
    ax.plot(history["epoch"], history["f1_mean"],  label="Val Macro F1")
    ax.plot(history["epoch"], history["acc_mean"], label="Val Accuracy")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Training Curves — Learned Embeddings + MLP", fontsize=13)
plt.tight_layout()
plt.show()
# We see both models converge nicely with plateaus, likely in part 
# due to dropout and early stopping. 

# F1 and accuracy are basically identical on both plots throughout.

# neural_text converges faster, but neural_hybrid plateaus a little higher.
# This confirms that our style features add useful signals to our models.

# Both models approach but do not meet our ensemble performance.
# Neural_Hybrid: 0.781 F1 vs. Ensemble_Stack: 0.804 F1

# The neural models, which are simple and built from scratch,
# do get surprisingly close to our ensemble results.

# However, given that we only have ~21k headlines to train on, 
# we are not convinced that we can provide enough data for a NN 
# to outperform our ensemble.