import re
import torch
import contractions
import nltk
from nltk.corpus import stopwords
from model import LSTMClassifier
import sys

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


#  Load vocab and model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word2idx = torch.load("models/word2idx.pt", map_location=device)
vocab_size = len(word2idx)

# Load the checkpoint robustly
# 1) Try weights-only (PyTorch >= 2.5). This avoids executing pickled code.
loaded_model = None
try:
    loaded_model = torch.load("models/model.pt", map_location=device, weights_only=True)
except TypeError:
    # Older PyTorch without weights_only
    pass
except Exception:
    # Any other unexpected error, fall back to safe path below
    loaded_model = None

# 2) If weights-only not available or failed, attempt normal load.
#    If the model was saved as a full pickled object referencing __main__.LSTMClassifier,
#    create a compatibility alias so unpickling can succeed.
if loaded_model is None:
    try:
        loaded_model = torch.load("models/model.pt", map_location=device, weights_only=False)
    except AttributeError:
        # Provide __main__.LSTMClassifier alias to resolve pickled references
        main_mod = sys.modules.get('__main__')
        if main_mod is not None and not hasattr(main_mod, 'LSTMClassifier'):
            setattr(main_mod, 'LSTMClassifier', LSTMClassifier)
        loaded_model = torch.load("models/model.pt", map_location=device, weights_only=False)

# Instantiate model with the same architecture
model = LSTMClassifier(vocab_size=vocab_size)

# If the loaded model is a state dict, load it directly
if isinstance(loaded_model, dict):
    model.load_state_dict(loaded_model)
else:
    # If it's a full model, extract its state dict
    model.load_state_dict(loaded_model.state_dict())

model.to(device)
model.eval()

#  Text cleaning and tokenization
def clean_text(text: str):
    text = contractions.fix(text)
    text = text.encode('ascii', 'ignore').decode()
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\b\w{1}\b", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return tokens

#  Encode and pad tokens
def encode(tokens, max_len=300):
    ids = [word2idx.get(w, word2idx.get('<UNK>', 1)) for w in tokens]
    if len(ids) < max_len:
        ids += [word2idx.get('<PAD>', 0)] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

#  Prediction helper
def predict(text: str) -> int:
    tokens = clean_text(text)
    seq = encode(tokens).to(device)
    with torch.no_grad():
        score = model(seq).item()
    return int(score >= 0.5)