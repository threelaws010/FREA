from sentence_transformers import SentenceTransformer

# Load model (downloads automatically if not present)
model = SentenceTransformer("allenai/specter2")

# Encode text to embeddings
text = "Deep learning for protein folding"
embedding = model.encode(text)
print(len(embedding), embedding[:5])
