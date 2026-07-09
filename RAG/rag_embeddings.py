from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "how hot is outside right now in lahore."
sentence2 = "tell me the weather in Lahore."

embeddings1 = model.encode(sentence1)
embeddings2 = model.encode(sentence2)

score = util.cos_sim(embeddings1, embeddings2)

print(score)

# print(len(embeddings1))
# print(len(embeddings2))
