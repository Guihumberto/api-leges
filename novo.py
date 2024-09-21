#teste
import ollama
import time

prompt = "teste novament4 laorem pararam ldsodsaoda rfdjofd kfjdishidfh kjdskjhfed ee"
start = time.perf_counter()
prompt_embbeding = ollama.embeddings(model="llama3.1", prompt=prompt)[
    "embedding"
]

print(time.perf_counter() - start)

print(len(prompt_embbeding))