import sys, torch
sys.path.insert(0, ".")
from grinvi.tokenizer_morph import GrinViMorphTokenizer
tok = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
text = "n:?\n답변:"
tokens = tok.encode(text)
print("Encoded tokens:", tokens)
for t in tokens:
    print(f"Token {t}: {tok.decode([t])}")
