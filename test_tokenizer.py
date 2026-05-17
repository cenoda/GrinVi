import sys
sys.path.insert(0, ".")
from grinvi.tokenizer_morph import GrinViMorphTokenizer
tok = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
text = "질문: 안녕?\n답변: 안녕하세요."
tokens = tok.encode(text)
decoded = tok.decode(tokens)
print("Encoded:", tokens)
print("Decoded:", decoded)
