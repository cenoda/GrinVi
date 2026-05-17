import sys
sys.path.insert(0, ".")
from grinvi.tokenizer_morph import GrinViMorphTokenizer
tok = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
text1 = "질문: 한국의 수도는 어디인가요?\n답변:"  # Real newline
text2 = "질문: 한국의 수도는 어디인가요?\\n답변:" # Literal \n
print("Real newline:")
print(tok.decode(tok.encode(text1)))
print("Literal backslash n:")
print(tok.decode(tok.encode(text2)))
