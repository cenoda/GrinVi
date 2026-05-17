import sys
sys.path.insert(0, ".")
from grinvi.tokenizer_morph import GrinViMorphTokenizer
tok = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
print("UNK token ID:", tok.unk_token_id if hasattr(tok, 'unk_token_id') else tok.encode("<unk>"))
print("Decoded just 3:", tok.decode([3]))
