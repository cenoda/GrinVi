import sys
sys.path.insert(0, ".")
from grinvi.tokenizer_morph import GrinViMorphTokenizer
tok = GrinViMorphTokenizer("data/raw/ko_wikipedia/ko_tokenizer.json")
unk_id = 3 # unk token id
total_tokens = 0
unk_count = 0
with open("data/processed/train.txt", "r", encoding="utf-8") as f:
    for i in range(1000): # First 1000 lines
        line = f.readline()
        if not line: break
        tokens = tok.encode(line)
        total_tokens += len(tokens)
        unk_count += tokens.count(unk_id)
print(f"--- 검증 1: 훈련 데이터 오염 여부 ---")
print(f"검사한 텍스트 줄 수: 1000 줄")
print(f"생성된 전체 토큰 수: {total_tokens:,} 개")
print(f"그 중 들어있는 <unk> 토큰 개수: {unk_count} 개")
print(f"오염률: {unk_count/total_tokens*100:.6f}%")
