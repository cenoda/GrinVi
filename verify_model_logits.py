import sys, torch
sys.path.insert(0, ".")
from grinvi.model import GrinViModel, GrinViConfig
# 아무 가중치나 랜덤하게 초기화된 모델 생성 (step 0의 모델과 똑같음)
config = GrinViConfig.tiny() 
model = GrinViModel(config)
model.eval()
# "질문: " 이라는 단어 하나만 던져봄
dummy_input = torch.tensor([[1, 3607, 41]]) # bos, 질문, :
with torch.no_grad():
    logits = model(dummy_input)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
# 모델이 다음에 <unk> 를 예측할 확률 (token_id=3)
unk_prob = probs[3].item() * 100
print("--- 검증 2: 초기 모델의 무작위성 검증 ---")
print(f"방금 태어난 랜덤 모델이 <unk>를 찍어서 뱉을 기본 확률: {unk_prob:.4f}%")
print(f"가장 높게 예측된 토큰 확률: {probs.max().item() * 100:.4f}%")
print(f"64000개 단어를 무작위로 찍을 때 기대 확률: {1/64000*100:.4f}%")
