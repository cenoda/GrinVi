import sys, torch
sys.path.insert(0, ".")
from grinvi.model import GrinViModel, GrinViConfig
config = GrinViConfig.tiny() 
model = GrinViModel(config)
model.eval()
# "질문: " 이라는 단어 하나만 던져봄
dummy_input = torch.tensor([[1, 3607, 41]]) # bos, 질문, :
with torch.no_grad():
    out = model(dummy_input)
    # Some models return logits directly, some return a tuple
    logits = out[0] if isinstance(out, tuple) else out
    probs = torch.softmax(logits[0, -1, :], dim=-1)
unk_prob = probs[3].item() * 100
print("--- 검증 2: 초기 모델의 무작위 확률 출력 증명 ---")
print(f"방금 만들어진 모델이 <unk>를 무작위로 뱉을 확률: {unk_prob:.4f}%")
print(f"가장 높게 예측된 임의의 단어 확률: {probs.max().item() * 100:.4f}%")
print(f"64,000개 사전을 눈 감고 무작위로 찍을 때 수학적 확률: {1/64000*100:.4f}%")
