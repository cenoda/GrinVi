#!/bin/bash
# Setup Google Gemini FREE tier for Korean responses (5 minutes)

cat << 'EOF'

════════════════════════════════════════════════════════════════════════════
         🇰🇷 GOOGLE GEMINI FREE - Korean Response Generator
════════════════════════════════════════════════════════════════════════════

This is 100% FREE:
  ✓ 60 requests per minute
  ✓ No credit card needed (real free)
  ✓ Excellent Korean quality
  ✓ Setup: 5 minutes

════════════════════════════════════════════════════════════════════════════
STEP 1: Get Free API Key (2 minutes)
════════════════════════════════════════════════════════════════════════════

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API key in new project"
3. Copy the key
4. Paste below when prompted

════════════════════════════════════════════════════════════════════════════
STEP 2: Install Library (1 minute)
════════════════════════════════════════════════════════════════════════════

pip install -q google-generativeai

════════════════════════════════════════════════════════════════════════════
STEP 3: Create Script (copy below)
════════════════════════════════════════════════════════════════════════════

EOF

cat > /home/cenoda/GrinVi/korean_chat_gemini.py << 'SCRIPT'
#!/usr/bin/env python3
"""
Korean Chat with Google Gemini (100% Free)
"""
import google.generativeai as genai

# GET YOUR FREE API KEY FROM: https://aistudio.google.com/app/apikey
API_KEY = input("Paste your Google Gemini API key: ").strip()

if not API_KEY:
    print("❌ No API key provided!")
    exit(1)

genai.configure(api_key=API_KEY)

print("\n✅ Connected to Google Gemini!")
print("=" * 70)
print("🇰🇷 Korean Chat (Type 'quit' to exit)")
print("=" * 70)

model = genai.GenerativeModel('gemini-pro')

system_prompt = "You are a helpful Korean assistant. Respond in Korean whenever possible."

while True:
    user_input = input("\n>>> ").strip()

    if user_input.lower() in ['quit', 'exit', '나가기']:
        print("안녕히 계세요!")
        break

    if not user_input:
        continue

    try:
        response = model.generate_content(
            f"{system_prompt}\n\nUser: {user_input}"
        )
        print(f"\n🤖 {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

SCRIPT

chmod +x /home/cenoda/GrinVi/korean_chat_gemini.py

cat << 'EOF'

════════════════════════════════════════════════════════════════════════════
STEP 4: Run It!
════════════════════════════════════════════════════════════════════════════

cd /home/cenoda/GrinVi
python korean_chat_gemini.py

Then type Korean prompts:
  안녕하세요
  한국에 대해 말해주세요
  서울의 날씨는 어때요?

════════════════════════════════════════════════════════════════════════════
OR: Use KoAlpaca Locally (100% Free, No API Key)
════════════════════════════════════════════════════════════════════════════

If you have Ollama installed:

1. Install: https://ollama.ai
2. Run: ollama run koalpaca
3. Chat locally, no internet needed!

════════════════════════════════════════════════════════════════════════════
COMPARISON
════════════════════════════════════════════════════════════════════════════

                    Google Gemini      KoAlpaca Local      Your GrinVi 1B
Cost                FREE               FREE                FREE (GPU)
Quality             ⭐⭐⭐⭐⭐          ⭐⭐⭐⭐⭐          ⭐⭐⭐⭐ (soon)
Korean              Perfect            Perfect             Good
Setup                5 min              10 min              Done in 34h
Internet            Required           NO                  NO
Speed               Cloud              GPU                 RTX 5080

════════════════════════════════════════════════════════════════════════════

Your best option TODAY:
  1. Use Google Gemini free (instant)
  2. Wait for GrinVi 1B (in 34 hours)
  3. Switch to local GrinVi (no costs, perfect for you!)

════════════════════════════════════════════════════════════════════════════

EOF

