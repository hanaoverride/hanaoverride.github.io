from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 1. 모델과 토크나이저 초기화
model_name = "tiiuae/falcon-7b"  # Falcon 7B 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True, torch_dtype=torch.float16, device_map="auto")

# 2. 입력 텍스트 설정
texts = [
    "Most of the time travellers worry about their luggage",
    "Most of the time, travellers worry about their luggage.",
    "Let's eat grandma.",
    "Let's eat, grandma.",
    "Your donation just helped someone get a job.",
    "Your donation just helped someone. get a job.",
    "I like cooking, my family, and my dog.",
    "I like cooking my family and my dog.",
    "The panda eats shoots and leaves.",
    "The panda eats, shoots, and leaves."
]

# 3. 어텐션 가중치 추출
attention_data = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 어텐션 데이터 저장
    attention_data.append({
        "text": text,
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu()),
        "attentions": outputs.attentions  # 어텐션 맵
    })

# 4. 어텐션 맵 저장 함수

def save_attention_maps_and_text(attention_data, layer=0, head=0, output_dir="attention_maps"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, example in enumerate(attention_data):
        attention = example["attentions"][layer][0, head].cpu().numpy()  # Layer와 Head 선택
        tokens = example["tokens"]
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            cbar=True
        )
        plt.title(f"Attention Map (Layer {layer + 1}, Head {head + 1}, Example {i + 1})")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        
        # 이미지 저장
        image_file_path = os.path.join(output_dir, f"attention_map_{i + 1}.png")
        plt.savefig(image_file_path)
        plt.close()
        print(f"Saved image: {image_file_path}")
        
        # 텍스트 저장 (UTF-8 인코딩 명시)
        text_file_path = os.path.join(output_dir, f"attention_map_{i + 1}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:  # 여기에서 인코딩을 UTF-8로 설정
            f.write(f"Text: {example['text']}\n\n")
            f.write("Tokens:\n")
            f.write("\t".join(tokens) + "\n\n")
            f.write("Attention Map:\n")
            np.savetxt(f, attention, fmt="%.4f", delimiter="\t")
        print(f"Saved text: {text_file_path}")


# 5. 어텐션 맵 저장 실행
save_attention_maps_and_text(attention_data, layer=0, head=0)
