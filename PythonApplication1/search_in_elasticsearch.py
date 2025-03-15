from elasticsearch import Elasticsearch
import torch
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai
import numpy as np

# Cấu hình Gemini AI
genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")
modelGMN = genai.GenerativeModel("gemini-2.0-flash")

# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200/")

# Cấu hình model nhúng văn bản
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # Vector 768 chiều

# Câu hỏi đầu vào
query = "làm thế nào để tạo một phếu nhập viện"

# Chuyển query thành vector
query_vector = text_to_vector(query)

# Truy vấn Elasticsearch lấy 5 kết quả tốt nhất
search_query = {
    "size": 5,
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}

# Gửi truy vấn và nhận kết quả
response = es.search(index="hdsd", body=search_query)
results = response["hits"]["hits"]

# In kết quả tìm được
print("\n🔍 Kết quả tìm kiếm cho:", query)
for i, hit in enumerate(results, 1):
    print(f"\n🔹 Kết quả {i}:")
    print(f"   - Tiêu đề: {hit['_source']['title']}")
    print(f"   - Trả lời: {hit['_source']['answer']}")
    print(f"   - Điểm số: {hit['_score']:.4f}")

# Nếu không có đủ kết quả, dừng lại
if not results:
    print("\n⚠️ Không có kết quả phù hợp!")
else:
    # Tạo danh sách câu trả lời từ Elasticsearch
    answers = []
    for idx, result in enumerate(results):
        answers.append(f"{idx}️⃣ Đáp án {idx}: \"{result['_source']['answer']}\"")

    # Ghép tất cả các câu trả lời thành 1 chuỗi
    answers_text = "\n".join(answers)

    # Gửi yêu cầu đến Gemini AI
    prompt = f"""
    Tôi có một câu hỏi: "{query}".
    Dưới đây là các câu trả lời từ hệ thống:
    {answers_text}

    Hãy phân tích và chọn câu trả lời nào đúng nhất cho câu hỏi trên. 
    Nếu tất cả đều không phù hợp, hãy trả về "False".
    Nếu một câu phù hợp, hãy trả về số thứ tự (0, 1, 2, ...) của câu trả lời đó.
    Chỉ trả về số thứ tự hoặc "False", không có bất kỳ từ ngữ nào khác.
    """

    response = modelGMN.generate_content(prompt)
    gemini_result = response.text.strip()

    print("\n🔹 Kết quả từ Gemini AI:", gemini_result)
