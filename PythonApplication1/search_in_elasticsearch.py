from docx import Document
import re
import torch
from transformers import AutoModel, AutoTokenizer
from elasticsearch import Elasticsearch
import numpy as np
import unicodedata


# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "hdsd"  # Đổi thành index của bạn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # vector gốc (768 chiều)
    
    # Giảm về 256 chiều bằng cách lấy trung bình nhóm (hoặc PCA)
    # vector = np.array(vector).reshape(256, 3).mean(axis=1).tolist()
    
    return vector

def search_in_elasticsearch(query, index_name, top_k=5):
    # Chuyển câu hỏi thành vector
    query_vector = text_to_vector(query)

    # Truy vấn Elasticsearch
    search_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},  # Lấy tất cả, sau đó tính điểm dựa trên vector
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    response = es.search(index=index_name, body=search_query)

    # Lấy danh sách kết quả
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "title": hit["_source"]["title"],
            "answer": hit["_source"]["answer"],
            "score": hit["_score"]
        })

    return results

def normalize_text(text):
    return unicodedata.normalize("NFC", text).lower()
query = "làm thế nào để đăng nhập vào hệ thống"
query = normalize_text(query)
index_name = "_all"

if es.ping():
    print("🔹 Elasticsearch đã kết nối thành công!")
    results = search_in_elasticsearch(query, index_name)
    
    # In kết quả
    for i, result in enumerate(results, 1):
        print(f"\n🔹 Kết quả {i}:")
        print(f"📌 Title: {result['title']}")
        print(f"📝 Answer: {result['answer']}")
        print(f"⭐ Score: {result['score']}")
else:
    print("⚠️ Không thể kết nối Elasticsearch!")