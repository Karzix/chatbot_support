from docx import Document
import re
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from elasticsearch import Elasticsearch
import numpy as np
import unicodedata
import google.generativeai as genai

# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "hdsd"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")

# Chọn mô hình Gemini Pro (miễn phí)
modelGMN = genai.GenerativeModel("gemini-2.0-flash")



def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # vector gốc (768 chiều)
    
    return vector

def search_in_elasticsearch(query, index_name, top_k=1):
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
query = "làm thế nào để tiếp nhận ngoại trú"
query = normalize_text(query)
index_name = "_all"


def refine_answer(answer):
    response = modelGMN.generate_content("Câu hỏi của người dùng: " + query + " - Đáp án mẫu: " + answer 
    + "\nHãy chuyển đáp án mẫu lại sao cho tự nhiên hơn")
    print(response.text)
    return response



if es.ping():
    print("🔹 Elasticsearch đã kết nối thành công!")
    results = search_in_elasticsearch(query, index_name)
    
    # In kết quả sau khi đã cải thiện câu trả lời bằng Llama
    for i, result in enumerate(results, 1):
        print(f"\n🔹 Cur {result['answer']}:")
        improved_answer = refine_answer(result['answer'])
        # print(f"🔹 Improved {improved_answer.text}")
        
else:
    print("⚠️ Không thể kết nối Elasticsearch!")