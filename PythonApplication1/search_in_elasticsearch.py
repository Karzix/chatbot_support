from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import torch
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai
import unicodedata
from flask_cors import CORS

# Khởi tạo Flask
app = Flask(__name__)
CORS(app)
# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "hdsd"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")
modelGMN = genai.GenerativeModel("gemini-2.0-flash")

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # vector gốc (768 chiều)
    return vector

def search_in_elasticsearch(query, index_name, top_k=1):
    query_vector = text_to_vector(query)
    search_query = {
        "size": top_k,
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
    response = es.search(index=index_name, body=search_query)
    results = [
        {
            "title": hit["_source"]["title"],
            "answer": hit["_source"]["answer"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results

def refine_answer(query, answer):
    response = modelGMN.generate_content(f"Câu hỏi của người dùng: {query} - Đáp án mẫu: {answer}\nHãy chuyển đáp án mẫu lại sao cho tự nhiên hơn")
    return response.text

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get("query", "")
    query = unicodedata.normalize("NFC", query).lower()
    if not query:
        return jsonify({"error": "Query không được để trống."}), 400

    if not es.ping():
        return jsonify({"error": "Không thể kết nối Elasticsearch!"}), 500
    
    results = search_in_elasticsearch(query, INDEX_NAME)
    
    if not results:
        return jsonify({"message": "Không tìm thấy kết quả phù hợp."})
    
    best_result = results[0]  # Lấy kết quả tốt nhất
    improved_answer = refine_answer(query, best_result['answer'])
    
    return jsonify({
        "query": query,
        "original_answer": best_result['answer'],
        "refined_answer": improved_answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
