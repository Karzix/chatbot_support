from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
import re
import torch
from transformers import AutoModel, AutoTokenizer
from elasticsearch import Elasticsearch
import numpy as np
import unicodedata
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")
modelGMN = genai.GenerativeModel("gemini-2.0-flash")
# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200/")
INDEX_NAME = "hdsd"  

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0] 
    
    return vector
def is_numbered_heading(para):
    text = para.text.strip()
    if not text:  # Kiểm tra xem đoạn văn có nội dung không
        return False

    if para.style.name.lower().startswith("heading"):
        return True

    # Kiểm tra nếu tất cả các run có chữ đều in đậm
    all_bold = True  # Giả định ban đầu là tất cả đều in đậm
    for run in para.runs:
        if run.text.strip():  # Chỉ xét những `run` có nội dung thực
            if not run.bold:  # Nếu có bất kỳ `run` nào không in đậm, đặt all_bold = False
                all_bold = False

    return all_bold
def normalize_text(text):
    return unicodedata.normalize("NFC", text).lower()


def extract_docx_content(file_path):
    doc = Document(file_path)
    extracted_data = []
    current_title = None
    current_answer = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if text == "Đăng nhập hệ thống":
            print("DNHT")
        # print(text)
        # Kiểm tra nếu đoạn văn bản có dạng chỉ mục tiêu đề
        match = re.match(r"^([IVXLCDM]+\.\d+(\.\d+)*) (.+)$", text)
        if match or is_numbered_heading(para):
            if current_title and current_answer:
                extracted_data.append({"title": normalize_text(current_title), "answer": current_answer})
            
            current_title = match.group(3) if match else text  # Lấy tiêu đề đầy đủ
            current_answer = None  # Reset answer
        elif current_title and text:  # Nếu có tiêu đề trước đó và nội dung không rỗng, thì đây là hướng dẫn sử dụng
            if current_answer is None:
                current_answer = text
            else:
                current_answer += " " + text

    # Thêm dữ liệu cuối cùng nếu có
    if current_title and current_answer:
        extracted_data.append({"title": current_title, "answer": current_answer})

    return extracted_data




# Hàm lưu dữ liệu vào Elasticsearch
def save_to_elasticsearch(index_name, data):
    # with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            title = item["title"]
            # f.write(title + "\n")  # Ghi tiêu đề vào file
            
            title_vector = text_to_vector(title)
            answer_vector = text_to_vector(item["answer"])
            doc = {
                "title": title,
                "title_vector": title_vector,
                "answer": item["answer"],
                "answer_vector": answer_vector,
            }
            es.index(index=index_name, body=doc)
    
    # print("OK, titles saved to", output_file)

file_mapping = {
    "mappings": {
        "properties": {
            "file_name": {"type": "text"},
            "uploaded_at": {"type": "date"}
        }
    }
}

# Tạo index nếu chưa có
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body={
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "answer": {"type": "text"},
                "title_vector": {"type": "dense_vector", "dims": 768},
                "answer_vector": {"type": "dense_vector", "dims": 768}
            }
        }
    })
    print("✅ Đã tạo index `hdsd` với `dense_vector`!")
else:
    print("⚠️ Index `hdsd` đã tồn tại, không cần tạo lại.")

if not es.indices.exists(index="file"):
    es.indices.create(index="file", body=file_mapping)
    print("✅ Đã tạo index `file` để lưu thông tin file!")
else:
    print("⚠️ Index `file` đã tồn tại, không cần tạo lại.")
# Chạy chương trình
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)
    
    # Lưu thông tin file vào Elasticsearch
    file_doc = {
        "file_name": file.filename,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    es.index(index="file", body=file_doc)
    
    data = extract_docx_content(file_path)
    save_to_elasticsearch(INDEX_NAME, data)
    os.remove(file_path)
    return jsonify({"message": "File processed successfully", "file_info": file_doc})
@app.route("/files", methods=["GET"])
def get_files():
    query = {
        "size": 100,
        "sort": [{"uploaded_at": {"order": "desc"}}]
    }
    response = es.search(index="file", body=query)
    files = [{"file_name": hit["_source"]["file_name"], "uploaded_at": hit["_source"]["uploaded_at"]} for hit in response["hits"]["hits"]]
    return jsonify(files)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


