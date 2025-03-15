from flask import Flask, request, jsonify  # Import Flask để tạo API
from flask_cors import CORS  # Import CORS để cho phép truy cập từ domain khác
from docx import Document  # Import để đọc file DOCX
import re  # Import thư viện regex để xử lý chuỗi
import torch  # Import PyTorch để sử dụng mô hình AI
from transformers import AutoModel, AutoTokenizer  # Import mô hình và tokenizer từ Hugging Face
from elasticsearch import Elasticsearch  # Import Elasticsearch để lưu và tìm kiếm dữ liệu
import numpy as np  # Import NumPy để xử lý mảng số học
import unicodedata  # Import để xử lý chuẩn hóa Unicode
import os  # Import os để làm việc với hệ thống file
import google.generativeai as genai  # Import Google Generative AI SDK
from datetime import datetime  # Import để xử lý thời gian


app = Flask(__name__)  # Khởi tạo ứng dụng Flask
CORS(app)  # Cho phép CORS để API có thể được gọi từ domain khác

# Thiết lập thiết bị cho mô hình AI (GPU nếu có, ngược lại dùng CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình và tokenizer để chuyển đổi văn bản thành vector
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Cấu hình API key cho Google Generative AI
genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")

# Khởi tạo mô hình Gemini
modelGMN = genai.GenerativeModel("gemini-2.0-flash")

# Kết nối đến Elasticsearch
es = Elasticsearch("http://localhost:9200/")

# Tên chỉ mục trong Elasticsearch
INDEX_NAME = "hdsd"

# Chuyển đổi văn bản thành vector sử dụng mô hình AI
def text_to_vector(text):
    text = normalize_text(text)  # Chuẩn hóa văn bản trước khi xử lý
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)  # Mã hóa văn bản thành tensor
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm tài nguyên
        outputs = model(**inputs)  # Chạy mô hình để trích xuất đặc trưng của văn bản
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # Lấy vector biểu diễn từ đầu ra của mô hình
    
    return vector  # Trả về vector biểu diễn của văn bản

# Kiểm tra đoạn văn có phải tiêu đề có đánh số không
def is_numbered_heading(para):
    text = para.text.strip()  # Lấy nội dung đoạn văn và loại bỏ khoảng trắng hai đầu
    if not text:  # Nếu nội dung rỗng, không phải tiêu đề
        return False

    if para.style.name.lower().startswith("heading"):  # Nếu đoạn văn có style bắt đầu bằng "heading" thì là tiêu đề
        return True

    all_bold = True  # Biến kiểm tra xem toàn bộ đoạn văn có in đậm không
    for run in para.runs:
        if run.text.strip():  # Nếu đoạn có nội dung
            if not run.bold:  # Nếu bất kỳ phần nào không in đậm thì không phải tiêu đề
                all_bold = False

    return all_bold  # Trả về kết quả kiểm tra

# Chuẩn hóa văn bản về dạng chuẩn NFC
def normalize_text(text):
    return unicodedata.normalize("NFC", text).lower()  # Chuyển văn bản về dạng chuẩn NFC và chuyển thành chữ thường


# Trích xuất nội dung từ file DOCX
def extract_docx_content(file):
    doc = Document(file)  # Mở file DOCX
    extracted_data = []  # Danh sách để lưu dữ liệu trích xuất
    current_title = None  # Biến lưu tiêu đề hiện tại
    current_answer = None  # Biến lưu nội dung hướng dẫn

    for para in doc.paragraphs:
        text = para.text.strip()  # Loại bỏ khoảng trắng đầu và cuối đoạn
        match = re.match(r"^([IVXLCDM]+\.\d+(\.\d+)*) (.+)$", text)  # Kiểm tra xem đoạn có phải tiêu đề có đánh số không
        if match or is_numbered_heading(para):  # Nếu là tiêu đề
            if current_title and current_answer:  # Nếu đã có tiêu đề và nội dung trước đó
                extracted_data.append({"title": normalize_text(current_title), "answer": current_answer})  # Lưu lại dữ liệu trước đó
            
            current_title = match.group(3) if match else text  # Cập nhật tiêu đề mới
            current_answer = None  # Reset nội dung câu trả lời
        elif current_title and text:  # Nếu không phải tiêu đề và có nội dung
            if current_answer is None:
                current_answer = text  # Gán nội dung mới
            else:
                current_answer += " " + text  # Nối tiếp nội dung

    if current_title and current_answer:  # Lưu lại mục cuối cùng nếu có
        extracted_data.append({"title": current_title, "answer": current_answer})

    return extracted_data  # Trả về danh sách dữ liệu trích xuất

# Lưu dữ liệu vào Elasticsearch
def save_to_elasticsearch(index_name, data):
    for item in data:
        title = item["title"]  # Lấy tiêu đề
        title_vector = text_to_vector(title)  # Chuyển tiêu đề thành vector
        answer_vector = text_to_vector(item["answer"])  # Chuyển nội dung hướng dẫn thành vector
        
        query = {"query": {"term": {"title.keyword": title}}}  # Truy vấn kiểm tra xem tiêu đề đã tồn tại chưa
        response = es.search(index=index_name, body=query)  # Tìm kiếm trong Elasticsearch
        
        if response["hits"]["hits"]:  # Nếu tiêu đề đã tồn tại
            doc_id = response["hits"]["hits"][0]["_id"]  # Lấy ID của tài liệu
            update_doc = {
                "doc": {
                    "answer": item["answer"],
                    "answer_vector": answer_vector
                }
            }
            es.update(index=index_name, id=doc_id, body=update_doc)  # Cập nhật nội dung
        else:
            doc = {
                "title": title,
                "title_vector": title_vector,
                "answer": item["answer"],
                "answer_vector": answer_vector,
            }
            es.index(index=index_name, body=doc)  # Lưu tài liệu mới vào Elasticsearch
    return "OK, data saved/updated in Elasticsearch"
    

# Định nghĩa cấu trúc mapping cho index "file" chứa thông tin về tên file và thời gian upload
file_mapping = {
    "mappings": {
        "properties": {
            "file_name": {"type": "text"},  # Tên file được lưu dưới dạng text
            "uploaded_at": {"type": "date"}  # Thời gian upload file, kiểu dữ liệu date
        }
    }
}

# Kiểm tra xem index `hdsd` đã tồn tại chưa, nếu chưa thì tạo mới
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body={
        "mappings": {
            "properties": {
                "title": {"type": "text"},  # Tiêu đề hướng dẫn
                "answer": {"type": "text"},  # Nội dung hướng dẫn
                "title_vector": {"type": "dense_vector", "dims": 768},  # Vector hóa tiêu đề (768 chiều)
                "answer_vector": {"type": "dense_vector", "dims": 768}  # Vector hóa nội dung hướng dẫn (768 chiều)
            }
        }
    })
    print("✅ Đã tạo index `hdsd` với `dense_vector`!")  # Thông báo tạo thành công
else:
    print("⚠️ Index `hdsd` đã tồn tại, không cần tạo lại.")  # Thông báo nếu index đã tồn tại

# Kiểm tra xem index "file" đã tồn tại chưa, nếu chưa thì tạo mới để lưu thông tin file
if not es.indices.exists(index="file"):
    es.indices.create(index="file", body=file_mapping)
    print("✅ Đã tạo index `file` để lưu thông tin file!")  # Thông báo tạo thành công
else:
    print("⚠️ Index `file` đã tồn tại, không cần tạo lại.")  # Thông báo nếu index đã tồn tại

# API tải tài liệu lên
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400  # Trả về lỗi nếu không có file được tải lên
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400  # Trả về lỗi nếu không có file nào được chọn
    
    # Trích xuất nội dung từ file DOCX
    data = extract_docx_content(file)
    
    # Lưu dữ liệu đã trích xuất vào Elasticsearch
    save_to_elasticsearch("hdsd", data)
    
    # Tạo tài liệu metadata cho file
    file_doc = {
        "file_name": file.filename,
        "uploaded_at": datetime.utcnow().isoformat()  # Lưu thời gian upload file
    }
    
    # Lưu metadata của file vào Elasticsearch
    es.index(index="file", body=file_doc)
    
    return jsonify({"message": "File processed successfully", "file_info": file_doc})  # Phản hồi thành công

# API lấy danh sách tài liệu
@app.route("/files", methods=["GET"])
def get_files():
    # Truy vấn Elasticsearch để lấy danh sách file đã upload, giới hạn 100 kết quả, sắp xếp theo thời gian tải lên mới nhất
    query = {
        "size": 100,
        "sort": [{"uploaded_at": {"order": "desc"}}]
    }
    
    response = es.search(index="file", body=query)  # Gửi truy vấn đến Elasticsearch
    
    # Trích xuất danh sách file từ kết quả truy vấn
    files = [{
        "file_name": hit["_source"]["file_name"], 
        "uploaded_at": hit["_source"]["uploaded_at"]
    } for hit in response["hits"]["hits"]]
    
    return jsonify(files)  # Trả về danh sách file dưới dạng JSON


def search_in_elasticsearch(query, index_name, top_k=5):
    # Chuyển đổi câu truy vấn thành vector
    query_vector = text_to_vector(query)
    
    # Xây dựng truy vấn tìm kiếm theo cosine similarity
    search_query = {
        "size": top_k,  # Giới hạn số lượng kết quả trả về
        "query": {
            "script_score": {
                "query": {"match_all": {}},  # Áp dụng cho tất cả các tài liệu
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",  # Tính điểm tương đồng
                    "params": {"query_vector": query_vector}  # Truyền vector của truy vấn vào
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)  # Gửi truy vấn đến Elasticsearch
    
    # Trích xuất kết quả tìm kiếm
    results = [
        {
            "title": hit["_source"]["title"],
            "answer": hit["_source"]["answer"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    
    return results  # Trả về danh sách kết quả tìm kiếm


def refine_answer(query, answer):
    # Sử dụng mô hình ngôn ngữ để làm câu trả lời trở nên tự nhiên hơn
    response = modelGMN.generate_content(
        f"Câu hỏi của người dùng: {query} - Đáp án mẫu: {answer}\nHãy chuyển đáp án mẫu lại sao cho tự nhiên hơn"
    )
    
    return response.text  # Trả về câu trả lời đã được chỉnh sửa

@app.route('/search', methods=['POST'])
def search():
    # Nhận dữ liệu từ request
    data = request.json
    query = data.get("query", "")
    
    # Chuẩn hóa truy vấn về dạng Unicode NFC và chuyển thành chữ thường
    query = unicodedata.normalize("NFC", query).lower()
    
    if not query:
        return jsonify({"error": "Query không được để trống."}), 400  # Kiểm tra nếu query trống
    
    if not es.ping():
        return jsonify({"error": "Không thể kết nối Elasticsearch!"}), 500  # Kiểm tra kết nối Elasticsearch
    
    # Tìm kiếm trong Elasticsearch
    results = search_in_elasticsearch(query, INDEX_NAME)
    
    if not results:
        return jsonify({"message": "Không tìm thấy kết quả phù hợp."})  # Trả về nếu không có kết quả
    
    # Tạo danh sách câu trả lời từ kết quả tìm kiếm
    answers = []
    for idx, result in enumerate(results):
        answers.append(f"{idx}️⃣ Đáp án {idx}: \"{result['answer']}\"")
    
    answers_text = "\n".join(answers)
    
    # Tạo prompt để mô hình ngôn ngữ phân tích và chọn câu trả lời phù hợp
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
    
    print("\n🔹 Kết quả từ Gemini AI:", gemini_result)  # Log kết quả từ mô hình ngôn ngữ
    
    if gemini_result == "False":
        return jsonify({
            "query": query,
            "original_answer": "Xin lỗi, tôi không hiểu câu hỏi của bạn. Hãy chi tiết câu hỏi hơn.",
            "refined_answer": "Xin lỗi, tôi không hiểu câu hỏi của bạn. Hãy chi tiết câu hỏi hơn."
        })
    
    # Lấy câu trả lời tốt nhất theo đánh giá của mô hình
    best_result = results[int(gemini_result)]
    
    # Tinh chỉnh lại câu trả lời để tự nhiên hơn
    improved_answer = refine_answer(query, best_result['answer'])
    print(improved_answer)  # Log câu trả lời đã chỉnh sửa
    
    return jsonify({
        "query": query,
        "original_answer": best_result['answer'],
        "refined_answer": improved_answer.replace("\n", "<br>")  # Định dạng HTML cho xuống dòng
    })


if __name__ == "__main__":
    # Chạy ứng dụng Flask trên tất cả các địa chỉ IP của máy chủ với cổng 5000
    # Bật chế độ debug để hỗ trợ phát triển
    app.run(host="0.0.0.0", port=5000, debug=True)


