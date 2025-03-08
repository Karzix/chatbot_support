from docx import Document
import re
import torch
from transformers import AutoModel, AutoTokenizer
from elasticsearch import Elasticsearch
import numpy as np
import unicodedata


# Load model và tokenizer cho chuyển đổi văn bản thành vector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Hàm chuyển văn bản thành vector
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()[0]  # vector gốc (768 chiều)
    
    # Giảm về 256 chiều bằng cách lấy trung bình nhóm (hoặc PCA)
    # vector = np.array(vector).reshape(256, 3).mean(axis=1).tolist()
    
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


# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200/")

# Hàm lưu dữ liệu vào Elasticsearch
def save_to_elasticsearch(index_name, data, output_file=r"D:\title.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            title = item["title"]
            f.write(title + "\n")  # Ghi tiêu đề vào file
            
            title_vector = text_to_vector(title)
            answer_vector = text_to_vector(item["answer"])
            doc = {
                "title": title,
                "title_vector": title_vector,
                "answer": item["answer"],
                "answer_vector": answer_vector,
            }
            es.index(index=index_name, body=doc)
    
    print("OK, titles saved to", output_file)

# Chạy chương trình
file_path = r"C:\Users\ANH KIET\Downloads\VNPT_HIS_ORACLE_UM_1.3.docx"
index_name = "hdsd"
if es.ping():
    print("🔹 Elasticsearch đã kết nối thành công!")
    data = extract_docx_content(file_path)
    save_to_elasticsearch(index_name, data)
else:
    print("⚠️ Không thể kết nối Elasticsearch!")

