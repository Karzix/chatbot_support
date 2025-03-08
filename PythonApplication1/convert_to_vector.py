from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Tải mô hình tạo vector (hỗ trợ tiếng Việt)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def text_to_vector(text):
    return model.encode(text).tolist()  # Chuyển thành danh sách Python

# Test
if __name__ == "__main__":
    sample_text = "Hướng dẫn sử dụng phần mềm quản lý bệnh viện"
    vector = text_to_vector(sample_text)
    print(f"Vector đầu ra: {json.dumps(vector)}")  # In 5 giá trị đầu tiên
