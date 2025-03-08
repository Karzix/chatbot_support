

from elasticsearch import Elasticsearch

# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")
mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "answer": {"type": "text"},
            "title_vector": {"type": "dense_vector", "dims": 768},
            "answer_vector": {"type": "dense_vector", "dims": 768}
        }
    }
}

# Xóa index cũ (nếu có)
if es.indices.exists(index="hdsd"):
    es.indices.delete(index="hdsd")

# Tạo index mới
es.indices.create(index="hdsd", body=mapping)
print("✅ Đã tạo lại index `hdsd` với `dense_vector`!")
