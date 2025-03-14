from elasticsearch import Elasticsearch

# Kết nối Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Kiểm tra kết nối
if es.ping():
    print("🔹 Elasticsearch đã kết nối thành công!")
else:
    print("⚠️ Không thể kết nối Elasticsearch!")