# Chatbot Hỗ Trợ Hướng Dẫn Sử Dụng Phần Mềm Quản Lý Bệnh Viện

## Giới thiệu
Dự án này triển khai một chatbot sử dụng Python, Flask và Elasticsearch để hỗ trợ nhân viên sử dụng phần mềm y tế. Chatbot có thể đọc nội dung từ file DOCX, trả lời câu hỏi dựa trên nội dung đó và cập nhật dữ liệu khi bổ sung file mới.

## Yêu cầu hệ thống
- **Docker** (Để chạy Elasticsearch)
- **Python** (Phiên bản mới nhất khuyến nghị)
- **pip** (Công cụ quản lý gói cho Python)

---

## Cài đặt Elasticsearch
### Chạy Elasticsearch với Docker
Mở terminal hoặc CMD và chạy lệnh sau để khởi chạy Elasticsearch:

```sh
docker run --name elasticsearch -d \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.7.0
```

#### Giải thích các tham số:
- `--name elasticsearch` : Đặt tên container là `elasticsearch`.
- `-d` : Chạy container ở chế độ nền.
- `-p 9200:9200` : Mở cổng 9200 để truy cập API của Elasticsearch.
- `-p 9300:9300` : Mở cổng 9300 để giao tiếp với các node khác (dùng cho cluster).
- `-e "discovery.type=single-node"` : Chạy Elasticsearch ở chế độ đơn node.
- `-e "xpack.security.enabled=false"` : Tắt bảo mật mặc định để dễ truy cập.
- `docker.elastic.co/elasticsearch/elasticsearch:8.7.0` : Sử dụng phiên bản Elasticsearch 8.7.0.

**Lưu ý:** Nếu bạn đã có container Elasticsearch, chỉ cần khởi động lại bằng lệnh:
```sh
docker start elasticsearch
```

---

## Chạy API
### Cài đặt các thư viện cần thiết
Đảm bảo Python đã được cài đặt trên máy. Chạy lệnh sau để cài đặt các thư viện cần thiết:

```sh
pip install flask flask-cors python-docx torch transformers elasticsearch numpy google-generativeai
```

#### Danh sách các thư viện:
- `flask`, `flask-cors`: Xây dựng API và hỗ trợ CORS.
- `python-docx`: Đọc file DOCX.
- `torch`: Chạy mô hình ngôn ngữ.
- `transformers`: Làm việc với mô hình NLP.
- `elasticsearch`: Kết nối với Elasticsearch.
- `numpy`: Xử lý dữ liệu số.
- `google-generativeai`: Sử dụng mô hình AI của Google.

> **Lưu ý:** Nếu gặp lỗi khi cài đặt, có thể cần sử dụng:
> ```sh
> python -m pip install <tên thư viện>
> ```

### Chạy API
1. Mở terminal hoặc CMD và di chuyển đến thư mục chứa `app.py`:
   ```sh
   cd chatbot_support/PythonApplication1/
   ```
2. Chạy ứng dụng bằng lệnh:
   ```sh
   py app.py
   ```

---

## Chạy giao diện người dùng
Mở file giao diện `index.html` trong thư mục:
```sh
chatbot_support/PythonApplication1/index.html
```
Dùng trình duyệt để kiểm tra giao diện chatbot.

---

## Cấu hình API Key
Trong file mã nguồn, bạn có thể thay đổi API key cho Gemini AI bằng cách chỉnh sửa dòng sau trong mã:
```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```
> **Lưu ý:** Hãy thay thế `YOUR_API_KEY_HERE` bằng API key của bạn.

---

## Build Extension Chrome

### Chạy API
Trước khi build extension, cần chạy API. Dockerfile nằm tại:
```sh
chatbot_support/PythonApplication1/Dockerfile
```

### Đổi API URL
Dự án có 2 folder:
- `extension_chatbot`
- `extension_upload`

Truy cập vào `index.js` và chỉnh sửa dòng sau:
```js
const api = "http://127.0.0.1:5000/"; 
// Đổi thành API URL của bạn:
const api = "{your api url}";
```

### Tạo Extension
1. Mở trình duyệt Chrome và truy cập:
   ```sh
   chrome://extensions/
   ```
2. Bật chế độ "Nhà phát triển" (góc trên bên phải màn hình).
3. Chọn **"Tải tiện ích đã giải nén"**.
4. Chọn một trong các thư mục extension, ví dụ: `extension_chatbot`.
5. Sau khi tải lên thành công, bạn sẽ thấy extension mới trong danh sách và có thể sử dụng.

---


