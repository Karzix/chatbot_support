# Sử dụng Python 3.10 làm base image
FROM python:3.10

# Đặt thư mục làm thư mục chính
WORKDIR /app

# Copy toàn bộ file từ thư mục hiện tại vào thư mục trong container
COPY . /app

# Cài đặt các thư viện khác trước
RUN pip install --no-cache-dir flask flask-cors elasticsearch \
    transformers python-docx numpy google-generativeai

# Cài đặt PyTorch riêng với chỉ mục CPU
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Xóa cache không cần thiết
RUN rm -rf /root/.cache

# Mở cổng 5000 để chạy API Flask
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
