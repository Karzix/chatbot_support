import google.generativeai as genai

# Thiết lập API Key
genai.configure(api_key="AIzaSyC7aoODhVimXdVvsKgKlS6Oe3qZwMEV41k")

# Chọn mô hình Gemini Pro (miễn phí)
model = genai.GenerativeModel("gemini-2.0-flash")

# Gửi câu hỏi
response = model.generate_content("Giới thiệu về trí tuệ nhân tạo.")

# In kết quả
print(response.text)
