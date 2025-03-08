import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFC", text).lower()

title = normalize_text("Đăng nhập hệ thống")
query = normalize_text("đăng nhập hệ thống")

print([ord(c) for c in title])
print([ord(c) for c in query])