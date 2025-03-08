from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return texts

if __name__ == "__main__":
    file_path = r"C:\Users\ANH KIET\Downloads\VNPT_HIS_ORACLE_UM_1.3.docx"
    content = read_docx(file_path)
    print("\n".join(content[:5]))  
