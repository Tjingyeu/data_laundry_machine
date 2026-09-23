# pdf_to_chunks.py
# PDF加载与分块

import fitz  # PyMuPDF
from tqdm import tqdm
import os

PDF_PATH = "Bawden D. Introduction to Information Science 2ed 2022.pdf"
CHUNK_SIZE = 1500  # 每个chunk的字符数
OVERLAP = 150     # 重叠字符数


def load_pdf(file_path):
    """加载PDF并提取所有文本"""
    doc = fitz.open(file_path)
    text = ""

    for page in tqdm(doc, desc="读取PDF页面"):
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    """将文本分割为重叠的chunks"""
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def save_chunks(chunks, output_path="chunks.txt"):
    """保存chunks到文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(f"--- Chunk {i} ---\n")
            f.write(c.replace("\n", " ") + "\n")
            f.write("===\n")


if __name__ == "__main__":
    print(f"📖 加载PDF: {PDF_PATH}")

    if not os.path.exists(PDF_PATH):
        print(f"❌ 文件不存在: {PDF_PATH}")
        exit(1)

    # Step 1: 加载PDF
    text = load_pdf(PDF_PATH)
    print(f"✅ PDF总字符数: {len(text):,}")

    # Step 2: 分块
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"✅ 生成chunks: {len(chunks)}")

    # Step 3: 保存
    save_chunks(chunks)
    print(f"✅ chunks已保存到 chunks.txt")
