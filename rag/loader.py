import os
import glob
import json
import hashlib
import docx
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_FOLDER = "documents"
CHROMA_PATH = "data/chroma"
INDEX_FILE = "data/docs_index.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("docs")


model = SentenceTransformer(MODEL_NAME)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def read_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {e}")
        return None


def read_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {e}")
        return None


def read_pdf(path):
    try:
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {e}")
        return None


def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index(index):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def build_vectorstore():
    old_index = load_index()
    new_index = {}

    current_files = set(glob.glob(os.path.join(DOCS_FOLDER, "*")))
    indexed_files = set(old_index.keys())
    deleted_files = indexed_files - current_files

    if deleted_files:
        ids_to_delete = []
        for fname in deleted_files:
            results = collection.get(where={"source": fname})
            ids_to_delete.extend(results['ids'])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"🗑️ Удалены документы из базы: {list(deleted_files)}")
            for fname in deleted_files:
                del old_index[fname]

    texts_to_add = []
    ids_to_add = []
    metadatas_to_add = []

    all_files = glob.glob(os.path.join(DOCS_FOLDER, "*"))

    for path in all_files:
        h = file_hash(path)
        new_index[path] = h

        if old_index.get(path) == h:
            continue

        print(f"🔄 Индексирую новый/измененный файл: {path}")

        content = ""
        if path.endswith(".txt"):
            content = read_txt(path)
        elif path.endswith(".docx"):
            content = read_docx(path)
        elif path.endswith(".pdf"):
            content = read_pdf(path)

        if not content:
            continue

        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{path}_{i}"
            texts_to_add.append(chunk)
            ids_to_add.append(chunk_id)
            metadatas_to_add.append({"source": path})

    if texts_to_add:
        print(f"📦 Добавляю {len(texts_to_add)} новых чанков в базу...")

        embeddings_to_add = model.encode(texts_to_add).tolist()

        collection.upsert(
            ids=ids_to_add,
            documents=texts_to_add,
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add
        )
        print("Добавление завершено.")
    else:
        print("База данных актуальна, обновления не требуются.")

    save_index(new_index)


def search_docs(query, top_k=5):
    """Ищет релевантные чанки и возвращает их текст."""
    if not query:
        return []

    q_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k
    )

    return results["documents"][0] if results.get("documents") else []


if __name__ == '__main__':
    build_vectorstore()
