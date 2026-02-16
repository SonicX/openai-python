import os
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import faiss
    import numpy as np
    import PyPDF2
    from docx import Document
except ImportError as e:
    logger.error(f"Отсутствуют необходимые зависимости: {e}")
    logger.info("Установите зависимости: pip install sentence-transformers transformers faiss-cpu PyPDF2 python-docx")
    raise


@dataclass
class DocumentChunk:
    """Класс для представления фрагмента документа"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class DocumentProcessor:
    """Класс для обработки различных типов документов"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Извлечение текста из PDF файла"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Извлечение текста из DOCX файла"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Ошибка при чтении DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Извлечение текста из TXT файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Ошибка при чтении TXT {file_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> str:
        """Загрузка документа в зависимости от его типа"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Неподдерживаемый тип файла: {ext}. Попробуем прочитать как текстовый.")
            return self.extract_text_from_txt(file_path)


class TextSplitter:
    """Класс для разделения текста на фрагменты"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Разделение текста на фрагменты"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Определяем конец текущего фрагмента
            end = start + self.chunk_size
            
            # Если конец за пределами текста, корректируем
            if end > len(text):
                end = len(text)
            
            # Извлекаем фрагмент
            chunk_text = text[start:end]
            
            # Создаем ID для фрагмента на основе хэша содержимого
            chunk_id = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()
            
            # Создаем объект фрагмента
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
            
            # Переходим к следующему фрагменту с учетом перекрытия
            start = end - self.overlap
        
        return chunks


class EmbeddingManager:
    """Класс для работы с эмбеддингами"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Кодирование текстов в эмбеддинги"""
        return self.model.encode(texts)
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Кодирование одного текста в эмбеддинг"""
        return self.model.encode([text])[0]


class VectorStore:
    """Класс для хранения и поиска векторов"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусного сходства
        self.chunks = []  # Список фрагментов
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """Добавление эмбеддингов в индекс"""
        # Нормализация векторов для косинусного сходства
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentChunk]:
        """Поиск релевантных фрагментов по эмбеддингу запроса"""
        # Нормализация запроса
        query_embedding_norm = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding_norm)
        
        # Поиск ближайших соседей
        scores, indices = self.index.search(query_embedding_norm, k)
        
        # Формирование результата
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.chunks):  # Проверяем, что индекс валиден
                chunk = self.chunks[idx]
                # Добавляем оценку релевантности
                chunk.metadata['relevance_score'] = float(scores[0][i])
                results.append(chunk)
        
        return results


class RAGSystem:
    """Основная система RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 overlap: int = 50):
        
        self.document_processor = DocumentProcessor()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(dimension=self.embedding_manager.model.get_sentence_embedding_dimension())
        
        # Путь для сохранения/загрузки индекса
        self.index_file = "rag_index.pkl"
        self.chunks_file = "rag_chunks.pkl"
    
    def add_document(self, file_path: str, metadata: Dict = None):
        """Добавление документа в систему"""
        if metadata is None:
            metadata = {"source": file_path}
        else:
            metadata["source"] = file_path
        
        # Загружаем текст из файла
        text = self.document_processor.load_document(file_path)
        if not text.strip():
            logger.warning(f"Не удалось извлечь текст из файла {file_path}")
            return
        
        # Разбиваем текст на фрагменты
        chunks = self.text_splitter.split_text(text, metadata)
        
        # Создаем эмбеддинги для фрагментов
        texts_for_embedding = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.encode_texts(texts_for_embedding)
        
        # Добавляем в векторное хранилище
        self.vector_store.add_embeddings(embeddings, chunks)
        
        logger.info(f"Добавлено {len(chunks)} фрагментов из файла {file_path}")
    
    def add_documents_from_directory(self, directory_path: str, recursive: bool = True):
        """Добавление всех поддерживаемых документов из директории"""
        supported_extensions = ['.pdf', '.docx', '.txt']
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*') if recursive else directory.glob('*'):
            if file_path.suffix.lower() in supported_extensions:
                self.add_document(str(file_path))
    
    def search_relevant_chunks(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Поиск релевантных фрагментов для запроса"""
        query_embedding = self.embedding_manager.encode_single_text(query)
        return self.vector_store.search(query_embedding, k)
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Получение контекста для LLM на основе запроса"""
        relevant_chunks = self.search_relevant_chunks(query, k)
        
        # Объединяем релевантные фрагменты в один контекст
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Фрагмент из {chunk.metadata.get('source', 'неизвестный источник')}:\n{chunk.content}\n")
        
        return "\n".join(context_parts)
    
    def save_index(self):
        """Сохранение индекса и фрагментов на диск"""
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'index': self.vector_store.index,
                'dimension': self.vector_store.dimension
            }, f)
        
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.vector_store.chunks, f)
        
        logger.info("Индекс и фрагменты сохранены")
    
    def load_index(self):
        """Загрузка индекса и фрагментов с диска"""
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
                self.vector_store.index = index_data['index']
                self.vector_store.dimension = index_data['dimension']
            
            with open(self.chunks_file, 'rb') as f:
                self.vector_store.chunks = pickle.load(f)
            
            logger.info("Индекс и фрагменты загружены")
        else:
            logger.info("Файлы индекса не найдены, создается новая система")
    
    def generate_response(self, query: str, k: int = 5) -> str:
        """Генерация ответа на основе RAG (в реальной системе здесь будет вызов LLM)"""
        context = self.retrieve_context(query, k)
        
        # Используем API для генерации ответа на основе контекста
        import requests
        
        # Настройки API
        api_key = "sk-a07f8549cccc4e2985799e0c17893ba6"
        api_url = "https://ai.redelephant.ru/api"
        
        # Формируем промпт для API
        prompt = f"Контекст: {context}\n\nВопрос: {query}\n\nПожалуйста, дай исчерпывающий и точный ответ на основе предоставленного контекста."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("choices", [{}])[0].get("text", "").strip()
            else:
                generated_text = f"Ошибка при обращении к API: {response.status_code} - {response.text}"
        except Exception as e:
            generated_text = f"Произошла ошибка при обращении к API: {str(e)}"
        
        response = generated_text
        
        return response


def main():
    """Пример использования системы RAG"""
    # Создание экземпляра RAG системы
    rag_system = RAGSystem()
    
    # Загрузка индекса (если существует)
    rag_system.load_index()
    
    # Добавление тестового документа
    rag_system.add_document("test_document.json")
    
    # Пример поиска
    query = "Каково содержание этого документа?"
    response = rag_system.generate_response(query)
    print("Ответ на запрос:", response)
    
    # Сохранение индекса
    rag_system.save_index()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()