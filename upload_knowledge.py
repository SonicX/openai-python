#!/usr/bin/env python3
"""
Скрипт для загрузки файлов в качестве знаний на сервер
Использование: python upload_knowledge.py <путь_к_файлу> <api_key> [опциональные_параметры]
"""

import os
import sys
import json
import mimetypes
import requests
from pathlib import Path
from typing import Optional, Dict, Any


def validate_file_path(file_path: str) -> bool:
    """Проверяет существование файла и его доступность для чтения"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Ошибка: Файл {file_path} не существует")
        return False
    
    if not path.is_file():
        print(f"Ошибка: {file_path} не является файлом")
        return False
    
    if not os.access(path, os.R_OK):
        print(f"Ошибка: Нет прав на чтение файла {file_path}")
        return False
    
    return True


def get_file_type(file_path: str) -> str:
    """Определяет тип файла по расширению и MIME-типу"""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Поддерживаемые типы файлов
    supported_types = {
        '.txt': 'text/plain',
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.html': 'text/html',
        '.htm': 'text/html'
    }
    
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if extension in supported_types:
        return supported_types[extension]
    elif mime_type:
        return mime_type
    else:
        return 'application/octet-stream'  # Бинарный тип по умолчанию


def read_file_content(file_path: str) -> Optional[str]:
    """Читает содержимое файла и возвращает его как строку"""
    try:
        # Определяем тип файла
        file_type = get_file_type(file_path)
        
        # Для текстовых файлов читаем напрямую
        if file_type.startswith('text/') or file_type in [
            'application/json', 'application/xml', 'text/html'
        ]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Для бинарных файлов (PDF, DOCX и т.д.) возвращаем содержимое как байты
        # которые затем можно закодировать или обработать соответствующей библиотекой
        else:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')
                
    except UnicodeDecodeError:
        # Если возникла ошибка декодирования, пробуем другие кодировки
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                # Пробуем декодировать как CP1251 (часто используется для русских документов)
                return raw_data.decode('cp1251')
        except UnicodeDecodeError:
            print(f"Ошибка: Не удалось декодировать файл {file_path}")
            return None
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None


def prepare_file_data(file_path: str) -> Dict[str, Any]:
    """Подготавливает данные файла для отправки"""
    file_path_obj = Path(file_path)
    
    return {
        'filename': file_path_obj.name,
        'filepath': str(file_path_obj.absolute()),
        'size': file_path_obj.stat().st_size,
        'content': read_file_content(file_path),
        'file_type': get_file_type(file_path)
    }


def upload_to_server(file_data: Dict[str, Any], api_key: str, server_url: str) -> bool:
    """
    Отправляет файл на сервер
    ВАЖНО: Здесь нужно заменить на реальный API вашего сервера
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Подготовка данных для отправки
    payload = {
        'filename': file_data['filename'],
        'file_type': file_data['file_type'],
        'content': file_data['content'][:10000],  # Ограничиваем длину для примера
        'size': file_data['size']
    }
    
    try:
        print("Отправка файла на сервер...")
        response = requests.post(
            server_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            print("✓ Файл успешно загружен на сервер")
            print(f"Ответ сервера: {response.json()}")
            return True
        else:
            print(f"✗ Ошибка при загрузке файла: {response.status_code}")
            print(f"Ответ сервера: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Ошибка подключения к серверу")
        return False
    except requests.exceptions.Timeout:
        print("✗ Время ожидания истекло")
        return False
    except Exception as e:
        print(f"✗ Произошла ошибка при отправке файла: {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Использование: python upload_knowledge.py <путь_к_файлу> <api_key> [server_url]")
        print("Пример: python upload_knowledge.py ./document.pdf abc123def456 https://my-server.com/api/upload")
        sys.exit(1)
    
    file_path = sys.argv[1]
    api_key = sys.argv[2]
    server_url = sys.argv[3] if len(sys.argv) > 3 else "https://example.com/api/upload"  # Значение по умолчанию
    
    print(f"Файл: {file_path}")
    print(f"API ключ: {'*' * len(api_key)} ({len(api_key)} символов)")
    print(f"URL сервера: {server_url}")
    
    # Проверяем файл
    if not validate_file_path(file_path):
        sys.exit(1)
    
    # Получаем информацию о файле
    print("\nЧтение файла...")
    file_data = prepare_file_data(file_path)
    
    if file_data['content'] is None:
        print("Не удалось прочитать содержимое файла")
        sys.exit(1)
    
    print(f"✓ Файл прочитан. Размер: {file_data['size']} байт")
    print(f"Тип файла: {file_data['file_type']}")
    
    # Отправляем файл на сервер
    success = upload_to_server(file_data, api_key, server_url)
    
    if success:
        print("\n✓ Загрузка завершена успешно")
    else:
        print("\n✗ Загрузка не удалась")
        sys.exit(1)


if __name__ == "__main__":
    main()