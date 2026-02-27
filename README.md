# 🧠 Alzheimer's Disease RAG Research Assistant

RAG-система (Retrieval-Augmented Generation) на базе искусственного интеллекта для анализа научной литературы и поиска потенциальных терапевтических мишеней при болезни Альцгеймера.

## 🚀 Основные возможности

- **Сбор данных**: Автоматизированный парсинг статей из PubMed с использованием `pubmed-parser`.
- **Умный поиск (RAG)**: Двухэтапная система поиска (Retrieval + Reranking) для максимальной точности.
- **Интерактивный UI**: Streamlit-интерфейс для удобной работы с результатами.
- **Оценка качества**: Встроенные инструменты для измерения MRR, Hit Rate и NDCG.

## 🛠 Технологический стек (RAG Pipeline)

### Векторизация и Embeddings
Для работы с медицинскими текстами используются специализированные модели **MedCPT** от NCBI:
- **Query Encoder**: `ncbi/MedCPT-Query-Encoder` — для преобразования вопросов пользователя в векторное представление.
- **Article Encoder**: `ncbi/MedCPT-Article-Encoder` — для индексации текстов статей.
- **Алгоритм**: Сопоставление плотных векторных представлений (768-dim).

### Поиск и Retrieval
- **Vector Store**: **ChromaDB** (используется `PersistentClient` для локального хранения).
- **Retrieval Logic**:
  1. **Первичный поиск**: HNSW (Hierarchical Navigable Small World) с использованием **косинусной близости** (cosine similarity). Извлекается топ-50 кандидатов.
  2. **Reranking**: Переранжирование кандидатов с помощью Cross-Encoder модели `ncbi/MedCPT-Cross-Encoder` для финального отбора топ-5 релевантных фрагментов.
- **Chunking**: "Умное" разбиение текста с перекрытием (overlap) и привязкой к точкам, чтобы сохранить целостность предложений.

### Генерация (Generation)
- **Модель**: `meta-llama/Meta-Llama-3-8B-Instruct` (HuggingFace Inference API).

## ⚙️ Установка и запуск

1. **Клонирование репозитория**:
   ```bash
   git clone https://github.com/VasilijEpishkin/alzheimer-rag-agent.git
   cd alzheimer-rag-agent
   ```

2. **Настройка окружения**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # или venv\Scripts\activate для Windows
   pip install -r requirements.txt
   ```

3. **Конфигурация**:
   Создайте файл `.env` на основе [example](.env.example) и добавьте ваш токен HuggingFace:
   ```text
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Запуск**:
   - **Интерфейс (Streamlit)**: `streamlit run src/app.py`
   - **Индексация данных**: `python src/rag/ingest.py`
   - **Оценка**: `python evaluation/run_evaluation.py`

## 📊 Оценка качества

Проект включает набор тестов для оценки эффективности поиска:
- **Hit Rate (Recall@K)**: вероятность того, что релевантный документ попал в топ-K.
- **MRR (Mean Reciprocal Rank)**: учитывает позицию первого релевантного документа.

Результаты оценки сохраняются в `evaluation/results/`.

---

