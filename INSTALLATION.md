# Установка и запуск экспериментов

## Требования к системе

- Python 3.8+
- Минимум 8 ГБ оперативной памяти
- Около 2 ГБ свободного места на диске

## Установка зависимостей

1. **Создание виртуального окружения (рекомендуется):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # или
   venv\Scripts\activate     # Windows
   ```

2. **Установка зависимостей:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Дополнительные модули для NLP:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Подготовка данных

1. Убедитесь, что файлы данных находятся в папке `data/raw/`:
   - `DS_хакатон_набор данных_train_231208_1030.xlsx`
   - `DS_хакатон_набор данных_test_231208_1030.xlsx`
   - `DS_хакатон_справочник_услуг_231208_1030.xlsx`

2. Файлы должны быть скопированы из исходной папки с данными.

## Запуск экспериментов

### Анализ данных (быстрый запуск)
```bash
python main.py --mode analysis
```

### Только базовый эксперимент
```bash
python main.py --mode baseline
```

### Эксперименты с синтетическими данными
```bash
python main.py --mode synthetic
```

### Полный набор экспериментов (воспроизводит диссертацию)
```bash
python main.py --mode full
```

## Ожидаемые результаты

Согласно диссертации, ожидаемые результаты:

- **Базовая модель**: F1-score ~0.75-0.80
- **С синтетическими данными**: F1-score ~0.83-0.88 (улучшение на 8-13%)
- **Точность для промышленного применения**: 88-89%

## Структура результатов

После запуска результаты сохраняются в `results/experiment_YYYYMMDD_HHMMSS/`:

```
results/experiment_20250129_143000/
├── experiment_results.json      # Полные результаты в JSON
├── data_analysis.json          # Анализ исходных данных
├── summary_report.md           # Сводный отчет
├── results_table.csv           # Таблица результатов
├── plots/                      # Графики и визуализации
│   ├── metrics_comparison.png
│   ├── synthetic_ablation.png
│   └── baseline_confusion_matrix.png
└── models/                     # Сохраненные модели
    ├── baseline_model.joblib
    └── synthetic_*_model.joblib
```

## Анализ результатов

1. **Jupyter Notebook**: `notebooks/analysis.ipynb`
2. **Сводный отчет**: Генерируется автоматически в `summary_report.md`
3. **Интерактивные графики**: В папке `plots/`

## Устранение проблем

### Проблема с памятью
Если возникают проблемы с памятью при загрузке sentence transformers:
- Уменьшите batch_size в конфигурации
- Используйте модель меньшего размера: `paraphrase-multilingual-MiniLM-L12-v2`

### Проблема с зависимостями
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Проблема с правами доступа к файлам
Убедитесь, что у пользователя есть права на чтение файлов Excel в папке `data/raw/`.

## Конфигурация

Настройки экспериментов можно изменить в файле `config.yaml`:

```yaml
# Размеры синтетических выборок для тестирования
synthetic_sizes: [50, 100, 200, 300]

# Тип классификатора
classifier_type: "logistic"  # или "random_forest"

# Модель для эмбеддингов
embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Временные затраты

- **Анализ данных**: ~1 минута
- **Базовый эксперимент**: ~5-10 минут
- **Синтетические эксперименты**: ~15-20 минут
- **Полный набор**: ~30-40 минут

*Время может варьироваться в зависимости от производительности системы.*