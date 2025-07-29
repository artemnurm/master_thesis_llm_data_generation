"""
Data loader for insurance letter classification
Handles loading and preprocessing of training, test, and reference data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import ast
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """Загрузчик и препроцессор данных страховых писем"""
    
    def __init__(self):
        self.service_code_pattern = re.compile(r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}')
        self.company_patterns = {
            'Согаз': re.compile(r'Согаз|СОГАЗ', re.IGNORECASE),
            'Ингосстрах': re.compile(r'Ингосстрах|ИНГОССТРАХ', re.IGNORECASE),
        }
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Загрузка данных из Excel файлов
        
        Args:
            data_path: путь к директории с данными
            
        Returns:
            train_df: DataFrame с обучающими данными (162 примера)
            test_df: DataFrame с тестовыми данными (156 примеров)
            services_df: DataFrame со справочником услуг (75 услуг)
        """
        data_dir = Path(data_path)
        
        try:
            # Загрузка обучающих данных
            train_file = data_dir / "DS_хакатон_набор данных_train_231208_1030.xlsx"
            train_df = pd.read_excel(train_file)
            logger.info(f"Loaded {len(train_df)} training samples from {train_file}")
            
            # Загрузка тестовых данных
            test_file = data_dir / "DS_хакатон_набор данных_test_231208_1030.xlsx"
            test_df = pd.read_excel(test_file)
            logger.info(f"Loaded {len(test_df)} test samples from {test_file}")
            
            # Загрузка справочника услуг
            services_file = data_dir / "DS_хакатон_справочник_услуг_231208_1030.xlsx"
            services_df = pd.read_excel(services_file)
            logger.info(f"Loaded {len(services_df)} services from {services_file}")
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return train_df, test_df, services_df
    
    def preprocess_letters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка текстов писем
        
        Args:
            df: DataFrame с письмами
            
        Returns:
            df: DataFrame с дополнительными признаками
        """
        df = df.copy()
        
        # Очистка текста
        df['text_cleaned'] = df['guarantee_letter_text'].apply(self._clean_text)
        
        # Извлечение признаков
        df['has_service_codes'] = df['text_cleaned'].apply(self._detect_service_codes)
        df['service_codes_list'] = df['text_cleaned'].apply(self._extract_service_codes)
        df['num_service_codes'] = df['service_codes_list'].apply(len)
        df['insurance_company'] = df['text_cleaned'].apply(self._extract_company)
        df['text_length'] = df['text_cleaned'].apply(len)
        df['num_words'] = df['text_cleaned'].apply(lambda x: len(x.split()))
        
        # Извлечение service_ids если есть
        if 'service_ids_list' in df.columns:
            df['parsed_service_ids'] = df['service_ids_list'].apply(self.extract_service_ids)
        
        logger.info(f"Preprocessed {len(df)} letters")
        logger.info(f"Found service codes in {df['has_service_codes'].sum()} letters")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и нормализация"""
        if pd.isna(text):
            return ""
        
        # Заменяем переносы строк на пробелы
        text = text.replace('\n', ' ')
        
        # Нормализуем пробелы
        text = ' '.join(text.split())
        
        # Убираем лишние пробелы вокруг знаков препинания
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        return text.strip()
    
    def _detect_service_codes(self, text: str) -> bool:
        """Определение наличия кодов услуг в тексте"""
        return bool(self.service_code_pattern.search(text))
    
    def _extract_service_codes(self, text: str) -> List[str]:
        """Извлечение всех кодов услуг из текста"""
        return self.service_code_pattern.findall(text)
    
    def _extract_company(self, text: str) -> Optional[str]:
        """Извлечение названия страховой компании"""
        for company, pattern in self.company_patterns.items():
            if pattern.search(text):
                return company
        return "Другая"
    
    def extract_service_ids(self, service_ids_str: str) -> List[int]:
        """
        Извлечение списка ID услуг из строкового представления
        
        Args:
            service_ids_str: строка с представлением списка ID
            
        Returns:
            список целочисленных ID услуг
        """
        if pd.isna(service_ids_str):
            return []
        
        try:
            # Пробуем распарсить как Python литерал
            parsed = ast.literal_eval(str(service_ids_str))
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
            elif isinstance(parsed, (int, float)):
                return [int(parsed)]
            else:
                return []
        except (ValueError, SyntaxError):
            # Если не получилось распарсить, пробуем извлечь числа
            numbers = re.findall(r'\d+', str(service_ids_str))
            return [int(x) for x in numbers]
    
    def create_service_mapping(self, services_df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
        """
        Создание маппинга ID услуги на код и название
        
        Args:
            services_df: DataFrame со справочником услуг
            
        Returns:
            словарь {service_id: {'code': код, 'name': название}}
        """
        service_mapping = {}
        
        for _, row in services_df.iterrows():
            service_mapping[row['service_id']] = {
                'code': row['ServiceCode'],
                'name': row['ServiceName']
            }
        
        logger.info(f"Created mapping for {len(service_mapping)} services")
        return service_mapping
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Анализ качества данных
        
        Returns:
            словарь с метриками качества данных
        """
        quality_metrics = {
            'total_samples': len(df),
            'missing_texts': df['guarantee_letter_text'].isna().sum(),
            'empty_texts': (df['guarantee_letter_text'] == '').sum(),
            'class_distribution': df['class'].value_counts().to_dict() if 'class' in df.columns else None,
            'avg_text_length': df['text_length'].mean() if 'text_length' in df.columns else None,
            'avg_words': df['num_words'].mean() if 'num_words' in df.columns else None,
            'companies_distribution': df['insurance_company'].value_counts().to_dict() if 'insurance_company' in df.columns else None,
        }
        
        # Анализ для классифицированных данных
        if 'class' in df.columns and 'has_service_codes' in df.columns:
            # Проверка соответствия классов и наличия кодов
            class_1_without_codes = ((df['class'] == 1) & (~df['has_service_codes'])).sum()
            class_0_with_codes = ((df['class'] == 0) & (df['has_service_codes'])).sum()
            
            quality_metrics['class_1_without_codes'] = class_1_without_codes
            quality_metrics['class_0_with_codes'] = class_0_with_codes
            quality_metrics['potential_labeling_errors'] = class_1_without_codes + class_0_with_codes
        
        return quality_metrics