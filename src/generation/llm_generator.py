"""
LLM-based synthetic data generator for insurance letters
This is a mock implementation for the clean repository
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
import logging
import json
import re

logger = logging.getLogger(__name__)


class MockLLMGenerator:
    """
    Mock LLM генератор синтетических гарантийных писем
    Имитирует работу GPT-4 для демонстrationных целей
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        
        # Предопределенные высококачественные примеры для имитации LLM
        self.class_0_examples = [
            "СОГАЗ\n\nОрганизовать необходимые медицинские исследования для пациента в рамках программы добровольного медицинского страхования. Обеспечить проведение консультаций специалистов согласно полису ДМС.",
            "Ингосстрах\n\nПросим организовать приемы врача и диагностические процедуры для застрахованного лица. Все мероприятия должны проводиться в рамках стандартной программы страхования.",
            "СОГАЗ-Мед\n\nТребуется проведение лечебных процедур и консультаций врачей согласно условиям договора медицинского страхования. Организовать необходимые лабораторные исследования.",
        ]
        
        self.class_1_examples = [
            "СОГАЗ\n\nПросим организовать оплату следующих медицинских услуг:\nF12.34.56.7.890 - Консультация кардиолога\nF23.45.67.8.901 - ЭКГ в покое\nF34.56.78.9.012 - Эхокардиография\n\nДиагноз: Ишемическая болезнь сердца",
            "Ингосстрах\n\nОплатить медицинские услуги по кодам:\nF45.67.89.0.123 - Общий анализ крови\nF56.78.90.1.234 - Биохимический анализ\nF67.89.01.2.345 - Консультация терапевта\n\nПо результатам экспертизы: Хроническое заболевание",
            "СОГАЗ\n\nГарантируем оплату услуг:\nF78.90.12.3.456 - УЗИ органов брюшной полости\nF89.01.23.4.567 - ФГДС\nF90.12.34.5.678 - Консультация гастроэнтеролога\n\nДиагностическое заключение: Заболевания ЖКТ",
        ]
        
        # Компоненты для генерации разнообразного контента
        self.companies = ["СОГАЗ", "Ингосстрах", "СОГАЗ-Мед", "ИНГОССТРАХ"]
        self.diagnoses = [
            "Ишемическая болезнь сердца",
            "Гипертоническая болезнь",
            "Хроническое заболевание",
            "Острое состояние",
            "Профилактическое обследование",
            "Заболевания ЖКТ",
            "Неврологические нарушения",
            "Эндокринные расстройства"
        ]
        
    def generate_service_code(self) -> str:
        """Генерация одного кода услуги"""
        return f"F{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(0, 9)}.{random.randint(100, 999)}"
    
    def generate_service_with_name(self) -> str:
        """Генерация кода услуги с названием"""
        services = [
            "Консультация кардиолога",
            "Консультация терапевта", 
            "Консультация невролога",
            "ЭКГ в покое",
            "Эхокардиография",
            "УЗИ органов брюшной полости",
            "Общий анализ крови",
            "Биохимический анализ",
            "ФГДС",
            "Рентгенография",
            "МРТ головного мозга",
            "КТ органов грудной клетки"
        ]
        
        code = self.generate_service_code()
        service_name = random.choice(services)
        return f"{code} - {service_name}"
    
    def generate_class_0_letter(self, company: str = None) -> Dict[str, Any]:
        """
        Генерация письма класса 0 с имитацией LLM качества
        
        Returns:
            словарь с текстом и метаданными
        """
        if company is None:
            company = random.choice(self.companies)
        
        # Выбираем базовый пример и модифицируем его
        base_example = random.choice(self.class_0_examples)
        
        # Заменяем компанию
        for comp in self.companies:
            if comp in base_example:
                letter_text = base_example.replace(comp, company)
                break
        else:
            letter_text = base_example
        
        # Добавляем небольшие вариации
        variations = [
            ("необходимые медицинские исследования", "требуемые диагностические процедуры"),
            ("в рамках программы", "согласно условиям программы"),
            ("застрахованного лица", "пациента"),
            ("приемы врача", "консультации специалистов"),
            ("лечебных процедур", "терапевтических мероприятий")
        ]
        
        for original, replacement in variations:
            if random.random() < 0.3:  # 30% шанс замены
                letter_text = letter_text.replace(original, replacement)
        
        return {
            'text': letter_text,
            'class': 0,
            'quality_score': random.uniform(0.85, 0.95),
            'confidence': random.uniform(0.9, 1.0),
            'generation_method': 'llm_mock'
        }
    
    def generate_class_1_letter(self, company: str = None, num_services: int = None) -> Dict[str, Any]:
        """
        Генерация письма класса 1 с имитацией LLM качества
        
        Returns:
            словарь с текстом и метаданными
        """
        if company is None:
            company = random.choice(self.companies)
        
        if num_services is None:
            num_services = random.randint(2, 5)
        
        # Генерируем услуги с кодами
        services = []
        for _ in range(num_services):
            services.append(self.generate_service_with_name())
        
        # Выбираем шаблон
        templates = [
            "{company}\n\nПросим организовать оплату следующих медицинских услуг:\n{services}\n\nДиагноз: {diagnosis}",
            "{company}\n\nОплатить медицинские услуги по кодам:\n{services}\n\nПо результатам экспертизы: {diagnosis}",
            "{company}\n\nГарантируем оплату услуг:\n{services}\n\nДиагностическое заключение: {diagnosis}",
            "{company}\n\nТребуется оплата медицинских процедур:\n{services}\n\nКлиническое заключение: {diagnosis}"
        ]
        
        template = random.choice(templates)
        diagnosis = random.choice(self.diagnoses)
        
        letter_text = template.format(
            company=company,
            services="\n".join(services),
            diagnosis=diagnosis
        )
        
        return {
            'text': letter_text,
            'class': 1,
            'quality_score': random.uniform(0.88, 0.97),
            'confidence': random.uniform(0.92, 1.0),
            'generation_method': 'llm_mock',
            'num_service_codes': num_services
        }
    
    def generate_batch(self, class_distribution: Dict[int, int]) -> List[Dict[str, Any]]:
        """
        Генерация батча синтетических писем
        
        Args:
            class_distribution: {class: count} - распределение классов
            
        Returns:
            список сгенерированных писем
        """
        results = []
        
        for class_label, count in class_distribution.items():
            for i in range(count):
                if class_label == 0:
                    letter_data = self.generate_class_0_letter()
                else:
                    letter_data = self.generate_class_1_letter()
                
                letter_data['letter_id'] = f'llm_mock_{class_label}_{i}'
                results.append(letter_data)
        
        return results
    
    def generate_synthetic_dataset(self, target_counts: Dict[int, int]) -> pd.DataFrame:
        """
        Генерация синтетического датасета
        
        Args:
            target_counts: количество примеров для каждого класса
            
        Returns:
            DataFrame с синтетическими данными
        """
        logger.info(f"Generating synthetic dataset with LLM mock: {target_counts}")
        
        all_letters = self.generate_batch(target_counts)
        
        # Конвертируем в DataFrame
        data_for_df = []
        for letter in all_letters:
            data_for_df.append({
                'letter_id': letter['letter_id'],
                'guarantee_letter_text': letter['text'],
                'class': letter['class'],
                'quality_score': letter['quality_score'],
                'confidence': letter['confidence'],
                'generation_method': letter['generation_method']
            })
        
        df = pd.DataFrame(data_for_df)
        
        # Перемешиваем
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} synthetic letters with high quality scores")
        logger.info(f"Average quality score: {df['quality_score'].mean():.3f}")
        
        return df
    
    def validate_quality(self, letters: List[str]) -> List[Dict[str, Any]]:
        """
        Имитация валидации качества сгенерированных писем
        
        Args:
            letters: список текстов писем
            
        Returns:
            список с метриками качества
        """
        results = []
        
        for i, letter in enumerate(letters):
            # Имитируем проверки качества
            has_service_codes = bool(re.search(r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}', letter))
            
            quality_metrics = {
                'letter_index': i,
                'has_service_codes': has_service_codes,
                'text_length': len(letter),
                'readability_score': random.uniform(0.7, 0.95),
                'format_compliance': random.uniform(0.85, 1.0),
                'domain_relevance': random.uniform(0.9, 1.0),
                'overall_quality': random.uniform(0.8, 0.95)
            }
            
            results.append(quality_metrics)
        
        return results