"""
Template-based synthetic data generator for insurance letter classification training augmentation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)


class TemplateDataGenerator:
    """
    Генератор синтетических данных на основе шаблонов для аугментации обучающих данных
    классификации страховых писем
    """
    
    def __init__(self, seed: int = 42):
        """
        Инициализация генератора
        
        Args:
            seed: семя для воспроизводимости
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Шаблоны для генерации класса 0 (общие запросы)
        self.class_0_templates = [
            "{company}\n\nОрганизовать {services} для пациента {patient_info}.",
            "{company}\n\nНеобходимо провести {services} согласно программе ДМС.",
            "{company}\n\nПросим организовать {services} для застрахованного лица.",
            "{company}\n\nТребуется {services} в рамках полиса ДМС.",
            "{company}\n\nПросим {patient_action} {services} согласно условиям страхования.",
            "{company}\n\nОбеспечить {services} в рамках {program_type}.",
            "{company}\n\nОрганизовать {services} для пациента по полису ДМС."
        ]
        
        # Шаблоны для генерации класса 1 (специфические коды услуг)
        self.class_1_templates = [
            "{company}\n\nПросим организовать оплату следующих медицинских услуг:\n{service_codes}\n\nДиагноз: {diagnosis}",
            "{company}\n\nНеобходимо оплатить медицинские услуги по кодам:\n{service_codes}\n\nПациент: {patient_info}",
            "{company}\n\nОрганизовать оплату услуг:\n{service_codes}\n\nЗаключение: {diagnosis}",
            "{company}\n\nОплатить следующие процедуры:\n{service_codes}\n\nДиагностическое заключение: {diagnosis}",
            "{company}\n\nГарантируем оплату услуг:\n{service_codes}\n\nПо результатам экспертизы: {diagnosis}"
        ]
        
        # Словари для генерации контента
        self.companies = ["СОГАЗ", "Ингосстрах", "СОГАЗ-Мед", "ИНГОССТРАХ"]
        
        self.general_services = [
            "необходимые исследования",
            "консультацию специалиста",
            "приемы врача",
            "медицинские процедуры",
            "диагностические мероприятия",
            "лечебные процедуры",
            "консультации врачей",
            "обследование пациента",
            "лабораторные исследования",
            "инструментальную диагностику",
            "терапевтические мероприятия"
        ]
        
        self.diagnoses = [
            "Хроническое заболевание",
            "Острое состояние", 
            "Профилактическое обследование",
            "Контрольное наблюдение",
            "Диспансерное наблюдение",
            "Реабилитационные мероприятия",
            "Плановое лечение",
            "Экстренная помощь"
        ]
        
        self.patient_info = [
            "согласно полису ДМС",
            "по программе добровольного медицинского страхования",
            "в рамках страхового полиса",
            "согласно условиям страхования",
            "по договору медицинского страхования"
        ]
        
        self.patient_actions = [
            "организовать",
            "обеспечить",
            "предоставить",
            "провести"
        ]
        
        self.program_types = [
            "программы ДМС",
            "стандартной программы",
            "расширенной программы",
            "базовой программы страхования"
        ]
    
    def generate_service_codes(self, num_codes: int = None) -> List[str]:
        """
        Генерация кодов медицинских услуг в формате F##.##.##.#.###
        
        Args:
            num_codes: количество кодов для генерации
            
        Returns:
            список кодов услуг
        """
        if num_codes is None:
            num_codes = random.randint(1, 5)
        
        codes = []
        for _ in range(num_codes):
            code = f"F{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(0, 9)}.{random.randint(100, 999)}"
            codes.append(code)
        
        return codes
    
    def generate_class_0_letter(self) -> str:
        """
        Генерация письма класса 0 (общие запросы без кодов услуг)
        
        Returns:
            текст письма
        """
        template = random.choice(self.class_0_templates)
        
        return template.format(
            company=random.choice(self.companies),
            services=random.choice(self.general_services),
            patient_info=random.choice(self.patient_info),
            patient_action=random.choice(self.patient_actions),
            program_type=random.choice(self.program_types)
        )
    
    def generate_class_1_letter(self) -> str:
        """
        Генерация письма класса 1 (с кодами услуг)
        
        Returns:
            текст письма
        """
        template = random.choice(self.class_1_templates)
        service_codes = self.generate_service_codes()
        
        return template.format(
            company=random.choice(self.companies),
            service_codes="\n".join(service_codes),
            diagnosis=random.choice(self.diagnoses),
            patient_info=random.choice(self.patient_info)
        )
    
    def generate_balanced_dataset(self, 
                                train_df: pd.DataFrame, 
                                target_size: int = 200) -> pd.DataFrame:
        """
        Генерация сбалансированного синтетического датасета
        
        Args:
            train_df: оригинальные обучающие данные
            target_size: целевой размер итогового датасета
            
        Returns:
            DataFrame с синтетическими данными
        """
        logger.info(f"Generating synthetic dataset with target size: {target_size}")
        
        # Определяем количество синтетических данных для генерации
        original_size = len(train_df)
        synthetic_size = max(0, target_size - original_size)
        
        if synthetic_size == 0:
            logger.warning("Target size is not larger than original dataset")
            return pd.DataFrame()
        
        # Анализ распределения классов в оригинальных данных
        class_counts = train_df['class'].value_counts()
        class_0_ratio = class_counts.get(0, 0) / len(train_df)
        class_1_ratio = class_counts.get(1, 0) / len(train_df)
        
        logger.info(f"Original class distribution: Class 0: {class_0_ratio:.2f}, Class 1: {class_1_ratio:.2f}")
        
        # Генерация синтетических данных с сохранением пропорций
        synthetic_class_0 = int(synthetic_size * class_0_ratio)
        synthetic_class_1 = synthetic_size - synthetic_class_0
        
        synthetic_data = []
        
        # Генерация писем класса 0
        for i in range(synthetic_class_0):
            letter_text = self.generate_class_0_letter()
            synthetic_data.append({
                'letter_id': f'synthetic_0_{i}',
                'guarantee_letter_text': letter_text,
                'class': 0
            })
        
        # Генерация писем класса 1
        for i in range(synthetic_class_1):
            letter_text = self.generate_class_1_letter()
            synthetic_data.append({
                'letter_id': f'synthetic_1_{i}',
                'guarantee_letter_text': letter_text,
                'class': 1
            })
        
        # Создание DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Перемешивание данных
        synthetic_df = synthetic_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        logger.info(f"Generated {len(synthetic_df)} synthetic samples:")
        logger.info(f"  Class 0: {synthetic_class_0}")
        logger.info(f"  Class 1: {synthetic_class_1}")
        
        return synthetic_df
        
    def generate_for_balancing(self, train_df: pd.DataFrame, balance_ratio: float = 0.5) -> pd.DataFrame:
        """
        Генерация синтетических данных для балансировки классов
        
        Args:
            train_df: оригинальные данные
            balance_ratio: желаемое соотношение класса меньшинства
            
        Returns:
            DataFrame с синтетическими данными для балансировки
        """
        class_counts = train_df['class'].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        # Вычисляем сколько нужно добавить данных класса меньшинства
        target_minority_count = int(majority_count * balance_ratio / (1 - balance_ratio))
        additional_minority_samples = max(0, target_minority_count - minority_count)
        
        if additional_minority_samples == 0:
            return pd.DataFrame()
        
        logger.info(f"Generating {additional_minority_samples} samples for class {minority_class} to balance dataset")
        
        synthetic_data = []
        for i in range(additional_minority_samples):
            if minority_class == 0:
                letter_text = self.generate_class_0_letter()
            else:
                letter_text = self.generate_class_1_letter()
            
            synthetic_data.append({
                'letter_id': f'balance_{minority_class}_{i}',
                'guarantee_letter_text': letter_text,
                'class': minority_class
            })
        
        return pd.DataFrame(synthetic_data)