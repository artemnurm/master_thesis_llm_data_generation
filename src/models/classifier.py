"""
Binary classifier for insurance letter classification using sentence transformers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sentence_transformers import SentenceTransformer
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import re

logger = logging.getLogger(__name__)


class BinaryClassifier:
    """
    Бинарный классификатор для страховых писем
    Использует sentence transformers для получения эмбеддингов + рукотворные признаки
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 classifier_type: str = "logistic"):
        """
        Инициализация классификатора
        
        Args:
            model_name: название модели sentence transformers
            classifier_type: тип классификатора ("logistic" или "random_forest")
        """
        self.model_name = model_name
        self.classifier_type = classifier_type
        
        # Загрузка модели эмбеддингов
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Инициализация классификатора
        if classifier_type == "logistic":
            self.classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Учет дисбаланса классов
            )
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_features(self, texts: List[str]) -> np.ndarray:
        """
        Подготовка признаков из текстов
        
        Args:
            texts: список текстов
            
        Returns:
            матрица признаков
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Очистка текстов
        cleaned_texts = []
        for text in texts:
            if pd.isna(text):
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(text).strip())
        
        # Получение эмбеддингов
        embeddings = self.embedding_model.encode(
            cleaned_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def add_handcrafted_features(self, texts: List[str]) -> np.ndarray:
        """
        Добавление рукотворных признаков к эмбеддингам
        
        Args:
            texts: список текстов
            
        Returns:
            расширенная матрица признаков
        """
        additional_features = []
        service_code_pattern = re.compile(r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}')
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            # Признаки наличия кодов услуг
            service_codes = service_code_pattern.findall(text)
            has_service_codes = len(service_codes) > 0
            num_service_codes = len(service_codes)
            
            # Признаки длины текста
            text_length = len(text)
            word_count = len(text.split())
            
            # Признаки страховых компаний
            has_sogaz = 'согаз' in text.lower()
            has_ingosstrakh = 'ингосстрах' in text.lower()
            
            # Ключевые слова для класса 0
            class_0_keywords = ['необходимые исследования', 'приемы врача', 'консультация', 'дмс']
            class_0_score = sum(1 for keyword in class_0_keywords if keyword in text.lower())
            
            # Ключевые слова для класса 1
            class_1_keywords = ['оплату следующих', 'медицинских услуг', 'диагноз']
            class_1_score = sum(1 for keyword in class_1_keywords if keyword in text.lower())
            
            additional_features.append([
                int(has_service_codes),
                num_service_codes,
                text_length,
                word_count,
                int(has_sogaz),
                int(has_ingosstrakh),
                class_0_score,
                class_1_score
            ])
        
        return np.array(additional_features)
    
    def train(self, train_texts: List[str], train_labels: List[int]) -> Dict[str, float]:
        """
        Обучение классификатора
        
        Args:
            train_texts: обучающие тексты
            train_labels: метки классов
            
        Returns:
            метрики валидации
        """
        logger.info(f"Training classifier on {len(train_texts)} samples")
        
        # Подготовка признаков
        embeddings = self.prepare_features(train_texts)
        handcrafted_features = self.add_handcrafted_features(train_texts)
        
        # Обучение скейлера
        self.scaler.fit(handcrafted_features)
        handcrafted_features_scaled = self.scaler.transform(handcrafted_features)
        
        features = np.hstack([embeddings, handcrafted_features_scaled])
        
        # Обучение
        self.classifier.fit(features, train_labels)
        self.is_fitted = True
        
        # Кросс-валидация
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            cv_scores = cross_val_score(self.classifier, features, train_labels, 
                                       cv=skf, scoring='f1_weighted')
            cv_f1_mean = cv_scores.mean()
            cv_f1_std = cv_scores.std()
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Using training F1 score instead.")
            cv_f1_mean = 0
            cv_f1_std = 0
        
        # Метрики на обучающих данных
        train_preds = self.classifier.predict(features)
        train_metrics = {
            'train_accuracy': accuracy_score(train_labels, train_preds),
            'train_precision': precision_score(train_labels, train_preds, average='weighted', zero_division=0),
            'train_recall': recall_score(train_labels, train_preds, average='weighted', zero_division=0),
            'train_f1': f1_score(train_labels, train_preds, average='weighted', zero_division=0),
            'cv_f1_mean': cv_f1_mean,
            'cv_f1_std': cv_f1_std
        }
        
        logger.info(f"Training completed. CV F1: {cv_f1_mean:.3f} ± {cv_f1_std:.3f}")
        return train_metrics
    
    def predict(self, test_texts: List[str]) -> Tuple[List[int], List[float]]:
        """
        Предсказание классов
        
        Args:
            test_texts: тексты для предсказания
            
        Returns:
            предсказанные классы и вероятности
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions for {len(test_texts)} samples")
        
        # Подготовка признаков
        embeddings = self.prepare_features(test_texts)
        handcrafted_features = self.add_handcrafted_features(test_texts)
        handcrafted_features_scaled = self.scaler.transform(handcrafted_features)
        
        features = np.hstack([embeddings, handcrafted_features_scaled])
        
        # Предсказание
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        
        return predictions.tolist(), probabilities.tolist()
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict[str, float]:
        """
        Оценка качества классификатора
        
        Args:
            test_texts: тестовые тексты
            test_labels: истинные метки
            
        Returns:
            метрики качества
        """
        predictions, probabilities = self.predict(test_texts)
        
        # Вероятность для класса 1
        prob_class_1 = [p[1] for p in probabilities]
        
        report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_class_0': report.get('0', {}).get('precision', 0),
            'recall_class_0': report.get('0', {}).get('recall', 0),
            'f1_class_0': report.get('0', {}).get('f1-score', 0),
            'precision_class_1': report.get('1', {}).get('precision', 0),
            'recall_class_1': report.get('1', {}).get('recall', 0),
            'f1_class_1': report.get('1', {}).get('f1-score', 0),
            'avg_confidence': np.mean(prob_class_1)
        }
        
        return metrics
    
    def save_model(self, filepath: Path):
        """Сохранение модели"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'classifier_type': self.classifier_type
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Загрузка модели"""
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.model_name = model_data.get('model_name', "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.classifier_type = model_data.get('classifier_type', 'logistic')
        
        # Re-initialize embedding model if needed
        self.embedding_model = SentenceTransformer(self.model_name)
        
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Получение важности признаков"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
            
        if self.classifier_type == "logistic":
            return self.classifier.coef_
        elif self.classifier_type == "random_forest":
            return self.classifier.feature_importances_
        else:
            return None