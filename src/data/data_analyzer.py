"""
Insurance letter analysis module
Performs detailed analysis of guarantee letters for pattern discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class InsuranceLetterAnalyzer:
    """Глубокий анализ гарантийных писем страховых компаний"""
    
    def __init__(self):
        self.service_code_pattern = re.compile(r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}')
        self.diagnosis_pattern = re.compile(r'[A-Z]\d{2}\.?\d*')  # МКБ-10 коды
        
        # Ключевые фразы для каждого класса
        self.class_0_keywords = [
            'необходимые исследования',
            'необходимые лабораторные',
            'в рамках программы ДМС',
            'в рамках стандартной программы',
            'организовать приемы врача',
            'манипуляции в рамках',
            'согласно программе',
            'предусмотренной договором'
        ]
        
        self.class_1_keywords = [
            'оплата будет осуществляться',
            'диагноз',
            'код МКБ',
            'результатам экспертизы',
            'медицинской документации'
        ]
    
    def analyze_letter_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ паттернов в гарантийных письмах
        
        Args:
            df: DataFrame с предобработанными письмами
            
        Returns:
            словарь с результатами анализа
        """
        results = {
            'general_stats': self._analyze_general_statistics(df),
            'class_patterns': self._analyze_class_patterns(df),
            'company_patterns': self._analyze_company_patterns(df),
            'service_code_analysis': self._analyze_service_codes(df),
            'text_features': self._analyze_text_features(df),
            'keyword_analysis': self._analyze_keywords(df)
        }
        
        logger.info("Letter pattern analysis completed")
        return results
    
    def _analyze_general_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Общая статистика по датасету"""
        stats = {
            'total_samples': len(df),
            'class_distribution': df['class'].value_counts().to_dict() if 'class' in df.columns else None,
            'class_balance_ratio': df['class'].value_counts()[0] / df['class'].value_counts()[1] if 'class' in df.columns else None,
            'avg_text_length': {
                'overall': df['text_length'].mean(),
                'by_class': df.groupby('class')['text_length'].mean().to_dict() if 'class' in df.columns else None
            },
            'avg_words': {
                'overall': df['num_words'].mean(),
                'by_class': df.groupby('class')['num_words'].mean().to_dict() if 'class' in df.columns else None
            }
        }
        return stats
    
    def _analyze_class_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ паттернов по классам"""
        if 'class' not in df.columns:
            return {}
        
        patterns = {}
        
        for class_label in df['class'].unique():
            class_df = df[df['class'] == class_label]
            
            patterns[f'class_{class_label}'] = {
                'sample_count': len(class_df),
                'has_service_codes_ratio': class_df['has_service_codes'].mean() if 'has_service_codes' in class_df.columns else None,
                'avg_service_codes': class_df['num_service_codes'].mean() if 'num_service_codes' in class_df.columns else None,
                'company_distribution': class_df['insurance_company'].value_counts().to_dict() if 'insurance_company' in class_df.columns else None,
                'text_length_stats': {
                    'mean': class_df['text_length'].mean(),
                    'std': class_df['text_length'].std(),
                    'min': class_df['text_length'].min(),
                    'max': class_df['text_length'].max()
                }
            }
        
        # Анализ различий между классами
        if len(df['class'].unique()) == 2:
            class_0_df = df[df['class'] == 0]
            class_1_df = df[df['class'] == 1]
            
            patterns['class_differences'] = {
                'service_codes_presence': {
                    'class_0_with_codes': (class_0_df['has_service_codes'].sum() / len(class_0_df)) if 'has_service_codes' in class_0_df.columns else None,
                    'class_1_with_codes': (class_1_df['has_service_codes'].sum() / len(class_1_df)) if 'has_service_codes' in class_1_df.columns else None
                },
                'text_length_difference': class_1_df['text_length'].mean() - class_0_df['text_length'].mean()
            }
        
        return patterns
    
    def _analyze_company_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ паттернов по страховым компаниям"""
        if 'insurance_company' not in df.columns:
            return {}
        
        patterns = {}
        
        for company in df['insurance_company'].unique():
            company_df = df[df['insurance_company'] == company]
            
            patterns[company] = {
                'sample_count': len(company_df),
                'class_distribution': company_df['class'].value_counts().to_dict() if 'class' in company_df.columns else None,
                'avg_text_length': company_df['text_length'].mean(),
                'has_service_codes_ratio': company_df['has_service_codes'].mean() if 'has_service_codes' in company_df.columns else None
            }
        
        return patterns
    
    def _analyze_service_codes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ кодов медицинских услуг"""
        if 'service_codes_list' not in df.columns:
            return {}
        
        # Собираем все коды услуг
        all_codes = []
        for codes in df['service_codes_list']:
            if codes:
                all_codes.extend(codes)
        
        # Анализ частотности кодов
        code_frequency = Counter(all_codes)
        
        # Анализ префиксов кодов (первые две цифры)
        prefix_frequency = Counter()
        for code in all_codes:
            if len(code) >= 3:
                prefix = code[:3]  # F##
                prefix_frequency[prefix] += 1
        
        analysis = {
            'total_unique_codes': len(set(all_codes)),
            'total_code_occurrences': len(all_codes),
            'most_common_codes': dict(code_frequency.most_common(10)),
            'most_common_prefixes': dict(prefix_frequency.most_common(10)),
            'codes_per_letter_stats': {
                'mean': df[df['num_service_codes'] > 0]['num_service_codes'].mean() if 'num_service_codes' in df.columns else None,
                'max': df['num_service_codes'].max() if 'num_service_codes' in df.columns else None,
                'distribution': df['num_service_codes'].value_counts().to_dict() if 'num_service_codes' in df.columns else None
            }
        }
        
        # Анализ по классам
        if 'class' in df.columns:
            analysis['codes_by_class'] = {}
            for class_label in df['class'].unique():
                class_codes = []
                class_df = df[df['class'] == class_label]
                for codes in class_df['service_codes_list']:
                    if codes:
                        class_codes.extend(codes)
                
                analysis['codes_by_class'][f'class_{class_label}'] = {
                    'unique_codes': len(set(class_codes)),
                    'total_occurrences': len(class_codes),
                    'most_common': dict(Counter(class_codes).most_common(5))
                }
        
        return analysis
    
    def _analyze_text_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ текстовых признаков"""
        # TF-IDF анализ
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 3))
        
        try:
            tfidf_matrix = tfidf.fit_transform(df['text_cleaned'])
            feature_names = tfidf.get_feature_names_out()
            
            # Получаем средние TF-IDF веса для каждого признака
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Топ признаки по TF-IDF
            top_indices = np.argsort(mean_tfidf)[-20:][::-1]
            top_features = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
            
            features = {
                'top_tfidf_features': dict(top_features),
                'vocabulary_size': len(feature_names)
            }
            
            # Анализ по классам
            if 'class' in df.columns:
                features['tfidf_by_class'] = {}
                
                for class_label in df['class'].unique():
                    class_texts = df[df['class'] == class_label]['text_cleaned']
                    if len(class_texts) > 0:
                        class_tfidf_matrix = tfidf.transform(class_texts)
                        class_mean_tfidf = np.array(class_tfidf_matrix.mean(axis=0)).flatten()
                        
                        # Топ признаки для класса
                        class_top_indices = np.argsort(class_mean_tfidf)[-10:][::-1]
                        class_top_features = [(feature_names[i], class_mean_tfidf[i]) for i in class_top_indices]
                        
                        features['tfidf_by_class'][f'class_{class_label}'] = dict(class_top_features)
            
        except Exception as e:
            logger.error(f"Error in TF-IDF analysis: {e}")
            features = {'error': str(e)}
        
        return features
    
    def _analyze_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ ключевых слов и фраз"""
        analysis = {
            'class_0_keywords_frequency': {},
            'class_1_keywords_frequency': {}
        }
        
        # Анализ ключевых слов для класса 0
        for keyword in self.class_0_keywords:
            count = df['text_cleaned'].str.contains(keyword, case=False).sum()
            if count > 0:
                analysis['class_0_keywords_frequency'][keyword] = count
        
        # Анализ ключевых слов для класса 1  
        for keyword in self.class_1_keywords:
            count = df['text_cleaned'].str.contains(keyword, case=False).sum()
            if count > 0:
                analysis['class_1_keywords_frequency'][keyword] = count
        
        # Анализ по классам
        if 'class' in df.columns:
            analysis['keyword_effectiveness'] = {}
            
            # Для каждого ключевого слова считаем, насколько хорошо оно различает классы
            all_keywords = self.class_0_keywords + self.class_1_keywords
            
            for keyword in all_keywords:
                class_0_count = df[df['class'] == 0]['text_cleaned'].str.contains(keyword, case=False).sum()
                class_1_count = df[df['class'] == 1]['text_cleaned'].str.contains(keyword, case=False).sum()
                
                total_count = class_0_count + class_1_count
                if total_count > 0:
                    # Вычисляем "чистоту" ключевого слова для каждого класса
                    class_0_purity = class_0_count / total_count
                    class_1_purity = class_1_count / total_count
                    
                    analysis['keyword_effectiveness'][keyword] = {
                        'total_occurrences': total_count,
                        'class_0_ratio': class_0_purity,
                        'class_1_ratio': class_1_purity,
                        'preferred_class': 0 if class_0_purity > class_1_purity else 1
                    }
        
        return analysis
    
    def find_similar_letters(self, df: pd.DataFrame, target_index: int, n_similar: int = 5) -> List[Tuple[int, float]]:
        """
        Найти похожие письма по TF-IDF сходству
        
        Args:
            df: DataFrame с письмами
            target_index: индекс целевого письма
            n_similar: количество похожих писем для возврата
            
        Returns:
            список кортежей (индекс, similarity_score)
        """
        tfidf = TfidfVectorizer(max_features=200)
        tfidf_matrix = tfidf.fit_transform(df['text_cleaned'])
        
        # Вычисляем косинусное сходство
        target_vector = tfidf_matrix[target_index]
        similarities = cosine_similarity(target_vector, tfidf_matrix).flatten()
        
        # Исключаем само письмо и находим наиболее похожие
        similarities[target_index] = -1
        top_indices = np.argsort(similarities)[-n_similar:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices]