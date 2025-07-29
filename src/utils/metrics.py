"""
Metrics calculation utilities for classification experiments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)
import logging

logger = logging.getLogger(__name__)


def calculate_comprehensive_metrics(y_true: List[int], 
                                  y_pred: List[int], 
                                  y_proba: List[List[float]] = None) -> Dict[str, Any]:
    """
    Вычисление всех необходимых метрик для бинарной классификации
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки  
        y_proba: вероятности классов (опционально)
        
    Returns:
        словарь с метриками
    """
    try:
        # Основные метрики
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # Метрики по классам
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        for class_label in ['0', '1']:
            if class_label in report:
                metrics[f'precision_class_{class_label}'] = report[class_label]['precision']
                metrics[f'recall_class_{class_label}'] = report[class_label]['recall']  
                metrics[f'f1_class_{class_label}'] = report[class_label]['f1-score']
                metrics[f'support_class_{class_label}'] = report[class_label]['support']
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Детальная информация из матрицы ошибок
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Специфичность и чувствительность
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC AUC если есть вероятности
        if y_proba is not None:
            try:
                if len(y_proba[0]) == 2:  # бинарная классификация
                    y_proba_class_1 = [p[1] for p in y_proba]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba_class_1)
                    metrics['avg_confidence'] = np.mean(y_proba_class_1)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
                metrics['avg_confidence'] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'error': str(e)}


def compare_experiments(results_dict: Dict[str, Dict[str, Any]], 
                       metric_keys: List[str] = None) -> pd.DataFrame:
    """
    Сравнение результатов нескольких экспериментов
    
    Args:
        results_dict: словарь с результатами экспериментов
        metric_keys: ключи метрик для сравнения
        
    Returns:
        DataFrame с сравнением
    """
    if metric_keys is None:
        metric_keys = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    comparison_data = []
    
    for experiment_name, experiment_data in results_dict.items():
        if 'test_metrics' in experiment_data:
            metrics = experiment_data['test_metrics']
            row = {'experiment': experiment_name}
            
            for key in metric_keys:
                row[key] = metrics.get(key, 0.0)
            
            # Дополнительная информация
            if 'train_size' in experiment_data:
                row['train_size'] = experiment_data['train_size']
            if 'synthetic_size' in experiment_data:
                row['synthetic_size'] = experiment_data['synthetic_size']
            if 'llm_size' in experiment_data:
                row['llm_size'] = experiment_data['llm_size']
                
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Сортировка по F1-score
    if 'f1_weighted' in df.columns:
        df = df.sort_values('f1_weighted', ascending=False)
    
    return df


def calculate_statistical_significance(results1: List[float], 
                                     results2: List[float],
                                     test_type: str = 'paired_t') -> Dict[str, float]:
    """
    Вычисление статистической значимости различий между экспериментами
    
    Args:
        results1: результаты первого эксперимента
        results2: результаты второго эксперимента  
        test_type: тип теста ('paired_t', 'wilcoxon', 'mcnemar')
        
    Returns:
        результаты статистического теста
    """
    try:
        from scipy import stats
        
        if test_type == 'paired_t':
            statistic, p_value = stats.ttest_rel(results1, results2)
        elif test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results1, results2)
        else:
            logger.warning(f"Unknown test type: {test_type}. Using paired t-test.")
            statistic, p_value = stats.ttest_rel(results1, results2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'test_type': test_type
        }
    
    except ImportError:
        logger.warning("scipy not available for statistical tests")
        return {'error': 'scipy not available'}
    except Exception as e:
        logger.error(f"Error in statistical test: {e}")
        return {'error': str(e)}


def analyze_class_performance(y_true: List[int], 
                             y_pred: List[int],
                             class_names: List[str] = None) -> Dict[str, Any]:
    """
    Детальный анализ производительности по классам
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        class_names: названия классов
        
    Returns:
        анализ по классам
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    analysis = {
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true),
        'class_distribution': {
            'true': pd.Series(y_true).value_counts().to_dict(),
            'predicted': pd.Series(y_pred).value_counts().to_dict()
        }
    }
    
    # Анализ по каждому классу
    for i, class_name in enumerate(class_names):
        class_key = str(i)
        if class_key in report:
            class_metrics = report[class_key]
            
            analysis[f'class_{i}_analysis'] = {
                'name': class_name,
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1_score': class_metrics['f1-score'],
                'support': class_metrics['support']
            }
            
            # Ошибки для этого класса
            true_class_mask = np.array(y_true) == i
            pred_class_mask = np.array(y_pred) == i
            
            correct_predictions = np.sum(true_class_mask & pred_class_mask)
            missed_predictions = np.sum(true_class_mask & ~pred_class_mask)
            false_predictions = np.sum(~true_class_mask & pred_class_mask)
            
            analysis[f'class_{i}_analysis'].update({
                'correct_predictions': int(correct_predictions),
                'missed_predictions': int(missed_predictions),
                'false_predictions': int(false_predictions)
            })
    
    return analysis


def calculate_improvement_metrics(baseline_metrics: Dict[str, float],
                                improved_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Вычисление метрик улучшения относительно базовой модели
    
    Args:
        baseline_metrics: метрики базовой модели
        improved_metrics: метрики улучшенной модели
        
    Returns:
        метрики улучшения
    """
    improvements = {}
    
    key_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    for metric in key_metrics:
        if metric in baseline_metrics and metric in improved_metrics:
            baseline_val = baseline_metrics[metric]
            improved_val = improved_metrics[metric]
            
            if baseline_val > 0:
                absolute_improvement = improved_val - baseline_val
                relative_improvement = (absolute_improvement / baseline_val) * 100
                
                improvements[f'{metric}_absolute_improvement'] = absolute_improvement
                improvements[f'{metric}_relative_improvement'] = relative_improvement
                improvements[f'{metric}_is_better'] = improved_val > baseline_val
    
    # Общая оценка улучшения
    f1_improvement = improvements.get('f1_weighted_relative_improvement', 0)
    improvements['overall_improvement'] = f1_improvement
    
    if f1_improvement > 5:
        improvements['improvement_level'] = 'significant'
    elif f1_improvement > 1:
        improvements['improvement_level'] = 'moderate'
    elif f1_improvement > 0:
        improvements['improvement_level'] = 'slight'
    else:
        improvements['improvement_level'] = 'none_or_negative'
    
    return improvements