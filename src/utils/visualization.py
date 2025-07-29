"""
Visualization utilities for experiment results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")


def plot_class_distribution(data: pd.DataFrame, 
                           class_column: str = 'class',
                           title: str = "Распределение классов",
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Визуализация распределения классов
    
    Args:
        data: DataFrame с данными
        class_column: название колонки с классами
        title: заголовок графика
        save_path: путь для сохранения
        
    Returns:
        фигура matplotlib  
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Столбчатая диаграмма
    class_counts = data[class_column].value_counts().sort_index()
    ax1.bar(class_counts.index.astype(str), class_counts.values)
    ax1.set_title(f'{title} (Абсолютные значения)')
    ax1.set_xlabel('Класс')
    ax1.set_ylabel('Количество примеров')
    
    # Добавляем подписи значений
    for i, (class_label, count) in enumerate(class_counts.items()):
        ax1.text(i, count + max(class_counts.values()) * 0.01, 
                str(count), ha='center', va='bottom')
    
    # Круговая диаграмма
    ax2.pie(class_counts.values, labels=[f'Класс {i}' for i in class_counts.index], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} (Проценты)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str] = None,
                         title: str = "Матрица ошибок",
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Визуализация матрицы ошибок
    
    Args:
        cm: матрица ошибок
        class_names: названия классов
        title: заголовок
        save_path: путь для сохранения
        
    Returns:
        фигура matplotlib
    """
    if class_names is None:
        class_names = [f'Класс {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Тепловая карта
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Предсказанные классы')
    ax.set_ylabel('Истинные классы')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    return fig


def plot_metrics_comparison(results_dict: Dict[str, Dict[str, Any]],
                           metrics: List[str] = None,
                           title: str = "Сравнение метрик",
                           save_path: Optional[Path] = None) -> plt.Figure:
    """
    Сравнение метрик между экспериментами
    
    Args:
        results_dict: результаты экспериментов
        metrics: список метрик для отображения
        title: заголовок
        save_path: путь для сохранения
        
    Returns:
        фигура matplotlib
    """
    if metrics is None:
        metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    # Подготовка данных
    data = []
    for exp_name, exp_data in results_dict.items():
        if 'test_metrics' in exp_data:
            test_metrics = exp_data['test_metrics']
            for metric in metrics:
                if metric in test_metrics:
                    data.append({
                        'experiment': exp_name,
                        'metric': metric,
                        'value': test_metrics[metric]
                    })
    
    df = pd.DataFrame(data)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Сгруппированная столбчатая диаграмма  
    sns.barplot(data=df, x='experiment', y='value', hue='metric', ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Эксперименты')
    ax.set_ylabel('Значение метрики')
    plt.xticks(rotation=45, ha='right')
    
    # Добавляем значения на столбцы
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    return fig


def plot_ablation_study(ablation_results: Dict[str, Dict[str, Any]],
                       x_key: str = 'synthetic_size',
                       y_key: str = 'f1_weighted',
                       title: str = "Ablation Study",
                       save_path: Optional[Path] = None) -> plt.Figure:
    """
    Визуализация результатов ablation study
    
    Args:
        ablation_results: результаты ablation study  
        x_key: ключ для оси X
        y_key: ключ для оси Y (метрика)
        title: заголовок
        save_path: путь для сохранения
        
    Returns:
        фигура matplotlib
    """
    # Подготовка данных
    x_values = []
    y_values = []
    
    for exp_name, exp_data in ablation_results.items():
        if x_key in exp_data and 'test_metrics' in exp_data:
            x_values.append(exp_data[x_key])
            y_values.append(exp_data['test_metrics'][y_key])
    
    # Сортировка по X
    sorted_data = sorted(zip(x_values, y_values))
    x_values, y_values = zip(*sorted_data)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Линейный график с точками
    ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=8)
    
    # Добавление значений на точки
    for x, y in zip(x_values, y_values):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center')
    
    ax.set_title(title)
    ax.set_xlabel(x_key.replace('_', ' ').title())
    ax.set_ylabel(y_key.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Ablation study plot saved to {save_path}")
    
    return fig


def plot_training_curves(history: Dict[str, List[float]],
                        title: str = "Кривые обучения",
                        save_path: Optional[Path] = None) -> plt.Figure:
    """
    Визуализация кривых обучения (если доступны)
    
    Args:
        history: история обучения
        title: заголовок  
        save_path: путь для сохранения
        
    Returns:
        фигура matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        if metric in history and f'val_{metric}' in history:
            ax = axes[i]
            
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}')
            ax.plot(epochs, history[f'val_{metric}'], 'r-', label=f'Validation {metric}')
            
            ax.set_title(f'{metric.title()} во время обучения')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel(metric.title())
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves plot saved to {save_path}")
    
    return fig


def create_experiment_dashboard(results: Dict[str, Any],
                              save_dir: Path) -> List[Path]:
    """
    Создание комплексного дашборда с результатами экспериментов
    
    Args:
        results: все результаты экспериментов
        save_dir: директория для сохранения
        
    Returns:
        список путей к созданным графикам
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []
    
    try:
        # 1. Сравнение всех экспериментов
        if 'experiments' in results:
            all_experiments = {}
            
            # Собираем результаты базового эксперимента
            if 'baseline' in results['experiments']:
                all_experiments['baseline'] = results['experiments']['baseline']
            
            # Собираем лучшие результаты синтетических экспериментов
            if 'synthetic_augmentation' in results['experiments']:
                synthetic_results = results['experiments']['synthetic_augmentation']
                best_synthetic = max(synthetic_results.items(), 
                                   key=lambda x: x[1]['test_metrics']['f1_weighted'])
                all_experiments[f'synthetic_{best_synthetic[1]["synthetic_size"]}'] = best_synthetic[1]
            
            # Собираем результаты LLM экспериментов
            if 'llm_generation' in results['experiments']:
                llm_results = results['experiments']['llm_generation']
                best_llm = max(llm_results.items(),
                              key=lambda x: x[1]['test_metrics']['f1_weighted'])
                all_experiments[f'llm_{best_llm[1]["llm_size"]}'] = best_llm[1]
            
            # График сравнения
            comparison_path = save_dir / 'metrics_comparison.png'
            plot_metrics_comparison(all_experiments, save_path=comparison_path)
            saved_plots.append(comparison_path)
        
        # 2. Ablation study для синтетических данных
        if 'synthetic_augmentation' in results['experiments']:
            ablation_path = save_dir / 'synthetic_ablation.png'
            plot_ablation_study(
                results['experiments']['synthetic_augmentation'],
                title="Влияние размера синтетических данных на качество",
                save_path=ablation_path
            )
            saved_plots.append(ablation_path)
        
        # 3. Матрицы ошибок для лучших моделей
        if 'baseline' in results['experiments']:
            baseline = results['experiments']['baseline']
            cm_path = save_dir / 'baseline_confusion_matrix.png'
            plot_confusion_matrix(
                np.array(baseline['confusion_matrix']),
                class_names=['Общие запросы', 'Специфические коды'],
                title="Матрица ошибок: Базовая модель",
                save_path=cm_path
            )
            saved_plots.append(cm_path)
        
        logger.info(f"Experiment dashboard created with {len(saved_plots)} plots in {save_dir}")
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
    
    return saved_plots


def generate_results_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Генерация сводной таблицы результатов
    
    Args:
        results: результаты экспериментов
        
    Returns:
        DataFrame с результатами
    """
    table_data = []
    
    if 'experiments' in results:
        experiments = results['experiments']
        
        # Базовый эксперимент
        if 'baseline' in experiments:
            baseline = experiments['baseline']
            table_data.append({
                'Эксперимент': 'Базовая модель (только реальные данные)',
                'Размер обучающей выборки': baseline['train_size'],
                'Синтетические данные': 0,
                'Accuracy': f"{baseline['test_metrics']['accuracy']:.3f}",
                'F1-score': f"{baseline['test_metrics']['f1_weighted']:.3f}",
                'Precision': f"{baseline['test_metrics']['precision_weighted']:.3f}",
                'Recall': f"{baseline['test_metrics']['recall_weighted']:.3f}"
            })
        
        # Синтетические эксперименты
        if 'synthetic_augmentation' in experiments:
            for exp_name, exp_data in experiments['synthetic_augmentation'].items():
                table_data.append({
                    'Эксперимент': f'Шаблонные синтетические данные ({exp_data["synthetic_size"]})',
                    'Размер обучающей выборки': exp_data['total_train_size'],
                    'Синтетические данные': exp_data['synthetic_size'],
                    'Accuracy': f"{exp_data['test_metrics']['accuracy']:.3f}",
                    'F1-score': f"{exp_data['test_metrics']['f1_weighted']:.3f}",
                    'Precision': f"{exp_data['test_metrics']['precision_weighted']:.3f}",
                    'Recall': f"{exp_data['test_metrics']['recall_weighted']:.3f}"
                })
        
        # LLM эксперименты
        if 'llm_generation' in experiments:
            for exp_name, exp_data in experiments['llm_generation'].items():
                table_data.append({
                    'Эксперимент': f'LLM синтетические данные ({exp_data["llm_size"]})',
                    'Размер обучающей выборки': exp_data['total_train_size'],
                    'Синтетические данные': exp_data['llm_size'],
                    'Accuracy': f"{exp_data['test_metrics']['accuracy']:.3f}",
                    'F1-score': f"{exp_data['test_metrics']['f1_weighted']:.3f}",
                    'Precision': f"{exp_data['test_metrics']['precision_weighted']:.3f}",
                    'Recall': f"{exp_data['test_metrics']['recall_weighted']:.3f}"
                })
    
    df = pd.DataFrame(table_data)
    return df