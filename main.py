#!/usr/bin/env python3
"""
Главный скрипт для запуска экспериментов по классификации страховых писем
Воспроизводит эксперименты из магистерской диссертации
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys
import os

# Добавляем путь к модулям проекта
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_runner import ExperimentRunner
from src.utils.visualization import create_experiment_dashboard, generate_results_table


def setup_logging(config: dict):
    """Настройка логирования"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log') if config.get('logging', {}).get('save_logs', True) else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)


def run_baseline_only(config: dict):
    """Запуск только базового эксперимента"""
    print("🚀 Запуск базового эксперимента...")
    
    runner = ExperimentRunner(config)
    
    # Загрузка данных
    train_df, test_df, services_df = runner.load_and_analyze_data()
    print(f"✅ Данные загружены: обучение={len(train_df)}, тест={len(test_df)}")
    
    # Базовый эксперимент
    baseline_results = runner.run_baseline_experiment(train_df, test_df)
    runner.results['experiments']['baseline'] = baseline_results
    
    # Отчет
    print("\n📊 Результаты базового эксперимента:")
    print(f"Accuracy: {baseline_results['test_metrics']['accuracy']:.3f}")
    print(f"F1-score: {baseline_results['test_metrics']['f1_weighted']:.3f}")
    print(f"Precision: {baseline_results['test_metrics']['precision_weighted']:.3f}")
    print(f"Recall: {baseline_results['test_metrics']['recall_weighted']:.3f}")
    
    # Сохранение отчета
    summary = runner.generate_summary_report()
    print(f"\n✅ Результаты сохранены в {runner.experiment_dir}")


def run_synthetic_experiments(config: dict):
    """Запуск экспериментов с синтетическими данными"""
    print("🚀 Запуск экспериментов с синтетическими данными...")
    
    runner = ExperimentRunner(config)
    
    # Загрузка данных
    train_df, test_df, services_df = runner.load_and_analyze_data()
    
    # Базовый эксперимент
    print("⚡ Базовый эксперимент...")
    baseline_results = runner.run_baseline_experiment(train_df, test_df)
    runner.results['experiments']['baseline'] = baseline_results
    baseline_f1 = baseline_results['test_metrics']['f1_weighted']
    
    # Синтетическая аугментация
    print("⚡ Эксперименты с синтетическими данными...")
    synthetic_results = runner.run_synthetic_augmentation_experiment(train_df, test_df)
    runner.results['experiments']['synthetic_augmentation'] = synthetic_results
    
    # Анализ результатов
    print("\n📊 Сравнение результатов:")
    print(f"Базовая модель: F1-score = {baseline_f1:.3f}")
    
    best_synthetic = max(synthetic_results.values(), 
                        key=lambda x: x['test_metrics']['f1_weighted'])
    best_f1 = best_synthetic['test_metrics']['f1_weighted']
    best_size = best_synthetic['synthetic_size']
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"Лучший синтетический ({best_size} примеров): F1-score = {best_f1:.3f} (+{improvement:.1f}%)")
    
    # Детализация по размерам
    print("\n📈 Детализация по размерам синтетических данных:")
    for size, result in sorted(synthetic_results.items(), key=lambda x: x[1]['synthetic_size']):
        f1 = result['test_metrics']['f1_weighted']
        size_val = result['synthetic_size']
        improvement = ((f1 - baseline_f1) / baseline_f1) * 100
        print(f"  {size_val:3d} примеров: F1-score = {f1:.3f} ({improvement:+.1f}%)")
    
    # Сохранение отчета
    summary = runner.generate_summary_report()
    print(f"\n✅ Результаты сохранены в {runner.experiment_dir}")


def run_full_experiments(config: dict):
    """Запуск полного набора экспериментов"""
    print("🚀 Запуск полного набора экспериментов...")
    print("   (Базовый + Синтетические данные + LLM + Сравнение классификаторов)")
    
    runner = ExperimentRunner(config)
    
    # Запуск всех экспериментов
    results = runner.run_full_experiment_suite()
    
    # Создание визуализаций
    if config.get('visualization', {}).get('create_dashboard', True):
        print("📊 Создание визуализаций...")
        plots_dir = runner.experiment_dir / 'plots'
        saved_plots = create_experiment_dashboard(results, plots_dir)
        print(f"✅ Создано {len(saved_plots)} графиков в {plots_dir}")
    
    # Создание сводной таблицы
    print("📋 Создание сводной таблицы результатов...")
    results_table = generate_results_table(results)
    table_path = runner.experiment_dir / 'results_table.csv'
    results_table.to_csv(table_path, index=False, encoding='utf-8')
    
    # Печать сводки
    print("\n🎯 СВОДКА РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    
    baseline_f1 = results['experiments']['baseline']['test_metrics']['f1_weighted']
    print(f"Базовая модель (только реальные данные): F1-score = {baseline_f1:.3f}")
    
    # Лучший синтетический результат
    if 'synthetic_augmentation' in results['experiments']:
        synthetic_results = results['experiments']['synthetic_augmentation']
        best_synthetic = max(synthetic_results.values(), 
                           key=lambda x: x['test_metrics']['f1_weighted'])
        best_f1 = best_synthetic['test_metrics']['f1_weighted'] 
        best_size = best_synthetic['synthetic_size']
        improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
        print(f"Лучший с синтетическими данными ({best_size}): F1-score = {best_f1:.3f} (+{improvement:.1f}%)")
    
    # Лучший LLM результат
    if 'llm_generation' in results['experiments']:
        llm_results = results['experiments']['llm_generation']
        best_llm = max(llm_results.values(), 
                      key=lambda x: x['test_metrics']['f1_weighted'])
        llm_f1 = best_llm['test_metrics']['f1_weighted']
        llm_size = best_llm['llm_size']
        llm_improvement = ((llm_f1 - baseline_f1) / baseline_f1) * 100
        print(f"Лучший с LLM данными ({llm_size}): F1-score = {llm_f1:.3f} (+{llm_improvement:.1f}%)")
    
    print("=" * 60)
    print(f"✅ Все результаты сохранены в {runner.experiment_dir}")
    
    # Отчет
    summary = runner.generate_summary_report()
    
    return results


def run_analysis_only(config: dict):
    """Запуск только анализа данных без экспериментов"""
    print("🔍 Анализ данных...")
    
    runner = ExperimentRunner(config)
    train_df, test_df, services_df = runner.load_and_analyze_data()
    
    print(f"\n📊 Статистика данных:")
    print(f"Обучающая выборка: {len(train_df)} примеров")
    print(f"Тестовая выборка: {len(test_df)} примеров")
    print(f"Справочник услуг: {len(services_df)} услуг")
    
    class_dist = train_df['class'].value_counts()
    print(f"\nРаспределение классов в обучающей выборке:")
    print(f"Класс 0 (общие запросы): {class_dist[0]} ({class_dist[0]/len(train_df)*100:.1f}%)")
    print(f"Класс 1 (с кодами услуг): {class_dist[1]} ({class_dist[1]/len(train_df)*100:.1f}%)")
    print(f"Дисбаланс классов: {class_dist[0]/class_dist[1]:.1f}:1")
    
    print(f"\n✅ Анализ данных сохранен в {runner.experiment_dir}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Эксперименты по классификации страховых писем')
    parser.add_argument('--config', default='config.yaml', help='Путь к файлу конфигурации')
    parser.add_argument('--mode', choices=['full', 'baseline', 'synthetic', 'analysis'], 
                       default='full', help='Режим запуска экспериментов')
    
    args = parser.parse_args()
    
    # Проверка существования конфигурации
    if not Path(args.config).exists():
        print(f"❌ Конфигурационный файл {args.config} не найден!")
        sys.exit(1)
    
    # Загрузка конфигурации
    config = load_config(args.config)
    setup_logging(config)
    
    print("🔬 Эксперименты по классификации страховых писем")
    print("📚 Магистерская диссертация - Методы генерации синтетических данных")
    print("🎓 Автор: Нурмухаметов А.Н., НИ ТГУ, 2025")
    print("=" * 70)
    
    # Проверка данных
    data_path = Path(config.get('data_path', 'data/raw'))
    if not data_path.exists():
        print(f"❌ Директория с данными {data_path} не найдена!")
        print("   Убедитесь, что Excel файлы находятся в папке data/raw/")
        sys.exit(1)
    
    try:
        # Запуск соответствующего режима
        if args.mode == 'analysis':
            run_analysis_only(config)
        elif args.mode == 'baseline':
            run_baseline_only(config)
        elif args.mode == 'synthetic':
            run_synthetic_experiments(config)
        elif args.mode == 'full':
            run_full_experiments(config)
        
        print("\n🎉 Эксперименты завершены успешно!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Эксперименты прерваны пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()