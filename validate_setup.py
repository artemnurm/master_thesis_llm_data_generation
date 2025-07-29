#!/usr/bin/env python3
"""
Валидация настройки проекта
Проверяет корректность установки и структуры проекта
"""

import sys
from pathlib import Path
import importlib

def check_python_version():
    """Проверка версии Python"""
    print("🐍 Проверка версии Python...")
    if sys.version_info < (3, 8):
        print(f"❌ Требуется Python 3.8+, найден {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

def check_project_structure():
    """Проверка структуры проекта"""
    print("\n📁 Проверка структуры проекта...")
    
    required_dirs = [
        'src',
        'src/data',
        'src/generation', 
        'src/models',
        'src/experiments',
        'src/utils',
        'data',
        'data/raw',
        'results',
        'notebooks'
    ]
    
    required_files = [
        'main.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/data/data_loader.py',
        'src/data/data_analyzer.py',
        'src/generation/template_generator.py',
        'src/generation/llm_generator.py',
        'src/models/classifier.py',
        'src/experiments/experiment_runner.py',
        'src/utils/metrics.py',
        'src/utils/visualization.py'
    ]
    
    all_good = True
    
    # Проверка директорий
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - отсутствует")
            all_good = False
    
    # Проверка файлов
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - отсутствует")
            all_good = False
    
    return all_good

def check_data_files():
    """Проверка файлов данных"""
    print("\n📊 Проверка файлов данных...")
    
    data_files = [
        'data/raw/DS_хакатон_набор данных_train_231208_1030.xlsx',
        'data/raw/DS_хакатон_набор данных_test_231208_1030.xlsx', 
        'data/raw/DS_хакатон_справочник_услуг_231208_1030.xlsx'
    ]
    
    all_good = True
    for file_path in data_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - отсутствует")
            print(f"   Скопируйте из original medical_services_classification/data/raw/")
            all_good = False
    
    return all_good

def check_imports():
    """Проверка импортов модулей"""
    print("\n📦 Проверка импортов...")
    
    # Основные зависимости
    dependencies = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    # Специализированные зависимости (опционально)
    optional_deps = [
        'sentence_transformers',
        'transformers', 
        'torch',
        'yaml'
    ]
    
    all_good = True
    
    # Проверка основных зависимостей
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - не найден")
            all_good = False
    
    # Проверка опциональных зависимостей
    print("\nОпциональные зависимости:")
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"⚠️  {dep} - не найден (потребуется для полного функционала)")
    
    return all_good

def check_project_modules():
    """Проверка импортов модулей проекта"""
    print("\n🔧 Проверка модулей проекта...")
    
    # Добавляем путь к проекту
    sys.path.insert(0, str(Path.cwd()))
    
    project_modules = [
        'src.data.data_loader',
        'src.data.data_analyzer',
        'src.generation.template_generator',
        'src.generation.llm_generator',
        'src.models.classifier',
        'src.experiments.experiment_runner',
        'src.utils.metrics',
        'src.utils.visualization'
    ]
    
    all_good = True
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - ошибка импорта: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {module} - предупреждение: {e}")
    
    return all_good

def run_basic_functionality_test():
    """Базовый тест функциональности"""
    print("\n🧪 Базовый тест функциональности...")
    
    try:
        # Тест data loader
        sys.path.insert(0, str(Path.cwd()))
        from src.data.data_loader import InsuranceDataLoader
        
        loader = InsuranceDataLoader()
        print("✅ InsuranceDataLoader создан")
        
        # Тест template generator
        from src.generation.template_generator import TemplateDataGenerator
        generator = TemplateDataGenerator()
        
        # Генерация тестового примера
        test_letter = generator.generate_class_0_letter()
        if len(test_letter) > 10:
            print("✅ Генерация синтетических данных работает")
        else:
            print("❌ Проблема с генерацией данных")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в базовом тесте: {e}")
        return False

def main():
    """Главная функция валидации"""
    print("🔍 Валидация настройки проекта")
    print("=" * 50)
    
    checks = [
        ("Версия Python", check_python_version),
        ("Структура проекта", check_project_structure),
        ("Файлы данных", check_data_files),
        ("Зависимости", check_imports),
        ("Модули проекта", check_project_modules),
        ("Базовая функциональность", run_basic_functionality_test)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Ошибка в проверке '{name}': {e}")
            results.append((name, False))
    
    # Сводка
    print("\n" + "=" * 50)
    print("📋 СВОДКА ПРОВЕРОК:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ ПРОШЛА" if passed else "❌ НЕ ПРОШЛА"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ВСЕ ПРОВЕРКИ ПРОШЛИ УСПЕШНО!")
        print("Проект готов к запуску экспериментов.")
        print("\nДля запуска экспериментов используйте:")
        print("python main.py --mode analysis")
    else:
        print("⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        print("Устраните указанные проблемы перед запуском экспериментов.")
        print("\nДля установки зависимостей:")
        print("pip install -r requirements.txt")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)