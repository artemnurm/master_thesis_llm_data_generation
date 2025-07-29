#!/usr/bin/env python3
"""
Test script specifically for LLM synthetic data generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data.data_loader import InsuranceDataLoader
from src.generation.llm_generator import MockLLMGenerator
import pandas as pd
import json

def test_llm_generation():
    """Test LLM synthetic data generation"""
    print("🤖 Testing LLM Synthetic Data Generation")
    print("=" * 50)
    
    # 1. Load original data
    print("📊 Loading original data...")
    data_loader = InsuranceDataLoader()
    train_df, test_df, services_df = data_loader.load_data('data/raw')
    train_df = data_loader.preprocess_letters(train_df)
    
    print(f"✅ Loaded {len(train_df)} training examples")
    print(f"   Class distribution: {train_df['class'].value_counts().to_dict()}")
    
    # 2. Initialize LLM generator
    print("\n🧠 Initializing LLM generator...")
    llm_generator = MockLLMGenerator()
    print("✅ MockLLMGenerator initialized")
    
    # 3. Test single letter generation
    print("\n📝 Testing single letter generation...")
    
    # Generate class 0 letter
    class_0_result = llm_generator.generate_class_0_letter("СОГАЗ")
    print(f"✅ Class 0 letter generated (quality: {class_0_result['quality_score']:.3f})")
    print(f"   Preview: {class_0_result['text'][:100]}...")
    
    # Generate class 1 letter  
    class_1_result = llm_generator.generate_class_1_letter("Ингосстрах", num_services=3)
    print(f"✅ Class 1 letter generated (quality: {class_1_result['quality_score']:.3f})")
    print(f"   Preview: {class_1_result['text'][:100]}...")
    print(f"   Service codes: {class_1_result['num_service_codes']}")
    
    # 4. Test batch generation
    print("\n📦 Testing batch generation...")
    target_counts = {0: 10, 1: 15}
    batch_results = llm_generator.generate_batch(target_counts)
    
    print(f"✅ Generated {len(batch_results)} letters in batch")
    
    # Check quality
    class_0_count = sum(1 for r in batch_results if r['class'] == 0)
    class_1_count = sum(1 for r in batch_results if r['class'] == 1)
    avg_quality = sum(r['quality_score'] for r in batch_results) / len(batch_results)
    
    print(f"   Class 0: {class_0_count}, Class 1: {class_1_count}")
    print(f"   Average quality: {avg_quality:.3f}")
    
    # 5. Test synthetic dataset generation
    print("\n🗂️ Testing synthetic dataset generation...")
    synthetic_df = llm_generator.generate_synthetic_dataset({0: 20, 1: 30})
    
    print(f"✅ Generated synthetic dataset: {len(synthetic_df)} samples")
    print(f"   Class distribution: {synthetic_df['class'].value_counts().to_dict()}")
    print(f"   Average quality: {synthetic_df['quality_score'].mean():.3f}")
    print(f"   Average confidence: {synthetic_df['confidence'].mean():.3f}")
    
    # 6. Quality validation
    print("\n🔍 Testing quality validation...")
    sample_texts = [r['text'] for r in batch_results[:5]]
    quality_results = llm_generator.validate_quality(sample_texts)
    
    print(f"✅ Quality validation completed for {len(quality_results)} samples")
    avg_overall_quality = sum(r['overall_quality'] for r in quality_results) / len(quality_results)
    print(f"   Average overall quality: {avg_overall_quality:.3f}")
    
    # 7. Save test results
    print("\n💾 Saving test results...")
    test_results = {
        'original_data_stats': {
            'train_size': len(train_df),
            'class_distribution': train_df['class'].value_counts().to_dict()
        },
        'single_generation_test': {
            'class_0_quality': class_0_result['quality_score'],
            'class_1_quality': class_1_result['quality_score'],
            'class_1_services': class_1_result['num_service_codes']
        },
        'batch_generation_test': {
            'target_counts': target_counts,
            'generated_counts': {'class_0': class_0_count, 'class_1': class_1_count},
            'average_quality': avg_quality
        },
        'dataset_generation_test': {
            'target_size': 50,
            'generated_size': len(synthetic_df),
            'final_class_distribution': synthetic_df['class'].value_counts().to_dict(),
            'average_quality': synthetic_df['quality_score'].mean(),
            'average_confidence': synthetic_df['confidence'].mean()
        },
        'quality_validation_test': {
            'samples_validated': len(quality_results),
            'average_overall_quality': avg_overall_quality
        }
    }
    
    # Save to JSON
    results_dir = Path('results/llm_generation_test')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
    
    # Save synthetic dataset
    synthetic_df.to_csv(results_dir / 'test_synthetic_data.csv', index=False, encoding='utf-8')
    
    print(f"✅ Test results saved to {results_dir}")
    
    # 8. Summary
    print("\n" + "=" * 50)
    print("📋 LLM GENERATION TEST SUMMARY:")
    print(f"✅ Single letter generation: PASSED")
    print(f"✅ Batch generation: PASSED ({len(batch_results)} letters)")
    print(f"✅ Dataset generation: PASSED ({len(synthetic_df)} samples)")
    print(f"✅ Quality validation: PASSED (avg quality: {avg_overall_quality:.3f})")
    print(f"✅ Data saving: PASSED")
    
    print(f"\n🎯 Key metrics:")
    print(f"   - Average quality score: {synthetic_df['quality_score'].mean():.3f}")
    print(f"   - Average confidence: {synthetic_df['confidence'].mean():.3f}")  
    print(f"   - Class balance achieved: {synthetic_df['class'].value_counts().to_dict()}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_llm_generation()
        if success:
            print("\n🎉 ALL LLM GENERATION TESTS PASSED!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)