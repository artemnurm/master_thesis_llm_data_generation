"""
LLM-based synthetic data generator for insurance letters
Real OpenAI API implementation using gpt-4o-mini with structured output
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
import logging
import json
import re
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class InsuranceLetterOutput(BaseModel):
    """Pydantic model for structured output generation"""
    letter_text: str = Field(description="Generated insurance letter text")
    quality_score: float = Field(description="Self-assessed quality score from model (0.0-1.0)")
    reasoning: str = Field(description="Model's reasoning about how it created this letter")


class LLMGenerator:
    """
    Real LLM-based generator for synthetic insurance letters using OpenAI API
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize the LLM generator
        
        Args:
            model_name: OpenAI model name to use
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key to constructor"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Components for generation
        self.companies = ["СОГАЗ", "Ингосстрах", "РЕСО-Гарантия", "АльфаСтрахование"]
        self.service_code_prefixes = [
            "F59", "F70", "F55", "F71", "F72", "F73", "F74", "F75"
        ]
        
    def _generate_service_codes(self, num_codes: int = None) -> List[str]:
        """Generate medical service codes in format F##.##.##.#.###"""
        if num_codes is None:
            num_codes = random.randint(1, 4)
        
        codes = []
        for _ in range(num_codes):
            prefix = random.choice(self.service_code_prefixes)
            code = f"{prefix}.{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(0, 9)}.{random.randint(100, 999)}"
            codes.append(code)
        
        return codes
    
    def generate_class_0_letter(self, company: str = None) -> Dict[str, Any]:
        """
        Generate class 0 letter (general requests without service codes)
        
        Returns:
            Dictionary with letter text and metadata
        """
        if company is None:
            company = random.choice(self.companies)
        
        system_prompt = """You are an expert in composing insurance documents in Russia. 
Your task is to create a realistic guarantee letter from an insurance company without specifying specific medical service codes.
The letter should contain a general request for organizing medical care."""

        user_prompt = f"""Create a guarantee letter from insurance company {company}.

Requirements:
- The letter should contain ONLY general phrases like "organize medical care", "necessary examinations", "specialist consultations"
- DO NOT specify specific medical service codes (format F##.##.##.#.###)
- Use official business style
- Length: 2-4 sentences
- Include reference to voluntary medical insurance program

Also assess the quality of the created letter and explain your approach to creating it."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=InsuranceLetterOutput,
                temperature=0.7,
                max_tokens=400
            )
            
            result = completion.choices[0].message.parsed
            
            return {
                'letter_id': f'synthetic_llm_0_{random.randint(1000, 9999)}',
                'guarantee_letter_text': result.letter_text,
                'class': 0,
                'quality_score': result.quality_score,
                'reasoning': result.reasoning,
                'generation_method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"Error generating class 0 letter: {e}")
            return self._fallback_class_0_letter(company)
    
    def generate_class_1_letter(self, company: str = None, num_services: int = None) -> Dict[str, Any]:
        """
        Generate class 1 letter (with service codes)
        
        Returns:
            Dictionary with letter text and metadata
        """
        if company is None:
            company = random.choice(self.companies)
        
        service_codes = self._generate_service_codes(num_services)
        
        system_prompt = """You are an expert in composing insurance documents in Russia.
Your task is to create a realistic guarantee letter from an insurance company with specific medical service codes.
The letter should contain a list of specific medical services with their codes."""

        user_prompt = f"""Create a guarantee letter from insurance company {company}.

Requirements:
- MUST include the following service codes: {', '.join(service_codes)}
- The letter should contain a specific list of medical services
- Use official business style
- Length: 3-5 sentences
- You can add diagnosis or indications
- Include information about payment under voluntary medical insurance policy

Also assess the quality of the created letter and explain your approach to creating it."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=InsuranceLetterOutput,
                temperature=0.7,
                max_tokens=500
            )
            
            result = completion.choices[0].message.parsed
            
            return {
                'letter_id': f'synthetic_llm_1_{random.randint(1000, 9999)}',
                'guarantee_letter_text': result.letter_text,
                'class': 1,
                'quality_score': result.quality_score,
                'reasoning': result.reasoning,
                'generation_method': 'llm',
                'service_codes': service_codes
            }
            
        except Exception as e:
            logger.error(f"Error generating class 1 letter: {e}")
            return self._fallback_class_1_letter(company, service_codes)
    
    def _fallback_class_0_letter(self, company: str) -> Dict[str, Any]:
        """Fallback method for class 0 letter generation"""
        templates = [
            f"{company}\n\nОрганизовать необходимые медицинские исследования для пациента в рамках программы добровольного медицинского страхования.",
            f"{company}\n\nПросим организовать приемы врача и диагностические процедуры для застрахованного лица согласно полису ДМС.",
            f"{company}\n\nТребуется проведение лечебных процедур и консультаций врачей согласно условиям договора медицинского страхования."
        ]
        
        return {
            'letter_id': f'fallback_0_{random.randint(1000, 9999)}',
            'guarantee_letter_text': random.choice(templates),
            'class': 0,
            'quality_score': 0.8,
            'reasoning': 'Fallback template generation',
            'generation_method': 'fallback'
        }
    
    def _fallback_class_1_letter(self, company: str, service_codes: List[str]) -> Dict[str, Any]:
        """Fallback method for class 1 letter generation"""
        codes_text = '\n'.join(service_codes)
        letter_text = f"{company}\n\nПросим организовать оплату следующих медицинских услуг:\n{codes_text}\n\nОплата производится в рамках полиса ДМС."
        
        return {
            'letter_id': f'fallback_1_{random.randint(1000, 9999)}',
            'guarantee_letter_text': letter_text,
            'class': 1,
            'quality_score': 0.8,
            'reasoning': 'Fallback template generation',
            'generation_method': 'fallback',
            'service_codes': service_codes
        }
    
    def generate_batch(self, class_distribution: Dict[int, int]) -> List[Dict[str, Any]]:
        """
        Generate batch of synthetic letters
        
        Args:
            class_distribution: {class: count} - class distribution
            
        Returns:
            list of generated letters
        """
        results = []
        
        for class_label, count in class_distribution.items():
            logger.info(f"Generating {count} letters for class {class_label}")
            for i in range(count):
                if class_label == 0:
                    letter_data = self.generate_class_0_letter()
                else:
                    letter_data = self.generate_class_1_letter()
                
                if letter_data:
                    results.append(letter_data)
                    if (i + 1) % 5 == 0:
                        logger.info(f"Generated {i + 1}/{count} letters for class {class_label}")
        
        return results
    
    def generate_synthetic_dataset(self, target_counts: Dict[int, int]) -> pd.DataFrame:
        """
        Generate synthetic dataset
        
        Args:
            target_counts: number of examples for each class
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating synthetic dataset with LLM: {target_counts}")
        
        all_letters = self.generate_batch(target_counts)
        
        # Convert to DataFrame
        data_for_df = []
        for letter in all_letters:
            data_for_df.append({
                'letter_id': letter['letter_id'],
                'guarantee_letter_text': letter['guarantee_letter_text'],
                'class': letter['class'],
                'quality_score': letter['quality_score'],
                'reasoning': letter.get('reasoning', ''),
                'generation_method': letter['generation_method']
            })
        
        df = pd.DataFrame(data_for_df)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} synthetic letters")
        logger.info(f"Average quality score: {df['quality_score'].mean():.3f}")
        
        return df
    
    def validate_generation_quality(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate quality of generated data
        
        Args:
            synthetic_df: DataFrame with synthetic data
            
        Returns:
            Dictionary with quality metrics
        """
        if synthetic_df.empty:
            return {}
        
        # Basic metrics
        metrics = {
            'total_samples': len(synthetic_df),
            'avg_quality_score': synthetic_df['quality_score'].mean(),
            'min_quality_score': synthetic_df['quality_score'].min(),
            'class_balance': abs(0.5 - (synthetic_df['class'] == 0).mean()),
        }
        
        # Check presence of service codes in class 1
        class_1_samples = synthetic_df[synthetic_df['class'] == 1]
        if not class_1_samples.empty:
            contains_codes = class_1_samples['guarantee_letter_text'].str.contains(
                r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}', regex=True
            ).mean()
            metrics['class_1_code_presence'] = contains_codes
        
        # Check absence of codes in class 0
        class_0_samples = synthetic_df[synthetic_df['class'] == 0]
        if not class_0_samples.empty:
            no_codes = 1 - class_0_samples['guarantee_letter_text'].str.contains(
                r'F\d{2}\.\d{2}\.\d{2}\.\d\.\d{3}', regex=True
            ).mean()
            metrics['class_0_no_codes'] = no_codes
        
        return metrics


def main():
    """Test the generator"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize generator
        generator = LLMGenerator()
        print("✅ LLM generator initialized")
        
        # Test generation of one letter of each class
        print("\n🧪 Testing generation...")
        
        # Class 0
        letter_0 = generator.generate_class_0_letter()
        if letter_0:
            print(f"\n📝 Class 0 letter (quality: {letter_0['quality_score']:.2f}):")
            print(letter_0['guarantee_letter_text'])
            print(f"💭 Reasoning: {letter_0['reasoning']}")
        
        # Class 1
        letter_1 = generator.generate_class_1_letter()
        if letter_1:
            print(f"\n📝 Class 1 letter (quality: {letter_1['quality_score']:.2f}):")
            print(letter_1['guarantee_letter_text'])
            print(f"💭 Reasoning: {letter_1['reasoning']}")
        
        print("\n✅ Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()