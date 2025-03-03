# models/gemini-2.0-flash-exp
# models/gemini-2.0-flash
# models/gemini-2.0-flash-001
# models/gemini-2.0-flash-exp-image-generation
# models/gemini-2.0-flash-lite-001
# models/gemini-2.0-flash-lite
# models/gemini-2.0-flash-lite-preview-02-05
# models/gemini-2.0-flash-lite-preview
# models/gemini-2.0-pro-exp
# models/gemini-2.0-pro-exp-02-05
# models/gemini-exp-1206
# models/gemini-2.0-flash-thinking-exp-01-21
# models/gemini-2.0-flash-thinking-exp
# models/gemini-2.0-flash-thinking-exp-1219
# models/learnlm-1.5-pro-experimental
# models/gemma-3-27b-it
# models/embedding-001
# models/text-embedding-004
# models/gemini-embedding-exp-03-07
# models/gemini-embedding-exp
# models/aqa
# models/imagen-3.0-generate-002

import os
import re
import google.generativeai as genai
import argparse
import time
from pathlib import Path

def setup_gemini_model(api_key, model_name="gemini-2.0-flash"):
    """
    Set up and return the Gemini model with the provided API key and model name.
    
    Args:
        api_key (str): Gemini API key
        model_name (str): Name of the Gemini model (default: gemini-2.0-flash)
    
    Returns:
        genai.GenerativeModel: Configured Gemini model
    """
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error setting up model '{model_name}': {e}")
        print("Available models:")
        for model in genai.list_models():
            print(f"- {model.name}")
        raise

def extract_questions(file_path):
    """
    Extract bulleted questions from a text file.
    
    Args:
        file_path (str): Path to the text file containing bulleted questions
    
    Returns:
        list: List of questions
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Successfully read file: {file_path}")
        print(f"File content sample: {content[:200]}...")
        
        # Match lines starting with bullet points (*, -, •)
        questions = re.findall(r'^\s*[\*\-•]\s*(.+)', content, re.MULTILINE)
        
        print(f"Extracted {len(questions)} questions from the file")
        if questions:
            print(f"First question sample: {questions[0]}")
            
        return questions
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def sanitize_filename(filename):
    """
    Sanitize the filename by removing special characters and replacing spaces with hyphens.
    
    Args:
        filename (str): The original filename
    
    Returns:
        str: Sanitized filename
    """
    # Remove special characters, keep alphanumeric and spaces
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    # Replace spaces with hyphens
    sanitized = re.sub(r'\s+', '-', sanitized)
    return sanitized.lower()

def main():
    parser = argparse.ArgumentParser(description="Process questions with Gemini API")
    parser.add_argument("input_file", type=str, help="Path to the input question file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    model = setup_gemini_model(api_key, args.model)
    
    # Extract questions from the input file
    questions = extract_questions(args.input_file)
    if not questions:
        print("No questions found in the input file.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing: {question[:50]}...")
        try:
            # Create a prompt with instructions for step-by-step explanation
            prompt = "Explain the answer to this question in simple terms, step by step, so that someone without prior knowledge can understand: " + question
            response = model.generate_content(prompt)
            # Sanitize the filename using the first 50 characters of the question
            sanitized_filename = sanitize_filename(question[:50])
            output_file = output_dir / f"{sanitized_filename}.md"
            with open(output_file, "w") as f:
                f.write(response.text)
            print(f"Created: {output_file}")
        except Exception as e:
            print(f"Error getting answer for question: {question}")
            print(f"Error details: {e}")
        time.sleep(0.5)  # Rate limiting

if __name__ == "__main__":
    main()