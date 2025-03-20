#python3 svgGemini.py openaiQuestions3.txt --output_dir="visual_explanations" --model="gemini-2.0-pro-exp-02-05" --min_svgs=5


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
import json
from pathlib import Path
import xml.etree.ElementTree as ET

def setup_gemini_model(api_key, model_name="gemini-2.0-pro-exp-02-05"):
    """
    Set up and return the Gemini model with the provided API key and model name.
    Pro model is preferred for SVG generation capabilities.
    
    Args:
        api_key (str): Gemini API key
        model_name (str): Name of the Gemini model (default: gemini-2.0-pro-exp-02-05)
    
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

def extract_svgs_from_text(text):
    """
    Extract SVG content from the text response.
    
    Args:
        text (str): Text containing SVG content
    
    Returns:
        list: List of SVG strings
    """
    # Pattern to match SVG content (from <svg to </svg>)
    svg_pattern = r'<svg[\s\S]*?<\/svg>'
    svgs = re.findall(svg_pattern, text)
    
    # Validate each SVG by trying to parse it
    valid_svgs = []
    for svg in svgs:
        try:
            ET.fromstring(svg)
            valid_svgs.append(svg)
        except ET.ParseError:
            print(f"Found invalid SVG: {svg[:50]}...")
            
    return valid_svgs

def main():
    parser = argparse.ArgumentParser(description="Process questions and generate SVG explanations with Gemini API")
    parser.add_argument("input_file", type=str, help="Path to the input question file")
    parser.add_argument("--output_dir", type=str, default="svg_explanations", help="Output directory")
    parser.add_argument("--model", type=str, default="gemini-2.0-pro-exp-02-05", help="Gemini model name")
    parser.add_argument("--min_svgs", type=int, default=4, help="Minimum number of SVGs per question")
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
        
        # Create sanitized folder name for this question
        folder_name = sanitize_filename(question[:50])
        question_dir = output_dir / folder_name
        question_dir.mkdir(exist_ok=True)
        
        # Save original question in the folder
        with open(question_dir / "question.txt", "w", encoding="utf-8") as f:
            f.write(question)
        
        # Create a prompt requesting SVG visual explanations
        prompt = f"""
        Create {args.min_svgs} detailed SVG visual explanations for the following question:
        
        {question}
        
        For each explanation:
        1. Create a self-contained SVG that visually explains one key concept or step
        2. Each SVG should have a viewport with width and height attributes
        3. Add descriptive text elements within the SVG to label important parts
        4. Use colors effectively to distinguish different elements
        5. Make the diagrams educational and clear
        
        Provide ONLY the SVG code in your response, with each SVG separated by "---SVG_SEPARATOR---"
        """
        
        try:
            # Get response from Gemini
            response = model.generate_content(prompt)
            
            # Try to extract SVGs from the response
            svgs = extract_svgs_from_text(response.text)
            
            # Additional extraction attempt if separator was used correctly
            if "---SVG_SEPARATOR---" in response.text:
                svg_blocks = response.text.split("---SVG_SEPARATOR---")
                for block in svg_blocks:
                    block_svgs = extract_svgs_from_text(block)
                    svgs.extend(block_svgs)
            
            # Remove duplicates while preserving order
            unique_svgs = []
            for svg in svgs:
                if svg not in unique_svgs:
                    unique_svgs.append(svg)
            
            # Check if we have enough SVGs
            if len(unique_svgs) < args.min_svgs:
                print(f"Warning: Only generated {len(unique_svgs)} SVGs for question {i}, trying again...")
                
                # Try again with a more explicit prompt
                retry_prompt = f"""
                I need exactly {args.min_svgs} different SVG diagrams to explain this concept:
                
                {question}
                
                Make each SVG self-contained and educational. Each SVG should explain a different aspect of the concept.
                Include text labels within the SVGs. Only output SVG code, nothing else.
                """
                
                retry_response = model.generate_content(retry_prompt)
                retry_svgs = extract_svgs_from_text(retry_response.text)
                
                # Add new unique SVGs
                for svg in retry_svgs:
                    if svg not in unique_svgs:
                        unique_svgs.append(svg)
            
            # Save the SVGs in the question folder
            for j, svg in enumerate(unique_svgs, 1):
                svg_filename = f"explanation_{j}.svg"
                with open(question_dir / svg_filename, "w", encoding="utf-8") as f:
                    f.write(svg)
                print(f"  - Created SVG {j}: {svg_filename}")
            
            # If still not enough SVGs, save the full response for debugging
            if len(unique_svgs) < args.min_svgs:
                print(f"Warning: Still only generated {len(unique_svgs)} SVGs out of {args.min_svgs} requested")
                with open(question_dir / "full_response.txt", "w", encoding="utf-8") as f:
                    f.write(response.text)
            
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error details: {e}")
            # Save error information
            with open(question_dir / "error.txt", "w", encoding="utf-8") as f:
                f.write(f"Error processing: {str(e)}")
        
        # Wait between requests to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    main()