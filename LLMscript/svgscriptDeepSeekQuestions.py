#!/usr/bin/env python3
"""
SVG Generator using DeepSeek API

This script generates SVG visual explanations for technical questions using DeepSeek's
AI models. It takes questions from a text file, generates multiple SVG diagrams for 
each question, and saves them in an organized directory structure.

# python3 svgscriptDeepSeekQuestions.py openaiQuestions_os_Computervision.txt --output_dir="OS_VISION" --model="deepseek-coder-v2" --min_svgs=10

Usage:
    python3 deepseek_svg_generator.py input_file.txt --output_dir="output_folder" --min_svgs=10

Requirements:
    - Python 3.7+
    - requests package
    - Valid DeepSeek API key set as DEEPSEEK_API_KEY environment variable

Author: JK Engineer
"""

import os   
import re
import argparse
import time
import json
import requests
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
import html

# Try to import lxml for better XML handling
try:
    from lxml import etree
    USE_LXML = True
    print("Using lxml for enhanced XML handling")
except ImportError:
    USE_LXML = False
    print("Note: For better SVG parsing, install lxml: pip install lxml")


class DeepSeekClient:
    """Client for interacting with the DeepSeek API"""
    
    BASE_URL = "https://api.deepseek.com/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_completion(self, 
                           prompt: str, 
                           model: str = "deepseek-coder-v2", 
                           max_tokens: int = 4000,
                           temperature: float = 0.2) -> str:
        """
        Generate a completion using the DeepSeek API.
        
        Args:
            prompt: The prompt to send to the API
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter (0.0 to 1.0)
            
        Returns:
            Generated text as a string
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert at creating SVG diagrams to explain technical concepts. You only respond with valid, well-formed SVG code."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/chat/completions", 
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error from DeepSeek API: {response.status_code}")
                print(response.text)
                return ""
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return ""


def extract_questions(file_path: str) -> List[str]:
    """
    Extract questions from a text file. Supports both:
    - Bulleted lists (lines starting with *, -, •)
    - Plain text (one question per line)
    
    Args:
        file_path: Path to the text file containing questions
    
    Returns:
        List of questions
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Successfully read file: {file_path}")
        print(f"File content sample: {content[:100]}..." if len(content) > 100 else f"File content: {content}")
        
        # First try to match bulleted questions
        questions = re.findall(r'^\s*[\*\-•]\s*(.+)', content, re.MULTILINE)
        
        # If no bulleted questions found, treat each non-empty line as a question
        if not questions:
            questions = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"Extracted {len(questions)} questions from the file")
        if questions:
            print(f"First question sample: {questions[0]}")
            
        return questions
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by removing special characters and replacing spaces with hyphens.
    
    Args:
        filename: The original filename
    
    Returns:
        Sanitized filename
    """
    # Remove special characters, keep alphanumeric and spaces
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    # Replace spaces with hyphens
    sanitized = re.sub(r'\s+', '-', sanitized)
    # Ensure the filename is not too long for file systems
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized.lower()


def fix_svg(svg_text: str) -> Tuple[bool, str]:
    """
    Attempt to fix common SVG errors.
    
    Returns:
        Tuple of (success, fixed_svg)
    """
    # Check if it's already valid
    try:
        if USE_LXML:
            etree.fromstring(svg_text)
        else:
            ET.fromstring(svg_text)
        return True, svg_text
    except (ET.ParseError, etree.XMLSyntaxError) as e:
        print(f"  - Attempting to fix SVG: {e}")
    
    fixed_svg = svg_text
    
    # Fix 1: Handle unescaped ampersands in text or attributes
    fixed_svg = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', fixed_svg)
    
    # Fix 2: Replace problematic quotes in text fields
    # Pattern to find attributes with values
    attr_pattern = r'(\w+)=(".*?"|\'.*?\')'
    def clean_attr(match):
        attr_name = match.group(1)
        attr_value = match.group(2)
        if attr_name.lower() in ['style', 'font-family', 'text']:
            # Escape any remaining special chars in these attribute values
            inner_value = attr_value[1:-1]  # Remove surrounding quotes
            escaped_value = html.escape(inner_value)
            return f'{attr_name}="{escaped_value}"'
        return match.group(0)
    
    fixed_svg = re.sub(attr_pattern, clean_attr, fixed_svg)
    
    # Fix 3: Fix improper CDATA sections
    fixed_svg = re.sub(r'<!\[CDATA\[(.*?)\]\]>', 
                       lambda m: f'<![CDATA[{html.escape(m.group(1))}]]>', 
                       fixed_svg)
    
    # Fix 4: Close unclosed tags
    # This is a simplified approach - for complex cases, lxml would be better
    tag_pattern = r'<(\w+)([^>]*?)(?<!/)(>)(?!.*?</\1>)'
    fixed_svg = re.sub(tag_pattern, r'<\1\2/\3', fixed_svg)
    
    # Check if fixes worked
    try:
        if USE_LXML:
            etree.fromstring(fixed_svg)
        else:
            ET.fromstring(fixed_svg)
        print("  - Successfully fixed SVG")
        return True, fixed_svg
    except (ET.ParseError, etree.XMLSyntaxError):
        print("  - Could not fix SVG")
        return False, svg_text


def extract_svgs_from_text(text: str) -> List[str]:
    """
    Extract SVG content from the text response with improved validation.
    
    Args:
        text: Text containing SVG content
    
    Returns:
        List of valid SVG strings
    """
    # Pattern to match SVG content (from <svg to </svg>)
    svg_pattern = r'<svg[\s\S]*?<\/svg>'
    svgs = re.findall(svg_pattern, text)
    
    # Validate and try to fix each SVG
    valid_svgs = []
    for i, svg in enumerate(svgs):
        valid, fixed_svg = fix_svg(svg)
        if valid:
            valid_svgs.append(fixed_svg)
            print(f"  - Found valid SVG #{i+1}, size: {len(fixed_svg)} characters")
        else:
            print(f"  - Found invalid SVG #{i+1} that couldn't be fixed")
            print(f"    SVG snippet: {svg[:50]}...")
            
    return valid_svgs


def generate_svgs_in_batches(
    client: DeepSeekClient, 
    question: str, 
    min_svgs: int = 10, 
    language: str = "C++",
    model: str = "deepseek-coder-v2", 
    max_tokens: int = 4000,
    max_attempts: int = 5,
    batch_size: int = 3
) -> List[str]:
    """
    Generate SVGs in multiple batches to reach the minimum number.
    
    Args:
        client: Configured DeepSeek client
        question: The question to explain with SVGs
        min_svgs: Minimum number of SVGs to generate
        language: Programming language to use in examples
        model: DeepSeek model to use
        max_tokens: Maximum tokens to generate in the response
        max_attempts: Maximum number of generation attempts
        batch_size: Number of SVGs to request in each batch
        
    Returns:
        List of valid SVG strings
    """
    all_svgs = []
    attempts = 0
    
    # Initial attempt with larger batch
    initial_batch = min(min_svgs, 5)
    
    while len(all_svgs) < min_svgs and attempts < max_attempts:
        attempts += 1
        remaining = min_svgs - len(all_svgs)
        current_batch = initial_batch if attempts == 1 else min(batch_size, remaining)
        
        print(f"  - Batch generation attempt {attempts}/{max_attempts}: requesting {current_batch} SVGs ({len(all_svgs)}/{min_svgs} collected so far)")
        
        # Create a prompt for this batch
        prompt = f"""
        Create {current_batch} detailed SVG visual explanations for the following question:
        
        {question}
        
        For each explanation:
        1. Create a self-contained SVG that visually explains one key concept or step
        2. Each SVG should have a viewport with width="600" height="400" viewBox="0 0 600 400"
        3. Add text elements within the SVG to label important parts
        4. Use distinct colors to highlight different elements
        5. Make the diagrams educational and clear
        6. Use {language} in any code examples
        7. Each SVG must be valid XML and properly formatted
        
        Focus on {current_batch} DIFFERENT aspects of the question, making each SVG unique.
        
        Provide ONLY valid SVG code in your response. Separate each SVG with "---SVG_SEPARATOR---"
        Do not include any explanations outside of the SVGs.
        """
        
        try:
            # Get response from DeepSeek
            response_text = client.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=0.2 + (attempts * 0.1)  # Increase temperature slightly with each attempt
            )
            
            if not response_text:
                print("  - Received empty response from DeepSeek API")
                continue
                
            # Split by separator if present
            if "---SVG_SEPARATOR---" in response_text:
                svg_blocks = response_text.split("---SVG_SEPARATOR---")
                batch_svgs = []
                for block in svg_blocks:
                    block_svgs = extract_svgs_from_text(block)
                    batch_svgs.extend(block_svgs)
            else:
                # Try to extract directly
                batch_svgs = extract_svgs_from_text(response_text)
            
            # Add new unique SVGs to our collection
            for svg in batch_svgs:
                if svg not in all_svgs:
                    all_svgs.append(svg)
            
            print(f"  - Found {len(batch_svgs)} new SVGs in this batch, total: {len(all_svgs)}/{min_svgs}")
            
            # If we didn't get any SVGs in this batch, adjust our approach
            if len(batch_svgs) == 0 and attempts < max_attempts:
                print("  - No SVGs found in this batch, changing approach for next attempt")
        
        except Exception as e:
            print(f"  - Error generating SVGs in batch {attempts}: {e}")
        
        # Short pause between batch requests
        if len(all_svgs) < min_svgs and attempts < max_attempts:
            time.sleep(2)
    
    return all_svgs


def save_svgs(svgs: List[str], output_dir: Path) -> None:
    """
    Save SVGs to individual files in the specified directory.
    
    Args:
        svgs: List of SVG strings
        output_dir: Directory path to save the SVGs
    """
    for i, svg in enumerate(svgs, 1):
        svg_filename = f"explanation_{i}.svg"
        try:
            with open(output_dir / svg_filename, "w", encoding="utf-8") as f:
                f.write(svg)
            print(f"  - Saved SVG {i}: {svg_filename}")
        except Exception as e:
            print(f"  - Error saving {svg_filename}: {e}")


def main():
    """
    Main function that processes the command-line arguments and runs the SVG generation process.
    """
    parser = argparse.ArgumentParser(description="Generate SVG explanations for technical questions using DeepSeek API")
    parser.add_argument("input_file", type=str, help="Path to the input question file")
    parser.add_argument("--output_dir", type=str, default="svg_explanations", help="Output directory")
    parser.add_argument("--model", type=str, default="deepseek-coder-v2", help="DeepSeek model to use")
    parser.add_argument("--min_svgs", type=int, default=5, help="Minimum number of SVGs per question")
    parser.add_argument("--language", type=str, default="C++", help="Programming language for code examples")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API requests (seconds)")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens in DeepSeek's response")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of SVGs to request in each batch")
    parser.add_argument("--max_attempts", type=int, default=5, help="Maximum attempts per question")
    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set")
        print("Please set it with: export DEEPSEEK_API_KEY='your_api_key'")
        return 1

    try:
        # Setup the DeepSeek client
        print(f"Setting up DeepSeek client with model: {args.model}")
        client = DeepSeekClient(api_key)
        
        # Extract questions from the input file
        print(f"Extracting questions from: {args.input_file}")
        questions = extract_questions(args.input_file)
        if not questions:
            print("No questions found in the input file.")
            return 1

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing question: {question[:50]}...")
            
            # Create sanitized folder name for this question
            folder_name = sanitize_filename(question[:50])
            question_dir = output_dir / folder_name
            question_dir.mkdir(exist_ok=True)
            
            # Save original question in the folder
            with open(question_dir / "question.txt", "w", encoding="utf-8") as f:
                f.write(question)
            
            # Generate SVGs for the question in batches
            svgs = generate_svgs_in_batches(
                client, 
                question,
                min_svgs=args.min_svgs,
                language=args.language,
                model=args.model,
                max_tokens=args.max_tokens,
                max_attempts=args.max_attempts,
                batch_size=args.batch_size
            )
            
            # Save the SVGs
            save_svgs(svgs, question_dir)
            
            # Log completion status
            if len(svgs) >= args.min_svgs:
                print(f"  ✅ Successfully generated {len(svgs)} SVGs for question {i}")
            else:
                print(f"  ⚠️ Only generated {len(svgs)}/{args.min_svgs} SVGs for question {i}")
            
            # Wait between questions to avoid rate limiting
            if i < len(questions):
                print(f"  - Waiting {args.delay} seconds before next question...")
                time.sleep(args.delay)

        print(f"\nProcess completed. Generated SVGs for {len(questions)} questions in {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())