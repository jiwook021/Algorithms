# python3 openaiQuestions.py openaiQuestion.txt --output_dir openaiQuestionStudy_1

import os
import re
import time
import openai
import argparse
from pathlib import Path

def setup_openai_client(api_key):
    """
    Set up and return the OpenAI client using the provided API key.
    
    Args:
        api_key (str): OpenAI API key
    
    Returns:
        openai.Client: Configured OpenAI client
    """
    return openai.Client(api_key=api_key)

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
        print(f"File content sample: {content[:200]}...")  # Print first 200 chars for debugging
        
        # Match lines starting with bullet points (*, -, •)
        questions = re.findall(r'^\s*[\*\-•]\s*(.+)', content, re.MULTILINE)
        
        print(f"Extracted {len(questions)} questions from the file")
        if len(questions) > 0:
            print(f"First question sample: {questions[0]}")
            
        return questions
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def get_answer_from_openai(client, question, model="gpt-4-turbo"):
    """
    Get an answer for a question using the OpenAI API.
    
    Args:
        client (openai.Client): OpenAI API client
        question (str): The question to answer
        model (str): The OpenAI model to use
    
    Returns:
        str: Answer to the question
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed, accurate, and educational answers to questions."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting answer for question: {question}")
        print(f"Error details: {e}")
        return f"Error: Could not generate answer due to {str(e)}"

def create_markdown_file(question, answer, output_dir):
    """
    Create a markdown file for a question and its answer.
    
    Args:
        question (str): The question
        answer (str): The answer
        output_dir (str): Directory to save the markdown file
    
    Returns:
        str: Path to the created markdown file
    """
    # Create a sanitized filename from the question
    filename = re.sub(r'[^\w\s-]', '', question).strip().lower()
    filename = re.sub(r'[\s]+', '-', filename)
    
    # Truncate filename if it's too long
    if len(filename) > 100:
        filename = filename[:100]
    
    file_path = os.path.join(output_dir, f"{filename}.md")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {question}\n\n")
        f.write(answer)
    
    return file_path

def main():
    """
    Main function to process questions from a text file and create markdown files with OpenAI API answers.
    """
    parser = argparse.ArgumentParser(description='Process a text file of bulleted questions using OpenAI API.')
    parser.add_argument('input_file', help='Text file containing bulleted questions (using *, -, or • as bullets)')
    parser.add_argument('--output_dir', default='study', help='Directory to save markdown files (default: study)')
    parser.add_argument('--api_key', help='OpenAI API key (alternatively use OPENAI_API_KEY environment variable)')
    parser.add_argument('--model', default='gpt-4-turbo', help='OpenAI model to use (default: gpt-4-turbo)')
    
    args = parser.parse_args()
    
    # Verify file exists and is a text file
    if not os.path.isfile(args.input_file):
        print(f"Error: The file '{args.input_file}' does not exist.")
        return
        
    file_extension = os.path.splitext(args.input_file)[1].lower()
    if file_extension != '.txt' and file_extension:
        print(f"Warning: The file '{args.input_file}' doesn't have a .txt extension. Make sure it's a text file.")
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not provided. Use --api_key argument or set OPENAI_API_KEY environment variable.")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up OpenAI client
    client = setup_openai_client(api_key)
    
    # Extract questions from file
    questions = extract_questions(args.input_file)
    
    if not questions:
        print(f"No bulleted questions found in {args.input_file}")
        return
    
    print(f"Found {len(questions)} questions. Processing...")
    
    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing: {question[:50]}...")
        
        # Get answer from OpenAI
        answer = get_answer_from_openai(client, question, args.model)
        
        # Create markdown file
        file_path = create_markdown_file(question, answer, str(output_dir))
        
        print(f"Created: {file_path}")
        
        # Add a small delay to avoid rate limits
        if i < len(questions):
            time.sleep(0.5)
    
    print(f"\nAll {len(questions)} questions processed. Markdown files saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()