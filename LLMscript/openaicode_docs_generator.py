#!/usr/bin/env python3
"""
Sequential Code Analyzer Using OpenAI API with Enhanced Comprehensive Explanations

This script analyzes C, C++, and Python files by asking three sequential questions
in the same conversation context, saving each answer to a separate file:

1. Overview: What is the purpose, functionality, and structure of this code?
2. Line-by-Line: Provides a highly detailed, step-by-step explanation of the code.
3. Improvements: What improvements could be made to this code?

The line-by-line explanation is designed to be extremely comprehensive and accessible
to all levels of programmers, including beginners.

Usage:
  python3 enhanced_code_analyzer.py --dir /path/to/code --model gpt-4o\
  python3 code_docs_generator.py --dir /AI/MachineLearning/DeepLearning/Yolo --model gpt-4o

"""

import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import time
import argparse
import random
import logging
import json
import re
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("code_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not api_key:
    raise ValueError("Missing OpenAI API key. Add it to a .env file.")

# Configure custom session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)

# Create OpenAI client
client = OpenAI(
    api_key=api_key,
    timeout=60.0  # Default timeout of 60 seconds
)

# Supported file extensions - RESTRICTED TO C/C++/PYTHON ONLY
SUPPORTED_EXTENSIONS = ['.cpp', '.c', '.py','.cu' ]

# State tracking
STATE_FILE = "code_analyzer_state.json"

# The three questions to ask - enhanced for more comprehensive explanations
QUESTIONS = [
    "What is the purpose of this code? Explain the main functionality, algorithms used, and the overall structure. Include information about the problem being solved, the approach taken, and how the different parts of the code work together.",
    
    """Provide an extremely comprehensive, step-by-step explanation of the code, as if teaching someone who is learning to program. For each significant section:

1. Explain exactly what it does in simple terms
2. Break down the logic and control flow in detail
3. Define any technical terms or concepts when they first appear
4. Use examples to illustrate complex ideas when helpful
5. Explain WHY certain approaches or techniques are used, not just WHAT they do
6. For complex algorithms or data structures, explain the underlying principles
7. Include simple text-based diagrams where they would help clarify flow or structure

Be especially thorough with loops, conditionals, function calls, and any complex operations. Make no assumptions about the reader's prior knowledge. Your goal is to make this code completely understandable to everyone, from beginners to experts.""",
    
    "What improvements could be made to this code? Consider performance, readability, maintainability, potential bugs, error handling, and best practices. For each suggestion, explain WHY it would be an improvement and HOW it could be implemented with specific code examples where appropriate."
]

# Corresponding output file suffixes
OUTPUT_SUFFIXES = [
    "_overview.md",
    "_line_by_line.md",
    "_improvements.md"
]

def extract_code_essence(file_content, max_length=6000):
    """Extract the essential parts of the code to reduce prompt size."""
    # If code is already small enough, return as is
    if len(file_content) <= max_length:
        return file_content
    
    # Try to keep the structure by extracting:
    # 1. First part (headers, imports, class definitions)
    # 2. Key function definitions
    # 3. Main code section if available
    
    lines = file_content.split('\n')
    
    # Always include the first ~20% of the file (headers, imports, top-level declarations)
    top_section = '\n'.join(lines[:int(len(lines) * 0.2)])
    
    # Extract function/method definitions
    function_pattern = r'(def\s+\w+|class\s+\w+|\w+\s*\(\)|\w+::\w+)'
    functions = re.findall(function_pattern, file_content)
    
    # Look for main function or entry point
    main_section = ""
    main_patterns = ['def main', 'if __name__', 'int main', 'void main', 'public static void main']
    for pattern in main_patterns:
        if pattern in file_content:
            main_match = re.search(pattern + '.*?\{', file_content, re.DOTALL)
            if main_match:
                start_pos = main_match.start()
                # Extract a reasonable chunk around the main function
                main_section = file_content[start_pos:min(start_pos + 1000, len(file_content))]
    
    # Combine the sections with indicators for missing parts
    result = top_section
    if len(functions) > 0:
        result += "\n\n# Key function definitions found in the code:\n"
        result += "\n".join([f"# - {func}" for func in functions[:20]])  # Limit to top 20 functions
    
    if main_section:
        result += "\n\n# Main entry point:\n" + main_section
    
    # If still too long, truncate with a note
    if len(result) > max_length:
        result = result[:max_length] + "\n\n# [Code truncated due to length...]"
    
    return result

def create_initial_prompt(file_content, filename, max_code_length=6000):
    """ Generate an OpenAI-friendly prompt to introduce the code. """
    extension = os.path.splitext(filename)[1].lower()
    
    language_map = {
        '.cpp': "C++", 
        '.c': "C", 
        '.py': "Python",
        '.cu': "CUDA"
    }
    
    language = language_map.get(extension, "Unknown")
    
    # Extract essential parts of the code to reduce prompt size
    code_essence = extract_code_essence(file_content, max_code_length)
    
    prompt = f"""
    I'm going to ask you three sequential questions about the following {language} code file.
    Please analyze this code carefully as I'll be asking about its purpose, a comprehensive line-by-line explanation, and potential improvements.
    
    I need extremely thorough and educational explanations that anyone can understand, regardless of their programming experience level.
    
    Filename: {filename}
    
    Code:
    ```{language.lower()}
    {code_essence}
    ```
    """
    return prompt.strip()

def exponential_backoff(attempt, base_delay=2, max_delay=60):
    """Calculate backoff time with jitter for retries, using more resilient delays."""
    # Exponential backoff with jitter, with more aggressive delays for network issues
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter (±20%)
    jitter = random.uniform(0.8, 1.2)
    return delay * jitter

def chat_with_openai(messages, model_name, max_retries=5):
    """ Make a request to OpenAI's chat API with exponential backoff retry logic. """
    for attempt in range(max_retries):
        try:
            # Add request headers for better error tracking
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=4096,
                temperature=0.2,
                timeout=60  # Explicit request timeout
            )
            return response.choices[0].message.content.strip(), response.choices[0].message
        
        except (openai.APIConnectionError, openai.APITimeoutError, ConnectionError, TimeoutError) as e:
            delay = exponential_backoff(attempt)
            logger.warning(f"Connection error: {e}. Retrying in {delay:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except openai.RateLimitError as e:
            delay = exponential_backoff(attempt, base_delay=5)
            logger.warning(f"Rate limit exceeded: {e}. Waiting {delay:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"API Error: {error_message}")
            
            # Handle model not found error
            if "The model" in error_message and "does not exist" in error_message:
                logger.error(f"The model '{model_name}' was not found. Please check the model name.")
                return None, None
            
            # Handle overloaded server error
            elif "server_error" in error_message or "overloaded" in error_message:
                delay = exponential_backoff(attempt, base_delay=8)
                logger.warning(f"Server overloaded. Waiting {delay:.1f} seconds before retry... (Attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            
            # Other unknown errors
            else:
                if attempt < max_retries - 1:
                    delay = exponential_backoff(attempt)
                    logger.warning(f"Unexpected error. Waiting {delay:.1f} seconds before retry... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed after {max_retries} attempts with error: {error_message}")
                    return None, None
    
    logger.error(f"Failed to retrieve response after {max_retries} attempts.")
    return None, None

def ensure_output_directory(file_dir, output_dir="analysis"):
    """Ensure the output directory exists in the specified directory."""
    output_dir_path = os.path.join(file_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    return output_dir_path

def save_response(response, base_filepath, suffix, output_dir="analysis"):
    """ Save response to a markdown file in the specified output directory. """
    
    file_dir = os.path.dirname(base_filepath)
    base_filename = os.path.basename(base_filepath)
    name_without_ext = os.path.splitext(base_filename)[0]
    
    # Create output directory if it doesn't exist
    output_dir_path = os.path.join(file_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        logger.info(f"Created output directory: {output_dir_path}")
    
    # Create filename for the output
    output_filename = f"{name_without_ext}{suffix}"
    output_filepath = os.path.join(output_dir_path, output_filename)
    
    # Add a heading based on the suffix type
    heading = ""
    if "_overview.md" in suffix:
        heading = f"# Code Overview: {base_filename}\n\n"
    elif "_line_by_line.md" in suffix:
        heading = f"# Step-by-Step Explanation: {base_filename}\n\n"
    elif "_improvements.md" in suffix:
        heading = f"# Suggested Improvements: {base_filename}\n\n"
    
    formatted_response = f"{heading}{response}"
    
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_response)
        logger.info(f"Response saved: {output_filepath}")
        return output_filepath
    except Exception as e:
        logger.error(f"Error saving response to {output_filepath}: {e}")
        return None

def load_state():
    """Load the current processing state from file or create a new one."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
    
    # Create new state if file doesn't exist or is invalid
    return {
        "pending": [],
        "completed": [],
        "failed": [],
        "last_run": None
    }

def save_state(state):
    """Save the current processing state to file."""
    state["last_run"] = datetime.now().isoformat()
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving state file: {e}")
        return False

def find_eligible_files(root_dir, skip_existing=True, output_dir="analysis"):
    """Find all eligible files for processing."""
    state = load_state()
    
    # Clear pending list if it exists
    state["pending"] = []
    
    # Find all eligible files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(subdir, file)
                
                # Check if all three analysis files already exist for this file
                name_without_ext = os.path.splitext(file)[0]
                analysis_dir = os.path.join(subdir, output_dir)
                
                # Skip if all three analysis files already exist and skip_existing is True
                if skip_existing and os.path.exists(analysis_dir):
                    all_files_exist = True
                    for suffix in OUTPUT_SUFFIXES:
                        analysis_filename = f"{name_without_ext}{suffix}"
                        analysis_filepath = os.path.join(analysis_dir, analysis_filename)
                        if not os.path.exists(analysis_filepath):
                            all_files_exist = False
                            break
                    
                    if all_files_exist:
                        if file_path not in state["completed"]:
                            state["completed"].append(file_path)
                        continue
                
                # Skip if already completed or failed
                if file_path in state["completed"] or file_path in state["failed"]:
                    continue
                    
                state["pending"].append(file_path)
    
    save_state(state)
    logger.info(f"Found {len(state['pending'])} pending files, {len(state['completed'])} completed, {len(state['failed'])} failed")
    return state

def process_file(file_path, model_name, max_code_length, output_dir="analysis"):
    """Process a single file with sequential questions in the same context window."""
    
    logger.info(f"Processing: {file_path}")
    
    try:
        # Try with utf-8 encoding first
        try:
            with open(file_path, 'r', encoding='utf-8') as source_file:
                file_content = source_file.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as source_file:
                file_content = source_file.read()

        # Initialize the conversation with the system message and code
        conversation = [
            {"role": "system", "content": "You are an exceptional programming teacher and code reviewer who provides extremely detailed, step-by-step explanations of code. You excel at breaking down complex concepts into simple terms that anyone can understand, while still covering all technical details thoroughly. You use examples, analogies, and clear language to make programming accessible to everyone from beginners to experts."},
            {"role": "user", "content": create_initial_prompt(file_content, os.path.basename(file_path), max_code_length)}
        ]
        
        success = True
        saved_files = []
        
        # Ask each question in sequence
        for i, question in enumerate(QUESTIONS):
            # Add the question to the conversation
            conversation.append({"role": "user", "content": question})
            
            # Get response from OpenAI
            response, message = chat_with_openai(conversation, model_name)
            
            if response:
                # Save the response to a file
                saved_file = save_response(response, file_path, OUTPUT_SUFFIXES[i], output_dir)
                if saved_file:
                    saved_files.append(saved_file)
                    # Add the assistant's response to the conversation to maintain context
                    conversation.append({"role": "assistant", "content": response})
                else:
                    success = False
                    break
            else:
                success = False
                break
                
            # Add a small delay between questions to avoid API rate limits
            if i < len(QUESTIONS) - 1:
                time.sleep(1)
        
        if success and len(saved_files) == len(QUESTIONS):
            logger.info(f"Successfully processed {file_path} with {len(saved_files)} responses")
            return "completed"
        else:
            logger.error(f"Failed to complete all questions for {file_path}")
            return "failed"
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return "failed"

def process_batch(batch_size, model_name, throttle_delay=5, max_code_length=6000, output_dir="analysis"):
    """Process a batch of files with throttling between each file."""
    state = load_state()
    
    # Check if we have pending files
    if not state["pending"]:
        logger.info("No pending files to process.")
        return 0
    
    # Process batch_size files
    files_processed = 0
    for i in range(min(batch_size, len(state["pending"]))):
        if not state["pending"]:
            break
            
        file_path = state["pending"][0]
        result = process_file(file_path, model_name, max_code_length, output_dir)
        
        # Update state based on result
        state["pending"].remove(file_path)
        if result == "completed":
            state["completed"].append(file_path)
        else:
            state["failed"].append(file_path)
        
        # Save state after each file
        save_state(state)
        files_processed += 1
        
        # Apply throttling if not the last file and throttling is enabled
        if throttle_delay > 0 and i < min(batch_size, len(state["pending"])) - 1 and state["pending"]:
            logger.info(f"Throttling for {throttle_delay} seconds before next file...")
            time.sleep(throttle_delay)
    
    return files_processed

def process_directory_with_throttling(root_dir, model_name, batch_size=1, throttle_delay=5, 
                                     skip_existing=True, max_code_length=6000, output_dir="analysis"):
    """Process all files in a directory with throttling and batching."""
    # Find eligible files and initialize state
    state = find_eligible_files(root_dir, skip_existing, output_dir)
    
    if not state["pending"]:
        logger.info("No files to process.")
        return
    
    logger.info(f"Starting batch processing with throttle delay of {throttle_delay}s")
    logger.info(f"Analysis files will be saved to '{output_dir}' folders")
    
    # Process in batches
    total_processed = 0
    while state["pending"]:
        logger.info(f"Processing batch of up to {batch_size} files")
        files_processed = process_batch(batch_size, model_name, throttle_delay, max_code_length, output_dir)
        total_processed += files_processed
        
        if files_processed < batch_size:
            break
            
        # Reload state
        state = load_state()
    
    logger.info(f"Batch processing completed. Total files processed: {total_processed}")
    logger.info(f"Status: {len(state['pending'])} pending, {len(state['completed'])} completed, {len(state['failed'])} failed")

def create_index_file(root_dir, output_dir="analysis"):
    """Create an index file of all generated analyses."""
    index_content = []
    index_content.append("# Comprehensive Code Analysis Index\n")
    index_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    index_content.append("This index contains links to comprehensive, step-by-step explanations of code files.\n\n")
    
    # Track files by language
    languages = {
        '.py': {'name': 'Python', 'files': []},
        '.c': {'name': 'C', 'files': []},
        '.cpp': {'name': 'C++', 'files': []},
        '.cu': {'name': 'CUDA', 'files': []}
    }
    
    # Find all analysis files and group by original file
    file_analyses = {}
    
    for subdir, _, files in os.walk(root_dir):
        if subdir.endswith(output_dir):
            for file in files:
                for suffix in OUTPUT_SUFFIXES:
                    if file.endswith(suffix):
                        name_without_suffix = file[:-len(suffix)]
                        analysis_path = os.path.join(subdir, file)
                        
                        # Add to file_analyses dict
                        if name_without_suffix not in file_analyses:
                            file_analyses[name_without_suffix] = {
                                'name': name_without_suffix,
                                'dir': subdir,
                                'analyses': {}
                            }
                        
                        # Add this analysis type
                        current_type = suffix
                        file_analyses[name_without_suffix]['analyses'][current_type] = analysis_path
    
    # Match files to their original extension
    for name, info in file_analyses.items():
        # Try to determine original file extension
        found = False
        for ext in SUPPORTED_EXTENSIONS:
            code_file = f"{name}{ext}"
            potential_code_path = os.path.join(os.path.dirname(os.path.dirname(info['dir'])), code_file)
            
            if os.path.exists(potential_code_path):
                languages[ext]['files'].append({
                    'name': code_file,
                    'analyses': info['analyses']
                })
                found = True
                break
        
        if not found:
            # If original can't be found, still add to index
            # Choose Python by default, doesn't really matter
            languages['.py']['files'].append({
                'name': name,
                'analyses': info['analyses']
            })
    
    # Create language sections
    for ext, lang_info in languages.items():
        if lang_info['files']:
            index_content.append(f"## {lang_info['name']} Files\n")
            
            for file_info in sorted(lang_info['files'], key=lambda x: x['name']):
                index_content.append(f"### {file_info['name']}\n")
                
                # List each analysis type
                if "_overview.md" in file_info['analyses']:
                    rel_path = os.path.relpath(file_info['analyses']["_overview.md"], start=root_dir)
                    index_content.append(f"* [Overview]({rel_path})\n")
                
                if "_line_by_line.md" in file_info['analyses']:
                    rel_path = os.path.relpath(file_info['analyses']["_line_by_line.md"], start=root_dir)
                    index_content.append(f"* [Step-by-Step Explanation]({rel_path})\n")
                
                if "_improvements.md" in file_info['analyses']:
                    rel_path = os.path.relpath(file_info['analyses']["_improvements.md"], start=root_dir)
                    index_content.append(f"* [Suggested Improvements]({rel_path})\n")
                
                index_content.append("\n")
            
            index_content.append("\n")
    
    # Write index file
    index_path = os.path.join(root_dir, f"{output_dir}_index.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(''.join(index_content))
    
    logger.info(f"Created analysis index: {index_path}")
    return index_path

def main():
    parser = argparse.ArgumentParser(description='Analyze C, C++, and Python code files with comprehensive, step-by-step explanations using OpenAI API')
    parser.add_argument('--dir', type=str, default=".", help='Directory to process')
    parser.add_argument('--model', type=str, default="gpt-4o", 
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--process-all', action='store_true', 
                        help='Process all files, including those that already have analysis')
    parser.add_argument('--throttle', type=int, default=5,
                        help='Seconds to wait between processing files (default: 5, use 0 to disable)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of files to process in one run (default: 5)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset processing state (clears completed/failed lists)')
    parser.add_argument('--max-code-length', type=int, default=6000,
                        help='Maximum length of code to include in prompt (default: 6000)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds for API requests (default: 60)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    parser.add_argument('--output-dir', type=str, default="analysis",
                        help='Name of output directory for analysis files (default: analysis)')
    parser.add_argument('--create-index', action='store_true',
                        help='Create an index file of all analyses (runs after processing)')
    
    # Add help on available models
    parser.add_argument('--list-models', action='store_true',
                        help='List recommended OpenAI models and exit')
    parser.add_argument('--test-connection', action='store_true',
                        help='Test connection to OpenAI API and exit')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Set debugging for urllib3 to see connection details
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Update client timeout if specified
    if args.timeout:
        client.timeout = float(args.timeout)
        logger.info(f"API timeout set to {args.timeout} seconds")
    
    # Show available models if requested
    if args.list_models:
        print("Recommended OpenAI models for comprehensive code analysis:")
        print("  gpt-4o                   (best quality analysis)")
        print("  gpt-4-turbo              (high quality)")
        print("  gpt-4-0125-preview       (high quality)")
        print("  gpt-4                    (high quality, slower)")
        print("  gpt-3.5-turbo            (faster, less detailed)")
        return
    
    # Test API connection if requested
    if args.test_connection:
        print("Testing API connectivity...")
        try:
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=5
            )
            print("✓ Connection to OpenAI API successful!")
            print(f"API response: {test_response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Connection to OpenAI API failed: {e}")
            print("Please check your internet connection and API key")
        return
    
    # Reset state if requested
    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        logger.info("Processing state reset.")
    
    try:
        process_directory_with_throttling(
            args.dir, 
            args.model,
            batch_size=args.batch_size,
            throttle_delay=args.throttle,
            skip_existing=not args.process_all,
            max_code_length=args.max_code_length,
            output_dir=args.output_dir
        )
        
        # Create index file if requested
        if args.create_index:
            create_index_file(args.dir, args.output_dir)
            
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()