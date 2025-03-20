#!/usr/bin/env python3
"""
Multi-SVG Diagram Generator for Code Files

This script generates multiple SVG diagrams that explain the logic and functionality of code files
using OpenAI's API. Each code file will have multiple diagrams saved in an 'imgs' directory.

Usage:
  python3 multi_svg_generator_openai.py --dir /path/to/code --model gpt-4-turbo --throttle 5
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
        logging.FileHandler("svg_generation.log"),
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

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.cpp', '.c', '.py']

# State tracking
STATE_FILE = "svg_generation_state.json"

def extract_code_essence(file_content, max_length=3000):
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
                main_section = file_content[start_pos:min(start_pos + 800, len(file_content))]
    
    # Combine the sections with indicators for missing parts
    result = top_section
    if len(functions) > 0:
        result += "\n\n# Key function definitions found in the code:\n"
        result += "\n".join([f"# - {func}" for func in functions[:15]])  # Limit to top 15 functions
    
    if main_section:
        result += "\n\n# Main entry point:\n" + main_section
    
    # If still too long, truncate with a note
    if len(result) > max_length:
        result = result[:max_length] + "\n\n# [Code truncated due to length...]"
    
    return result

def create_prompt(file_content, filename, max_code_length=3000):
    """ Generate an OpenAI-friendly prompt to create multiple SVG diagrams explaining the code. """
    language = "C++" if filename.endswith('.cpp') else "C" if filename.endswith('.c') else "Python"
    
    # Extract essential parts of the code to reduce prompt size
    code_essence = extract_code_essence(file_content, max_code_length)
    
    prompt = f"""
    Create MULTIPLE SVG diagrams (at least 5) that explain different aspects of this {language} code step by step.
    Each diagram should focus on a different part of the code's functionality:
    
    Requirements:
    - Generate ONLY valid SVG code with no explanations between SVGs
    - Keep each SVG small (around 1200x800)
    - Use professional color schemes with good contrast
    - Add descriptive titles within each SVG
    - Use readable font sizes (at least 12px)
    - Include clear labels for all components
    - Use arrows to show flow direction
    - Separate each SVG diagram with a line of "---" 
    - Output should only contain SVG diagrams with separators between them
    
    Filename: {filename}
    
    Code:
    ```
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

def request_svg_diagrams(prompt, model_name, max_retries=5):
    """ Request multiple SVG diagrams from OpenAI's API with exponential backoff retry logic. """
    for attempt in range(max_retries):
        try:
            # Add request headers for better error tracking
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a technical diagram expert who creates clear, detailed SVG visualizations to explain code. Create professional diagrams with labeled components, clear workflows, and properly styled elements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,  # Need more tokens for multiple SVGs
                temperature=0.2,
                timeout=60  # Explicit request timeout
            )
            return response.choices[0].message.content.strip()
        
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
                return None
            
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
                    return None
    
    logger.error(f"Failed to retrieve SVGs after {max_retries} attempts.")
    return None

def extract_multiple_svgs(response_text):
    """ Extract all SVG content from OpenAI's response. """
    # Look for content between svg tags
    svg_matches = re.findall(r'<svg[\s\S]*?<\/svg>', response_text)
    
    if not svg_matches:
        logger.warning("No SVG tags found in the response.")
        # Try to save the response anyway, it might still be useful
        return [response_text]
    
    return svg_matches

def ensure_imgs_directory(file_dir):
    """Ensure the svgimg directory exists in the specified directory."""
    svgimg_dir = os.path.join(file_dir, "svgimg")
    if not os.path.exists(svgimg_dir):
        os.makedirs(svgimg_dir)
    return svgimg_dir

def save_multiple_svgs(svg_contents, base_filepath, output_dir="svgimg"):
    """ Save multiple SVG contents to files in the specified output directory. """
    
    file_dir = os.path.dirname(base_filepath)
    base_filename = os.path.basename(base_filepath)
    name_without_ext = os.path.splitext(base_filename)[0]
    
    # Create output directory if it doesn't exist
    output_dir_path = os.path.join(file_dir, output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        logger.info(f"Created output directory: {output_dir_path}")
    
    saved_files = []
    for i, svg_content in enumerate(svg_contents):
        # Create a filename like "filename_diagram1.svg", "filename_diagram2.svg", etc.
        svg_filename = f"{name_without_ext}_diagram{i+1}.svg"
        svg_filepath = os.path.join(output_dir_path, svg_filename)
        
        try:
            with open(svg_filepath, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            saved_files.append(svg_filepath)
            logger.info(f"SVG saved: {svg_filepath}")
        except Exception as e:
            logger.error(f"Error saving SVG to {svg_filepath}: {e}")
    
    return saved_files

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

def find_eligible_files(root_dir, skip_existing=True):
    """Find all eligible files for processing."""
    state = load_state()
    
    # Clear pending list if it exists
    state["pending"] = []
    
    # Find all eligible files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(subdir, file)
                
                # Check if svgimg directory already has diagrams for this file
                name_without_ext = os.path.splitext(file)[0]
                svgimg_dir = os.path.join(subdir, "svgimg")
                
                # Skip if SVGs already exist in svgimg directory and skip_existing is True
                if skip_existing and os.path.exists(svgimg_dir):
                    existing_svgs = [f for f in os.listdir(svgimg_dir) if f.startswith(name_without_ext) and f.endswith('.svg')]
                    if existing_svgs:
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

def process_file(file_path, model_name, max_code_length, output_dir="svgimg"):
    """Process a single file and generate multiple SVG diagrams."""
    
    logger.info(f"Processing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
            file_content = source_file.read()

        prompt = create_prompt(file_content, os.path.basename(file_path), max_code_length)
        svg_response = request_svg_diagrams(prompt, model_name)

        if svg_response:
            # Extract multiple SVGs
            svg_contents = extract_multiple_svgs(svg_response)
            
            if svg_contents:
                # Save SVGs to specified output directory
                saved_files = save_multiple_svgs(svg_contents, file_path, output_dir)
                
                if saved_files:
                    logger.info(f"Generated {len(saved_files)} SVG diagrams for {file_path}")
                    return "completed"
                else:
                    logger.error(f"Failed to save any SVGs for {file_path}")
                    return "failed"
            else:
                logger.error(f"No valid SVGs found in response for {file_path}")
                return "failed"
        else:
            logger.error(f"Failed to generate SVGs for {file_path}")
            return "failed"
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return "failed"

def process_batch(batch_size, model_name, throttle_delay=5, max_code_length=3000, output_dir="svgimg"):
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
                                     skip_existing=True, max_code_length=3000, output_dir="svgimg"):
    """Process all files in a directory with throttling and batching."""
    # Find eligible files and initialize state
    state = find_eligible_files(root_dir, skip_existing)
    
    if not state["pending"]:
        logger.info("No files to process.")
        return
    
    logger.info(f"Starting batch processing with throttle delay of {throttle_delay}s")
    logger.info(f"SVG files will be saved to '{output_dir}' folders")
    
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

def main():
    parser = argparse.ArgumentParser(description='Generate multiple SVG diagrams for code files using OpenAI API')
    parser.add_argument('--dir', type=str, default=".", help='Directory to process')
    parser.add_argument('--model', type=str, default="gpt-4o", 
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--process-all', action='store_true', 
                        help='Process all files, including those that already have SVGs')
    parser.add_argument('--throttle', type=int, default=5,
                        help='Seconds to wait between processing files (default: 5, use 0 to disable)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of files to process in one run (default: 5)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset processing state (clears completed/failed lists)')
    parser.add_argument('--max-code-length', type=int, default=3000,
                        help='Maximum length of code to include in prompt (default: 3000)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds for API requests (default: 60)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose logging')
    parser.add_argument('--output-dir', type=str, default="svgimg",
                        help='Name of output directory for SVG files (default: svgimg)')
    
    # Add help on available models
    parser.add_argument('--list-models', action='store_true',
                        help='List available OpenAI models names and exit')
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
        print("Available OpenAI models for SVG generation:")
        print("  gpt-4o                   (best quality SVG diagrams)")
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
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()