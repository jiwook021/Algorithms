#!/usr/bin/env python3
"""
Multi-SVG Diagram Generator for Code Files

This script generates multiple SVG diagrams that explain the logic and functionality of code files
using Anthropic's Claude API. Each code file will have multiple diagrams saved in an 'imgs' directory.

Usage:
  python3 multi_svg_generator.py --dir /path/to/code --model claude-3-sonnet-20240229 --throttle 5
"""

import os
import anthropic
from dotenv import load_dotenv
import time
import argparse
import random
import logging
import json
import re
from datetime import datetime

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
api_key = os.getenv("ANTHROPIC_API_KEY")

# Ensure API key is loaded
if not api_key:
    raise ValueError("Missing Anthropic API key. Add it to a .env file.")

# Create Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.cpp', '.c', '.py','cu']

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
    """ Generate a Claude-friendly prompt to create multiple SVG diagrams explaining the code. """
    language = "C++" if filename.endswith('.cpp') else "C" if filename.endswith('.c') else "Python"
    
    # Extract essential parts of the code to reduce prompt size
    code_essence = extract_code_essence(file_content, max_code_length)
    
    prompt = f"""
    Create MULTIPLE SVG diagrams (at least 5) that explain different aspects of this {language} code.
    Each diagram should focus on a different part of the code's functionality:
    
    1. First diagram: Overall architecture or main workflow
    2. Second diagram: Key algorithms or processes
    3. Third diagram: Data flow or state transitions
    4. Additional diagrams: Any other important aspects worth visualizing
    
    Requirements:
    - Generate ONLY valid SVG code with no explanations between SVGs
    - Keep each SVG small (around 600x400 pixels)
    - Add descriptive titles within each SVG
    - Separate each SVG diagram with a line of "---" 
    - Output should only contain SVG diagrams with separators between them
    
    Filename: {filename}
    
    Code:
    ```
    {code_essence}
    ```
    """
    return prompt.strip()

def exponential_backoff(attempt, base_delay=2, max_delay=30):
    """Calculate backoff time with jitter for retries, using shorter delays."""
    # Exponential backoff with jitter, but with shorter delays
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add jitter (±20%)
    jitter = random.uniform(0.8, 1.2)
    return delay * jitter

def request_svg_diagrams(prompt, model_name, max_retries=5):
    """ Request multiple SVG diagrams from Claude's API with exponential backoff retry logic. """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,  # Need more tokens for multiple SVGs
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        except anthropic.APIConnectionError as e:
            delay = exponential_backoff(attempt)
            logger.warning(f"Connection error: {e}. Retrying in {delay:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except anthropic.RateLimitError as e:
            delay = exponential_backoff(attempt, base_delay=5)
            logger.warning(f"Rate limit exceeded: {e}. Waiting {delay:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"API Error: {error_message}")
            
            # Handle model not found error
            if "not_found_error" in error_message:
                logger.error(f"The model '{model_name}' was not found. Please check the model name.")
                return None
            
            # Handle overloaded server error
            elif "overloaded_error" in error_message:
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
    """ Extract all SVG content from Claude's response. """
    # Look for content between svg tags
    svg_matches = re.findall(r'<svg[\s\S]*?<\/svg>', response_text)
    
    if not svg_matches:
        logger.warning("No SVG tags found in the response.")
        # Try to save the response anyway, it might still be useful
        return [response_text]
    
    return svg_matches

def ensure_imgs_directory(file_dir):
    """Ensure the imgs directory exists in the specified directory."""
    imgs_dir = os.path.join(file_dir, "imgs")
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    return imgs_dir

def save_multiple_svgs(svg_contents, base_filepath):
    """ Save multiple SVG contents to files in the imgs directory. """
    
    file_dir = os.path.dirname(base_filepath)
    base_filename = os.path.basename(base_filepath)
    name_without_ext = os.path.splitext(base_filename)[0]
    
    # Create imgs directory if it doesn't exist
    imgs_dir = ensure_imgs_directory(file_dir)
    
    saved_files = []
    for i, svg_content in enumerate(svg_contents):
        # Create a filename like "filename_diagram1.svg", "filename_diagram2.svg", etc.
        svg_filename = f"{name_without_ext}_diagram{i+1}.svg"
        svg_filepath = os.path.join(imgs_dir, svg_filename)
        
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
                
                # Check if imgs directory already has diagrams for this file
                name_without_ext = os.path.splitext(file)[0]
                imgs_dir = os.path.join(subdir, "imgs")
                
                # Skip if SVGs already exist in imgs directory and skip_existing is True
                if skip_existing and os.path.exists(imgs_dir):
                    existing_svgs = [f for f in os.listdir(imgs_dir) if f.startswith(name_without_ext) and f.endswith('.svg')]
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

def process_file(file_path, model_name, max_code_length):
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
                # Save SVGs to imgs directory
                saved_files = save_multiple_svgs(svg_contents, file_path)
                
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

def process_batch(batch_size, model_name, throttle_delay=5, max_code_length=3000):
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
        result = process_file(file_path, model_name, max_code_length)
        
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
                                     skip_existing=True, max_code_length=3000):
    """Process all files in a directory with throttling and batching."""
    # Find eligible files and initialize state
    state = find_eligible_files(root_dir, skip_existing)
    
    if not state["pending"]:
        logger.info("No files to process.")
        return
    
    logger.info(f"Starting batch processing with throttle delay of {throttle_delay}s")
    
    # Process in batches
    total_processed = 0
    while state["pending"]:
        logger.info(f"Processing batch of up to {batch_size} files")
        files_processed = process_batch(batch_size, model_name, throttle_delay, max_code_length)
        total_processed += files_processed
        
        if files_processed < batch_size:
            break
            
        # Reload state
        state = load_state()
    
    logger.info(f"Batch processing completed. Total files processed: {total_processed}")
    logger.info(f"Status: {len(state['pending'])} pending, {len(state['completed'])} completed, {len(state['failed'])} failed")

def main():
    parser = argparse.ArgumentParser(description='Generate multiple SVG diagrams for code files using Claude API')
    parser.add_argument('--dir', type=str, default=".", help='Directory to process')
    parser.add_argument('--model', type=str, default="claude-3-sonnet-20240229", 
                        help='Claude model to use (default: claude-3-sonnet-20240229)')
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
    
    # Add help on available models
    parser.add_argument('--list-models', action='store_true',
                        help='List available Claude models names and exit')
    
    args = parser.parse_args()
    
    # Show available models if requested
    if args.list_models:
        print("Available Claude models:")
        print("  claude-3-opus-20240229    (high quality, slower)")
        print("  claude-3-sonnet-20240229  (balanced performance)")
        print("  claude-3-haiku-20240307   (fastest)")
        print("  claude-3-7-sonnet-20250219 (newest model)")
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
            max_code_length=args.max_code_length
        )
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting gracefully.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()