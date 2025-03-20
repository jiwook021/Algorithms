import os
import LLMscript.svgscriptgpt as svgscriptgpt
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not api_key:
    raise ValueError("Missing OpenAI API key. Add it to a .env file.")

# Create OpenAI client
client = svgscriptgpt.OpenAI(api_key=api_key)

# File types to process
SUPPORTED_EXTENSIONS = ['.cpp', '.c', '.py']

def create_prompt(file_content, filename):
    """ Generate an OpenAI prompt to create an SVG diagram explaining the code. """
    language = "C++" if filename.endswith('.cpp') else "C" if filename.endswith('.c') else "Python"
    
    prompt = f"""
    Generate SVG images explaining deeply the logic and functionality of the following {language} code.
    Make many images as possible. but no text other than svg files
    Filename: {filename}
    Generate Nothing other than tag from <svg> to </svg>
    Code:
    ```
    {file_content[:10000]}  # Limit input size for API efficiency
    ```
    """
    return prompt.strip()

import time

def request_svg_diagram(prompt, retries=5, delay=5):
    """ Request an SVG diagram from OpenAI's API with retry logic. """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except svgscriptgpt.APIConnectionError as e:
            print(f"Connection error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except svgscriptgpt.RateLimitError:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(delay * 2)
        except Exception as e:
            print(f"Unexpected API Error: {e}")
            break
    print("Failed to retrieve SVG after multiple attempts.")
    return None


def save_svg(svg_content, output_filepath):
    """ Save the generated SVG content to a file. """
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)

def process_directory(root_dir):
    """ Recursively process files in the given directory. """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(subdir, file)
                svg_filename = os.path.splitext(file)[0] + "_explained.svg"
                svg_filepath = os.path.join(subdir, svg_filename)

                print(f"Processing: {file_path}")

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                    file_content = source_file.read()

                prompt = create_prompt(file_content, file)
                svg_content = request_svg_diagram(prompt)

                if svg_content:
                    save_svg(svg_content, svg_filepath)
                    print(f"SVG saved: {svg_filepath}")
                else:
                    print(f"Failed to generate SVG for {file_path}")

if __name__ == "__main__":
    directory_path = "."  # Change this to your target folder if needed
    print(f"Starting SVG generation in: {os.path.abspath(directory_path)}")
    process_directory(directory_path)
    print("Script completed.")
