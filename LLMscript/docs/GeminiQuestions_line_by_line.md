# Step-by-Step Explanation: GeminiQuestions.py

Let's break down the code step-by-step, explaining every significant section in detail. I'll use simple language, examples, and diagrams where helpful.

---

### **1. Imports and Setup**
```python
import os
import re
import google.generativeai as genai
import argparse
import time
from pathlib import Path
```

#### **What It Does**
These lines import external libraries that the script needs to function. Think of libraries as toolkits that provide pre-built functionality.

#### **Breakdown**
- **`os`**: Provides tools for interacting with the operating system, like reading environment variables.
- **`re`**: Stands for "regular expressions," a powerful tool for searching and manipulating text.
- **`google.generativeai`**: The library for interacting with Google's Gemini AI models.
- **`argparse`**: Helps parse command-line arguments (e.g., input file path, output directory).
- **`time`**: Used for adding delays (e.g., to avoid overloading the API).
- **`pathlib.Path`**: A modern way to handle file paths in Python.

#### **Why These Are Used**
- **`os`**: Needed to read the API key from the environment.
- **`re`**: Used to extract questions from the input file and sanitize filenames.
- **`google.generativeai`**: Required to interact with the Gemini AI.
- **`argparse`**: Makes the script user-friendly by allowing command-line customization.
- **`time`**: Prevents hitting API rate limits.
- **`pathlib.Path`**: Simplifies file path handling compared to older methods.

---

### **2. `setup_gemini_model` Function**
```python
def setup_gemini_model(api_key, model_name="gemini-2.0-flash"):
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error setting up model '{model_name}': {e}")
        print("Available models:")
        for model in genai.list_models():
            print(f"- {model.name}")
        raise
```

#### **What It Does**
This function sets up the Gemini AI model using the provided API key and model name.

#### **Breakdown**
1. **`genai.configure(api_key=api_key)`**: Configures the Gemini API with the user's API key.
2. **`try` Block**: Attempts to create a `GenerativeModel` object using the specified model name.
3. **`except` Block**: If something goes wrong (e.g., invalid model name), it prints an error message and lists all available models.

#### **Why This Approach**
- **Error Handling**: Prevents the script from crashing if the model setup fails.
- **User Feedback**: Lists available models to help the user choose a valid one.

#### **Example**
If the user provides an invalid model name like `gemini-3.0`, the script will:
1. Print an error message.
2. List all available models (e.g., `gemini-2.0-flash`, `gemini-2.0-pro`).
3. Stop execution to avoid further errors.

---

### **3. `extract_questions` Function**
```python
def extract_questions(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Successfully read file: {file_path}")
        print(f"File content sample: {content[:200]}...")
        
        questions = re.findall(r'^\s*[\*\-•]\s*(.+)', content, re.MULTILINE)
        
        print(f"Extracted {len(questions)} questions from the file")
        if questions:
            print(f"First question sample: {questions[0]}")
            
        return questions
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
```

#### **What It Does**
This function reads a text file and extracts questions that are formatted as bullet points.

#### **Breakdown**
1. **File Reading**:
   - Opens the file in read mode (`'r'`) with UTF-8 encoding.
   - Reads the entire content into the `content` variable.
2. **Regex Extraction**:
   - Uses `re.findall` to find all lines that start with a bullet point (`*`, `-`, or `•`).
   - The regex pattern `^\s*[\*\-•]\s*(.+)` breaks down as:
     - `^`: Start of a line.
     - `\s*`: Any number of spaces.
     - `[\*\-•]`: A bullet point character (`*`, `-`, or `•`).
     - `\s*`: Any number of spaces after the bullet.
     - `(.+)`: Captures the rest of the line as the question.
3. **Error Handling**:
   - If the file cannot be read, it prints an error and returns an empty list.

#### **Why This Approach**
- **Regex**: Efficiently extracts questions based on a specific pattern.
- **Error Handling**: Ensures the script doesn't crash if the file is missing or corrupted.

#### **Example**
If the input file contains:
```
* What is Python?
- How does AI work?
• Explain machine learning.
```
The function will extract:
```python
["What is Python?", "How does AI work?", "Explain machine learning."]
```

---

### **4. `sanitize_filename` Function**
```python
def sanitize_filename(filename):
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'\s+', '-', sanitized)
    return sanitized.lower()
```

#### **What It Does**
This function cleans up a filename by removing special characters and replacing spaces with hyphens.

#### **Breakdown**
1. **First `re.sub`**:
   - Removes any character that is not a word character (`\w`), space (`\s`), or hyphen (`-`).
   - `[^\w\s-]` means "any character not in this set."
2. **Second `re.sub`**:
   - Replaces one or more spaces (`\s+`) with a single hyphen (`-`).
3. **Lowercase Conversion**:
   - Converts the filename to lowercase for consistency.

#### **Why This Approach**
- **Safety**: Ensures filenames are valid and won't cause issues on any operating system.
- **Readability**: Makes filenames more consistent and easier to manage.

#### **Example**
If the input is `"What is AI?.txt"`, the output will be `"what-is-ai.txt"`.

---

### **5. `main` Function**
```python
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
    
    questions = extract_questions(args.input_file)
    if not questions:
        print("No questions found in the input file.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Processing: {question[:50]}...")
        try:
            prompt = "Explain the answer to this question in simple terms, step by step, so that someone without prior knowledge can understand: " + question
            response = model.generate_content(prompt)
            sanitized_filename = sanitize_filename(question[:50])
            output_file = output_dir / f"{sanitized_filename}.md"
            with open(output_file, "w") as f:
                f.write(response.text)
            print(f"Created: {output_file}")
        except Exception as e:
            print(f"Error getting answer for question: {question}")
            print(f"Error details: {e}")
        time.sleep(0.5)
```

#### **What It Does**
This is the main function that orchestrates the entire process:
1. Parses command-line arguments.
2. Sets up the Gemini model.
3. Extracts questions from the input file.
4. Processes each question with the AI and saves the responses.

#### **Breakdown**
1. **Argument Parsing**:
   - `input_file`: Path to the file containing questions.
   - `--output_dir`: Directory to save answers (default: `output`).
   - `--model`: Gemini model to use (default: `gemini-2.0-flash`).
2. **API Key Check**:
   - Retrieves the API key from the environment.
   - Raises an error if the key is missing.
3. **Model Setup**:
   - Calls `setup_gemini_model` to initialize the AI model.
4. **Question Extraction**:
   - Calls `extract_questions` to read questions from the file.
5. **Output Directory Setup**:
   - Creates the output directory if it doesn't exist.
6. **Question Processing**:
   - Loops through each question.
   - Constructs a prompt for the AI.
   - Saves the AI's response as a markdown file.
   - Adds a delay (`time.sleep(0.5)`) to avoid hitting API rate limits.

#### **Why This Approach**
- **Modularity**: Each step is handled by a separate function, making the code easy to maintain.
- **User-Friendly**: Command-line arguments make the script customizable.
- **Robustness**: Error handling ensures the script doesn't crash unexpectedly.

#### **Example**
If the input file contains:
```
* What is Python?
- How does AI work?
```
The script will:
1. Create a directory called `output`.
2. Generate two markdown files:
   - `what-is-python.md`
   - `how-does-ai-work.md`
3. Each file will contain the AI's step-by-step explanation of the question.

---

### **6. Execution Block**
```python
if __name__ == "__main__":
    main()
```

#### **What It Does**
This block ensures the `main` function runs only when the script is executed directly (not when imported as a module).

#### **Why This Approach**
- **Modularity**: Allows the script to be reused in other programs without running the `main` function automatically.
- **Best Practice**: A common Python idiom for script organization.

---

### **Summary**
This script is a well-structured, modular tool for automating question answering using AI. It:
1. Reads questions from a file.
2. Uses Gemini AI to generate explanations.
3. Saves the explanations as markdown files.

Each function has a clear purpose, and the code is designed to be robust, user-friendly, and easy to extend.