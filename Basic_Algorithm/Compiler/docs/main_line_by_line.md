# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, even if you’re a beginner.

---

### **1. Token Definitions**
#### **What It Does**
This section defines the types of tokens the lexer can recognize. Tokens are the smallest meaningful units in a programming language, like keywords, operators, or punctuation.

#### **Code Breakdown**
```c++
enum class TokenType {
    // Single-character tokens
    LeftParen, RightParen, LeftBrace, RightBrace,
    Comma, Semicolon, Colon,
    // One or two character tokens
    Plus, Minus, Star, Slash, Percent,
    Equal, EqualEqual, Bang, BangEqual,
    Less, LessEqual, Greater, GreaterEqual,
    // Literals
    Identifier, Number, String, Char,
    // Keywords
    If, Else, For, While, Return,
    Class, Struct, Static, Const,
    // End-of-file
    EndOfFile,
    // Error
    Error
};
```

#### **Explanation**
- **`enum class TokenType`**: This defines a list of possible token types. Each type represents a category of tokens, such as:
  - **Single-character tokens**: `(`, `)`, `{`, `}`, `,`, `;`, `:`.
  - **Multi-character tokens**: `==`, `!=`, `<=`, `>=`.
  - **Literals**: Identifiers (e.g., variable names), numbers, strings, and characters.
  - **Keywords**: Reserved words like `if`, `else`, `return`.
  - **Special tokens**: `EndOfFile` (marks the end of the input) and `Error` (for invalid input).

#### **Why This Is Useful**
- Tokens are the building blocks of a programming language. By defining them explicitly, the lexer can categorize and process each part of the source code correctly.

---

### **2. Token Structure**
#### **What It Does**
This defines the structure of a token, which includes:
- Its type (e.g., `LeftParen`, `Identifier`).
- The actual text (lexeme) of the token (e.g., `"if"`, `"=="`).
- Its position in the source code (line and column numbers).

#### **Code Breakdown**
```c++
struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
};
```

#### **Explanation**
- **`TokenType type`**: The category of the token (e.g., `If`, `EqualEqual`).
- **`std::string lexeme`**: The actual text of the token (e.g., `"if"`, `"=="`).
- **`int line` and `int column`**: The position of the token in the source code. This is useful for error reporting (e.g., "Error at line 5, column 10").

#### **Why This Is Useful**
- Storing the token’s type, text, and position allows the lexer to provide meaningful information to the next stages of compilation (e.g., parsing) and to report errors accurately.

---

### **3. Keyword Detection**
#### **What It Does**
This function checks if a given string is a reserved keyword (e.g., `if`, `else`, `return`) and returns the corresponding `TokenType`.

#### **Code Breakdown**
```c++
TokenType checkKeyword(const std::string &text) {
    if (text == "class")  return TokenType::Class;
    if (text == "struct") return TokenType::Struct;
    if (text == "static") return TokenType::Static;
    if (text == "const")  return TokenType::Const;
    if (text == "return") return TokenType::Return;
    if (text == "if")     return TokenType::If;
    if (text == "else")   return TokenType::Else;
    if (text == "for")    return TokenType::For;
    if (text == "while")  return TokenType::While;
    return TokenType::Identifier;
}
```

#### **Explanation**
- The function takes a string (`text`) as input and compares it to a list of reserved keywords.
- If the string matches a keyword, it returns the corresponding `TokenType` (e.g., `TokenType::If` for `"if"`).
- If the string doesn’t match any keyword, it returns `TokenType::Identifier`, indicating that the string is a variable name or function name.

#### **Why This Is Useful**
- Keywords are special words in a programming language that have specific meanings (e.g., `if` starts a conditional statement). By detecting them, the lexer ensures they are treated differently from regular identifiers.

---

### **4. Lexer Class**
#### **What It Does**
The `Lexer` class is the core of the program. It processes the source code string and generates tokens.

#### **Code Breakdown**
```c++
class Lexer {
public:
    Lexer(const std::string &source)
        : source(source), index(0), start(0), line(1), column(1) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (!isAtEnd()) {
            skipWhitespaceAndComments();
            if (isAtEnd()) break;
            start = index;
            Token t = scanToken();
            tokens.push_back(t);
        }
        tokens.push_back({TokenType::EndOfFile, "", line, column});
        return tokens;
    }
private:
    std::string source;
    size_t index, start;
    int line, column;

    bool isAtEnd() const {
        return index >= source.size();
    }
    char peek() const {
        return isAtEnd() ? '\0' : source[index];
    }
    char peekNext() const {
        return (index+1 < source.size()) ? source[index+1] : '\0';
    }
    char advance() {
        char c = source[index++];
        if (c == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
        return c;
    }
    bool match(char expected) {
        if (isAtEnd() || source[index] != expected) return false;
        advance();
        return true;
    }
};
```

#### **Explanation**
- **`Lexer` Constructor**:
  - Initializes the lexer with the source code string.
  - Sets `index`, `start`, `line`, and `column` to track the current position in the source code.

- **`tokenize()` Method**:
  - This is the main method that drives the tokenization process.
  - It creates a list of tokens and processes the source code until the end is reached.
  - It skips whitespace and comments, then scans the next token and adds it to the list.

- **Helper Methods**:
  - **`isAtEnd()`**: Checks if the lexer has reached the end of the source code.
  - **`peek()`**: Returns the current character without consuming it.
  - **`peekNext()`**: Returns the next character without consuming it.
  - **`advance()`**: Moves to the next character and updates the line and column numbers.
  - **`match()`**: Checks if the current character matches an expected character and consumes it if it does.

#### **Why This Is Useful**
- The `Lexer` class encapsulates all the logic for tokenizing the source code. It uses helper methods to navigate the input and generate tokens efficiently.

---

### **5. Skipping Whitespace and Comments**
#### **What It Does**
This method skips over whitespace (spaces, tabs, newlines) and comments (`//` and `/* ... */`) in the source code.

#### **Code Breakdown**
```c++
void skipWhitespaceAndComments() {
    while (!isAtEnd()) {
        char c = peek();
        if (isspace(c)) {
            advance();
        }
        else if (c == '/') {
            // "//" 주석
            if (peekNext() == '/') {
                while (!isAtEnd() && peek() != '\n') {
                    advance();
                }
            }
            // "/* ... */"
            else if (peekNext() == '*') {
                advance(); // '/'
                advance(); // '*'
                while (!isAtEnd() && !(peek() == '*' && peekNext() == '/')) {
                    advance();
                }
                if (!isAtEnd()) {
                    advance(); // '*'
                    advance(); // '/'
                }
            }
            else break;
        }
        else {
            break;
        }
    }
}
```

#### **Explanation**
- The method processes the source code character by character:
  - If the character is whitespace, it skips it.
  - If the character is `/`, it checks for comments:
    - `//`: Skips everything until the end of the line.
    - `/* ... */`: Skips everything until the closing `*/`.

#### **Why This Is Useful**
- Whitespace and comments are not meaningful for tokenization. Skipping them ensures the lexer focuses on the actual code.

---

### **6. Debugging Utilities**
#### **What It Does**
This section provides utilities for debugging, such as converting `TokenType` to a human-readable string and printing debug information about the tokens.

#### **Code Breakdown**
```c++
static const char* tokenTypeName(TokenType t) {
    switch(t) {
        case TokenType::LeftParen: return "LeftParen";
        case TokenType::RightParen: return "RightParen";
        // ... (other cases)
    }
    return "Unknown";
}
```

#### **Explanation**
- The `tokenTypeName()` function converts a `TokenType` to a string for debugging purposes.
- For example, `TokenType::If` becomes `"If"`.

#### **Why This Is Useful**
- Debugging utilities help developers understand the lexer’s behavior and diagnose issues.

---

### **Summary**
This code is a **lexical analyzer** that:
1. Defines token types and structures.
2. Processes source code to generate tokens.
3. Handles whitespace, comments, and keywords.
4. Provides debugging utilities for developers.

It serves as the foundation for further stages of compilation or interpretation, such as parsing and semantic analysis.