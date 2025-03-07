#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

//===========================================================
// 1. Lexical Analysis (키워드 처리 포함)
//===========================================================

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

// (디버그용) TokenType 이름
static const char* tokenTypeName(TokenType t) {
    switch(t) {
        case TokenType::LeftParen: return "LeftParen";
        case TokenType::RightParen: return "RightParen";
        case TokenType::LeftBrace: return "LeftBrace";
        case TokenType::RightBrace: return "RightBrace";
        case TokenType::Comma: return "Comma";
        case TokenType::Semicolon: return "Semicolon";
        case TokenType::Colon: return "Colon";
        case TokenType::Plus: return "Plus";
        case TokenType::Minus: return "Minus";
        case TokenType::Star: return "Star";
        case TokenType::Slash: return "Slash";
        case TokenType::Percent: return "Percent";
        case TokenType::Equal: return "Equal";
        case TokenType::EqualEqual: return "EqualEqual";
        case TokenType::Bang: return "Bang";
        case TokenType::BangEqual: return "BangEqual";
        case TokenType::Less: return "Less";
        case TokenType::LessEqual: return "LessEqual";
        case TokenType::Greater: return "Greater";
        case TokenType::GreaterEqual: return "GreaterEqual";
        case TokenType::Identifier: return "Identifier";
        case TokenType::Number: return "Number";
        case TokenType::String: return "String";
        case TokenType::Char: return "Char";
        case TokenType::If: return "If";
        case TokenType::Else: return "Else";
        case TokenType::For: return "For";
        case TokenType::While: return "While";
        case TokenType::Return: return "Return";
        case TokenType::Class: return "Class";
        case TokenType::Struct: return "Struct";
        case TokenType::Static: return "Static";
        case TokenType::Const: return "Const";
        case TokenType::EndOfFile: return "EndOfFile";
        case TokenType::Error: return "Error";
    }
    return "Unknown";
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;
};

// "class", "static", "const", "return" 등 키워드 판별
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

        // [DEBUG] 생성된 토큰 출력
        std::cerr << "[DEBUG][Lexer] Produced tokens:\n";
        for (auto &tk : tokens) {
            std::cerr << " line " << tk.line << " col " << tk.column
                      << " type=" << tokenTypeName(tk.type)
                      << " lexeme=\"" << tk.lexeme << "\"\n";
        }

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

    Token makeToken(TokenType t) {
        std::string text = source.substr(start, index - start);
        return { t, text, line, column - (int)(index - start) };
    }

    Token errorToken(const std::string &msg) {
        std::cerr << "Lexical Error [" << line << ":" << column << "]: " << msg << "\n";
        return { TokenType::Error, msg, line, column };
    }

    Token scanToken() {
        char c = advance();
        if (isalpha(c) || c == '_') {
            return identifierOrKeyword();
        }
        if (isdigit(c)) {
            return number();
        }

        switch (c) {
        case '(': return makeToken(TokenType::LeftParen);
        case ')': return makeToken(TokenType::RightParen);
        case '{': return makeToken(TokenType::LeftBrace);
        case '}': return makeToken(TokenType::RightBrace);
        case ',': return makeToken(TokenType::Comma);
        case ';': return makeToken(TokenType::Semicolon);
        case ':': return makeToken(TokenType::Colon);
        case '+': return makeToken(TokenType::Plus);
        case '-': return makeToken(TokenType::Minus);
        case '*': return makeToken(TokenType::Star);
        case '/': return makeToken(TokenType::Slash);
        case '%': return makeToken(TokenType::Percent);
        case '=':
            if (match('=')) return makeToken(TokenType::EqualEqual);
            return makeToken(TokenType::Equal);
        case '!':
            if (match('=')) return makeToken(TokenType::BangEqual);
            return makeToken(TokenType::Bang);
        case '<':
            if (match('=')) return makeToken(TokenType::LessEqual);
            return makeToken(TokenType::Less);
        case '>':
            if (match('=')) return makeToken(TokenType::GreaterEqual);
            return makeToken(TokenType::Greater);
        default:
            return errorToken(std::string("Unexpected character '") + c + "'");
        }
    }

    Token identifierOrKeyword() {
        while (!isAtEnd() && (isalnum(peek()) || peek() == '_')) {
            advance();
        }
        std::string text = source.substr(start, index - start);
        TokenType type = checkKeyword(text);
        return { type, text, line, column - (int)(index - start) };
    }

    Token number() {
        while (!isAtEnd() && isdigit(peek())) {
            advance();
        }
        return makeToken(TokenType::Number);
    }
};

//===========================================================
// 2. AST Node & Visitor
//===========================================================

class ASTVisitor;
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor &visitor) = 0;
};

class Expression : public ASTNode {};
class Statement  : public ASTNode {};

class LiteralExpression : public Expression {
public:
    std::string value;
    LiteralExpression(const std::string &v) : value(v) {}
    void accept(ASTVisitor &visitor) override;
};

class IdentifierExpression : public Expression {
public:
    std::string name;
    IdentifierExpression(const std::string &n) : name(n) {}
    void accept(ASTVisitor &visitor) override;
};

class BinaryExpression : public Expression {
public:
    std::string op;
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    BinaryExpression(const std::string &op,
                     std::unique_ptr<Expression> left,
                     std::unique_ptr<Expression> right)
      : op(op), left(std::move(left)), right(std::move(right)) {}
    void accept(ASTVisitor &visitor) override;
};

class UnaryExpression : public Expression {
public:
    std::string op;
    std::unique_ptr<Expression> operand;
    UnaryExpression(const std::string &op, std::unique_ptr<Expression> operand)
      : op(op), operand(std::move(operand)) {}
    void accept(ASTVisitor &visitor) override;
};

class BlockStatement : public Statement {
public:
    std::vector<std::unique_ptr<Statement>> statements;
    void accept(ASTVisitor &visitor) override;
};

class ExpressionStatement : public Statement {
public:
    std::unique_ptr<Expression> expression;
    ExpressionStatement(std::unique_ptr<Expression> expr)
      : expression(std::move(expr)) {}
    void accept(ASTVisitor &visitor) override;
};

class IfStatement : public Statement {
public:
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> thenBranch;
    std::unique_ptr<Statement> elseBranch;
    IfStatement(std::unique_ptr<Expression> cond,
                std::unique_ptr<Statement> thenBr,
                std::unique_ptr<Statement> elseBr)
      : condition(std::move(cond)), thenBranch(std::move(thenBr)), elseBranch(std::move(elseBr)) {}
    void accept(ASTVisitor &visitor) override;
};

class WhileStatement : public Statement {
public:
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Statement> body;
    WhileStatement(std::unique_ptr<Expression> cond,
                   std::unique_ptr<Statement> body)
      : condition(std::move(cond)), body(std::move(body)) {}
    void accept(ASTVisitor &visitor) override;
};

class ForStatement : public Statement {
public:
    std::unique_ptr<Statement> initializer;
    std::unique_ptr<Expression> condition;
    std::unique_ptr<Expression> increment;
    std::unique_ptr<Statement> body;
    ForStatement(std::unique_ptr<Statement> init,
                 std::unique_ptr<Expression> cond,
                 std::unique_ptr<Expression> incr,
                 std::unique_ptr<Statement> body)
      : initializer(std::move(init)), condition(std::move(cond)),
        increment(std::move(incr)), body(std::move(body)) {}
    void accept(ASTVisitor &visitor) override;
};

class ReturnStatement : public Statement {
public:
    std::unique_ptr<Expression> expression;
    ReturnStatement(std::unique_ptr<Expression> expr)
      : expression(std::move(expr)) {}
    void accept(ASTVisitor &visitor) override;
};

class VariableDeclaration : public Statement {
public:
    std::string name;
    std::string type;
    bool isStatic = false;
    bool isConst  = false;
    std::unique_ptr<Expression> initializer;
    VariableDeclaration(const std::string &n, const std::string &t)
      : name(n), type(t) {}
    void accept(ASTVisitor &visitor) override;
};

class FunctionDeclaration : public Statement {
public:
    std::string name;
    std::string returnType;
    std::vector<std::pair<std::string, std::string>> parameters;
    bool isStatic = false;
    bool isConst  = false;
    std::unique_ptr<BlockStatement> body;
    FunctionDeclaration(const std::string &n, const std::string &r)
      : name(n), returnType(r) {}
    void accept(ASTVisitor &visitor) override;
};

class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    virtual void visit(LiteralExpression* node) = 0;
    virtual void visit(IdentifierExpression* node) = 0;
    virtual void visit(BinaryExpression* node) = 0;
    virtual void visit(UnaryExpression* node) = 0;
    virtual void visit(BlockStatement* node) = 0;
    virtual void visit(ExpressionStatement* node) = 0;
    virtual void visit(IfStatement* node) = 0;
    virtual void visit(WhileStatement* node) = 0;
    virtual void visit(ForStatement* node) = 0;
    virtual void visit(ReturnStatement* node) = 0;
    virtual void visit(VariableDeclaration* node) = 0;
    virtual void visit(FunctionDeclaration* node) = 0;
};

void LiteralExpression::accept(ASTVisitor &v) { v.visit(this); }
void IdentifierExpression::accept(ASTVisitor &v) { v.visit(this); }
void BinaryExpression::accept(ASTVisitor &v) { v.visit(this); }
void UnaryExpression::accept(ASTVisitor &v) { v.visit(this); }
void BlockStatement::accept(ASTVisitor &v) { v.visit(this); }
void ExpressionStatement::accept(ASTVisitor &v) { v.visit(this); }
void IfStatement::accept(ASTVisitor &v) { v.visit(this); }
void WhileStatement::accept(ASTVisitor &v) { v.visit(this); }
void ForStatement::accept(ASTVisitor &v) { v.visit(this); }
void ReturnStatement::accept(ASTVisitor &v) { v.visit(this); }
void VariableDeclaration::accept(ASTVisitor &v) { v.visit(this); }
void FunctionDeclaration::accept(ASTVisitor &v) { v.visit(this); }

//===========================================================
// 3. Parser (parseExpression() 포함)
//===========================================================

class Parser {
public:
    Parser(const std::vector<Token> &tokens) : tokens(tokens), current(0) {}

    // 프로그램 전체를 파싱: 여러 Declaration을 block에 담는다
    std::unique_ptr<Statement> parse() {
        auto block = std::make_unique<BlockStatement>();
        while (!isAtEnd()) {
            try {
                block->statements.push_back(parseDeclaration());
            } catch (std::runtime_error &err) {
                synchronize();
            }
        }
        return block;
    }

private:
    const std::vector<Token>& tokens;
    size_t current;

    //=== 유틸리티 함수들 ===
    bool isAtEnd() const {
        return peek().type == TokenType::EndOfFile;
    }
    const Token& peek() const {
        return tokens[current];
    }
    const Token& previous() const {
        return tokens[current-1];
    }
    bool check(TokenType t) const {
        if (isAtEnd()) return false;
        return peek().type == t;
    }
    bool match(std::initializer_list<TokenType> types) {
        for (auto tt : types) {
            if (check(tt)) {
                advance();
                return true;
            }
        }
        return false;
    }
    const Token& advance() {
        if (!isAtEnd()) current++;
        return previous();
    }
    const Token& consume(TokenType t, const std::string &msg) {
        if (check(t)) return advance();
        throw error(peek(), msg);
    }
    std::runtime_error error(const Token &tk, const std::string &msg) {
        std::cerr << "[Parser Error] [Line " << tk.line << ", Col " << tk.column << "] " << msg << "\n";
        return std::runtime_error(msg);
    }
    void synchronize() {
        advance();
        while (!isAtEnd()) {
            if (previous().type == TokenType::Semicolon) return;
            switch(peek().type) {
            case TokenType::Class:
            case TokenType::Struct:
            case TokenType::If:
            case TokenType::For:
            case TokenType::While:
            case TokenType::Return:
                return;
            default:
                break;
            }
            advance();
        }
    }

    //=== 선언(Declaration) ===
    std::unique_ptr<Statement> parseDeclaration() {
        skipAccessSpecifier(); // e.g. public:
        if (match({TokenType::Class})) {
            return parseClassDeclaration();
        }
        if (match({TokenType::Struct})) {
            return parseStructDeclaration();
        }
        return parseFunctionOrVariableDeclaration();
    }
    bool skipAccessSpecifier() {
        if (check(TokenType::Identifier)) {
            std::string kw = peek().lexeme;
            if ((kw=="public"||kw=="private"||kw=="protected") &&
                current+1<tokens.size() && tokens[current+1].type==TokenType::Colon)
            {
                advance(); // skip identifier
                advance(); // skip colon
                return true;
            }
        }
        return false;
    }

    // class Foo : public Base { ... }
    std::unique_ptr<Statement> parseClassDeclaration() {
        Token name = consume(TokenType::Identifier, "Expected class name.");
        // 상속 구문(: public Base) 등 스킵
        if (match({TokenType::Colon})) {
            if (check(TokenType::Identifier)) advance(); // e.g. "public"
            if (check(TokenType::Identifier)) advance(); // e.g. "Base"
        }
        consume(TokenType::LeftBrace, "Expected '{' after class name.");
        // 여기서는 class 자체를 간단히 FunctionDeclaration 으로 대충 저장
        auto classFake = std::make_unique<FunctionDeclaration>(name.lexeme, "class");
        while (!check(TokenType::RightBrace) && !isAtEnd()) {
            // 멤버
            classFake->body = parseBlockStatementInsideClass();
        }
        consume(TokenType::RightBrace, "Expected '}' after class body.");
        if (check(TokenType::Semicolon)) advance();
        return classFake;
    }
    // struct Foo { ... }
    std::unique_ptr<Statement> parseStructDeclaration() {
        Token name = consume(TokenType::Identifier, "Expected struct name.");
        consume(TokenType::LeftBrace, "Expected '{' after struct name.");
        auto structFake = std::make_unique<FunctionDeclaration>(name.lexeme, "struct");
        while (!check(TokenType::RightBrace) && !isAtEnd()) {
            structFake->body = parseBlockStatementInsideClass();
        }
        consume(TokenType::RightBrace, "Expected '}' after struct body.");
        if (check(TokenType::Semicolon)) advance();
        return structFake;
    }
    // 클래스 내부에서 멤버 하나만 parseDeclaration() -> block에 저장
    std::unique_ptr<BlockStatement> parseBlockStatementInsideClass() {
        auto blk = std::make_unique<BlockStatement>();
        blk->statements.push_back(parseDeclaration());
        return blk;
    }

    // 전역 레벨 함수 / 변수 구분
    std::unique_ptr<Statement> parseFunctionOrVariableDeclaration() {
        bool isStatic = false;
        bool isConst  = false;
        while (match({TokenType::Static, TokenType::Const})) {
            if (previous().type == TokenType::Static) isStatic = true;
            else if (previous().type == TokenType::Const) isConst = true;
        }

        Token typeToken = consume(TokenType::Identifier, "Expected type in declaration.");
        std::string typeStr = typeToken.lexeme;

        Token nameToken = consume(TokenType::Identifier, "Expected name in declaration.");
        std::string nameStr = nameToken.lexeme;

        // 함수?
        if (check(TokenType::LeftParen)) {
            return parseFunctionDeclaration(typeStr, nameStr, isStatic, isConst);
        }
        else {
            // 변수
            return parseVariableDeclaration(typeStr, nameStr, isStatic, isConst);
        }
    }
    std::unique_ptr<Statement> parseVariableDeclaration(const std::string &typeStr,
                                                        const std::string &nameStr,
                                                        bool isStatic, bool isConst) {
        auto varDecl = std::make_unique<VariableDeclaration>(nameStr, typeStr);
        varDecl->isStatic = isStatic;
        varDecl->isConst  = isConst;

        if (match({TokenType::Equal})) {
            // **여기서 parseExpression() 호출**
            varDecl->initializer = parseExpression();
        }
        consume(TokenType::Semicolon, "Expected ';' after variable declaration.");
        return varDecl;
    }
    std::unique_ptr<Statement> parseFunctionDeclaration(const std::string &retType,
                                                        const std::string &nameStr,
                                                        bool isStatic, bool isConst) {
        auto func = std::make_unique<FunctionDeclaration>(nameStr, retType);
        func->isStatic = isStatic;
        func->isConst  = isConst;

        consume(TokenType::LeftParen, "Expected '(' after function name.");
        if (!check(TokenType::RightParen)) {
            // 간단 파라미터
            do {
                Token pType = consume(TokenType::Identifier, "Expected parameter type.");
                Token pName = consume(TokenType::Identifier, "Expected parameter name.");
                func->parameters.push_back({pType.lexeme, pName.lexeme});
            } while (match({TokenType::Comma}));
        }
        consume(TokenType::RightParen, "Expected ')' after parameters.");

        // 함수 본문
        func->body = parseBlockStatement();
        return func;
    }

    //=== parseBlockStatement() : 중괄호 { } 블록 ===
    std::unique_ptr<BlockStatement> parseBlockStatement() {
        auto block = std::make_unique<BlockStatement>();
        consume(TokenType::LeftBrace, "Expected '{' to start block.");
        while (!check(TokenType::RightBrace) && !isAtEnd()) {
            // 블록 내부에는 "선언" or "문장" 가능
            if (isDeclarationStart()) {
                auto decl = parseFunctionOrVariableDeclaration();
                block->statements.push_back(std::move(decl));
            } else {
                auto stmt = parseStatement();
                block->statements.push_back(std::move(stmt));
            }
        }
        consume(TokenType::RightBrace, "Expected '}' after block.");
        return block;
    }
    // “선언” 시작 판단 (static|const|Identifier Identifier)
    bool isDeclarationStart() {
        if (check(TokenType::Static) || check(TokenType::Const)) return true;
        if (check(TokenType::Identifier)) {
            // 미리보기
            if (current+1<tokens.size() && tokens[current+1].type==TokenType::Identifier) {
                return true;
            }
        }
        return false;
    }

    //=== parseStatement() : if, while, for, return, 블록, or expression ===
    std::unique_ptr<Statement> parseStatement() {
        if (match({TokenType::Return})) {
            return parseReturnStatement();
        }
        if (match({TokenType::If})) {
            return parseIfStatement();
        }
        if (match({TokenType::While})) {
            return parseWhileStatement();
        }
        if (match({TokenType::For})) {
            return parseForStatement();
        }
        if (match({TokenType::LeftBrace})) {
            return parseBlockStatementNoConsume();
        }
        // 그 외 => expression statement
        return parseExpressionStatement();
    }
    std::unique_ptr<Statement> parseReturnStatement() {
        // "return"
        if (!check(TokenType::Semicolon)) {
            // 예: return value;
            auto expr = parseExpression();
            consume(TokenType::Semicolon, "Expected ';' after return value.");
            return std::make_unique<ReturnStatement>(std::move(expr));
        } else {
            // return;
            consume(TokenType::Semicolon, "Expected ';' after return.");
            return std::make_unique<ReturnStatement>(nullptr);
        }
    }
    std::unique_ptr<Statement> parseIfStatement() {
        consume(TokenType::LeftParen, "Expected '(' after if.");
        auto cond = parseExpression();
        consume(TokenType::RightParen, "Expected ')' after if condition.");
        auto thenStmt = parseStatement();
        std::unique_ptr<Statement> elseStmt;
        if (match({TokenType::Else})) {
            elseStmt = parseStatement();
        }
        return std::make_unique<IfStatement>(std::move(cond), std::move(thenStmt), std::move(elseStmt));
    }
    std::unique_ptr<Statement> parseWhileStatement() {
        consume(TokenType::LeftParen, "Expected '(' after while.");
        auto cond = parseExpression();
        consume(TokenType::RightParen, "Expected ')' after while condition.");
        auto body = parseStatement();
        return std::make_unique<WhileStatement>(std::move(cond), std::move(body));
    }
    std::unique_ptr<Statement> parseForStatement() {
        consume(TokenType::LeftParen, "Expected '(' after for.");

        std::unique_ptr<Statement> init;
        if (!check(TokenType::Semicolon)) {
            if (isDeclarationStart()) {
                init = parseFunctionOrVariableDeclaration();
            } else {
                init = parseExpressionStatement();
            }
        } else {
            consume(TokenType::Semicolon, "Expected ';' after for init.");
        }

        std::unique_ptr<Expression> cond;
        if (!check(TokenType::Semicolon)) {
            cond = parseExpression();
        }
        consume(TokenType::Semicolon, "Expected ';' in for condition.");

        std::unique_ptr<Expression> incr;
        if (!check(TokenType::RightParen)) {
            incr = parseExpression();
        }
        consume(TokenType::RightParen, "Expected ')' after for clause.");

        auto body = parseStatement();
        return std::make_unique<ForStatement>(std::move(init), std::move(cond), std::move(incr), std::move(body));
    }

    std::unique_ptr<Statement> parseBlockStatementNoConsume() {
        // '{'를 match()한 상태로 진입했으므로
        auto block = std::make_unique<BlockStatement>();
        while (!check(TokenType::RightBrace) && !isAtEnd()) {
            if (isDeclarationStart()) {
                block->statements.push_back(parseFunctionOrVariableDeclaration());
            } else {
                block->statements.push_back(parseStatement());
            }
        }
        consume(TokenType::RightBrace, "Expected '}' after block.");
        return block;
    }

    std::unique_ptr<Statement> parseExpressionStatement() {
        auto expr = parseExpression();
        consume(TokenType::Semicolon, "Expected ';' after expression.");
        return std::make_unique<ExpressionStatement>(std::move(expr));
    }

    //=== parseExpression() (중요!) ===
    // 여기서 "간단한" 표현식 파서를 구현
    // - unary('!'/'-')
    // - 괄호
    // - 숫자
    // - 식별자
    std::unique_ptr<Expression> parseExpression() {
        return parsePrimary();
    }

    std::unique_ptr<Expression> parsePrimary() {
        if (match({TokenType::Minus})) {
            auto op = std::string("-");
            auto right = parsePrimary();
            return std::make_unique<UnaryExpression>(op, std::move(right));
        }
        if (match({TokenType::Bang})) {
            auto op = std::string("!");
            auto right = parsePrimary();
            return std::make_unique<UnaryExpression>(op, std::move(right));
        }
        if (match({TokenType::LeftParen})) {
            auto expr = parseExpression();
            consume(TokenType::RightParen, "Expected ')' after expression.");
            return expr;
        }
        if (match({TokenType::Number})) {
            // ex) 42
            return std::make_unique<LiteralExpression>(previous().lexeme);
        }
        if (match({TokenType::Identifier})) {
            // ex) value
            return std::make_unique<IdentifierExpression>(previous().lexeme);
        }
        throw error(peek(), "Expected expression.");
    }
};

//===========================================================
// 4. Semantic Analyzer (간단)
//===========================================================

class ASTVisitor;
class SemanticAnalyzer : public ASTVisitor {
public:
    SemanticAnalyzer() : errorCount(0) {}
    void analyze(ASTNode *root) {
        root->accept(*this);
        if (errorCount>0) {
            std::cerr << errorCount << " semantic error(s) found.\n";
        } else {
            std::cout << "Semantic analysis completed successfully.\n";
        }
    }
    bool hasErrors() const { return (errorCount>0); }

    void visit(LiteralExpression* node) override {}
    void visit(IdentifierExpression* node) override {}
    void visit(BinaryExpression* node) override {
        node->left->accept(*this);
        node->right->accept(*this);
    }
    void visit(UnaryExpression* node) override {
        node->operand->accept(*this);
    }
    void visit(BlockStatement* node) override {
        for (auto &stmt : node->statements) {
            stmt->accept(*this);
        }
    }
    void visit(ExpressionStatement* node) override {
        node->expression->accept(*this);
    }
    void visit(IfStatement* node) override {
        node->condition->accept(*this);
        node->thenBranch->accept(*this);
        if (node->elseBranch) node->elseBranch->accept(*this);
    }
    void visit(WhileStatement* node) override {
        node->condition->accept(*this);
        node->body->accept(*this);
    }
    void visit(ForStatement* node) override {
        if (node->initializer) node->initializer->accept(*this);
        if (node->condition)   node->condition->accept(*this);
        if (node->increment)   node->increment->accept(*this);
        node->body->accept(*this);
    }
    void visit(ReturnStatement* node) override {
        if (node->expression) node->expression->accept(*this);
    }
    void visit(VariableDeclaration* node) override {
        if (node->initializer) node->initializer->accept(*this);
    }
    void visit(FunctionDeclaration* node) override {
        if (node->body) node->body->accept(*this);
    }

private:
    int errorCount;
};

//===========================================================
// 5. Optimizer (간단)
//===========================================================

class Optimizer {
public:
    void optimize(ASTNode *root) {
        // 생략
    }
};

//===========================================================
// 6. CodeGenerator (간단)
//===========================================================

class CodeGenerator : public ASTVisitor {
public:
    std::string generate(ASTNode *root) {
        code.str("");
        root->accept(*this);
        return code.str();
    }

    void visit(BlockStatement* node) override {
        for (auto &stmt : node->statements) {
            stmt->accept(*this);
        }
    }
    void visit(ExpressionStatement* node) override {
        code << "; ExpressionStatement\n";
    }
    void visit(IfStatement* node) override {
        code << "; IfStatement\n";
        node->condition->accept(*this);
        node->thenBranch->accept(*this);
        if (node->elseBranch) node->elseBranch->accept(*this);
    }
    void visit(WhileStatement* node) override {
        code << "; WhileStatement\n";
        node->condition->accept(*this);
        node->body->accept(*this);
    }
    void visit(ForStatement* node) override {
        code << "; ForStatement\n";
        if (node->initializer) node->initializer->accept(*this);
        if (node->condition)   node->condition->accept(*this);
        if (node->increment)   node->increment->accept(*this);
        node->body->accept(*this);
    }
    void visit(ReturnStatement* node) override {
        code << "; ReturnStatement\n";
        if (node->expression) {
            node->expression->accept(*this);
        }
    }
    void visit(VariableDeclaration* node) override {
        code << "; VarDecl " << node->type << " " << node->name << "\n";
        if (node->initializer) {
            code << "; initializer:\n";
            node->initializer->accept(*this);
        }
    }
    void visit(FunctionDeclaration* node) override {
        code << "define i32 @" << node->name << "() {\n";
        if (node->body) {
            node->body->accept(*this);
        }
        code << "  ret i32 0\n}\n\n";
    }
    // Expressions
    void visit(LiteralExpression* node) override {
        code << "; literal " << node->value << "\n";
    }
    void visit(IdentifierExpression* node) override {
        code << "; identifier " << node->name << "\n";
    }
    void visit(BinaryExpression* node) override {
        code << "; binary " << node->op << "\n";
        node->left->accept(*this);
        node->right->accept(*this);
    }
    void visit(UnaryExpression* node) override {
        code << "; unary " << node->op << "\n";
        node->operand->accept(*this);
    }
private:
    std::stringstream code;
};

//===========================================================
// 7. Compiler Driver
//===========================================================

class Compiler {
public:
    Compiler(const std::string &src) : source(src) {}

    bool compile() {
        if (!lexicalAnalysis()) return false;
        if (!parsing()) return false;
        if (!semanticAnalysis()) return false;
        optimization();
        if (!codeGeneration()) return false;
        return true;
    }
    const std::string& getGeneratedCode() const {
        return generatedCode;
    }
private:
    std::string source;
    std::vector<Token> tokens;
    std::unique_ptr<Statement> ast;
    std::string generatedCode;

    bool lexicalAnalysis() {
        std::cout << "[Compiler] Starting lexical analysis...\n";
        Lexer lexer(source);
        tokens = lexer.tokenize();
        if (tokens.empty()) {
            std::cerr << "[Compiler] No tokens.\n";
            return false;
        }
        std::cout << "[Compiler] Lexical analysis complete. Tokens: " << tokens.size() << "\n";
        return true;
    }
    bool parsing() {
        std::cout << "[Compiler] Starting parsing...\n";
        Parser parser(tokens);
        try {
            ast = parser.parse();
        } catch (const std::exception &e) {
            std::cerr << "[Compiler] Parse error: " << e.what() << "\n";
            return false;
        }
        std::cout << "[Compiler] Parsing complete.\n";
        return true;
    }
    bool semanticAnalysis() {
        std::cout << "[Compiler] Starting semantic analysis...\n";
        SemanticAnalyzer sem;
        ast->accept(sem);
        if (sem.hasErrors()) {
            std::cerr << "[Compiler] Semantic errors found.\n";
            return false;
        }
        std::cout << "[Compiler] Semantic analysis complete.\n";
        return true;
    }
    void optimization() {
        std::cout << "[Compiler] Starting optimization...\n";
        Optimizer opt;
        opt.optimize(ast.get());
        std::cout << "[Compiler] Optimization complete.\n";
    }
    bool codeGeneration() {
        std::cout << "[Compiler] Starting code generation...\n";
        CodeGenerator cg;
        generatedCode = cg.generate(ast.get());
        if (generatedCode.empty()) {
            std::cerr << "[Compiler] No code generated.\n";
            return false;
        }
        std::cout << "[Compiler] Code generation complete.\n";
        return true;
    }
};

//===========================================================
// 8. Main (테스트)
//===========================================================

int main() {
    // 간단 C++ 유사 문법 예시
    std::string sourceCode = R"(
        // Example
        class MyClass : public BaseClass {
        public:
            static const int value = 42;
            int getValue() {
                return value;
            }
        };
    )";

    Compiler compiler(sourceCode);
    if (!compiler.compile()) {
        std::cerr << "[Main] Compilation failed.\n";
        return 1;
    }
    std::cout << "[Main] Compilation succeeded.\n";
    std::cout << "=== Generated Code ===\n";
    std::cout << compiler.getGeneratedCode() << "\n";
    return 0;
}
