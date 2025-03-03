/**
 * @file Compiler.cpp
 * @brief Implementation of the simple C++ compiler
 */

#include "Compiler.hpp"

namespace CompilerModule {

// ===== TokenTypeName =====
const char* TokenTypeName(TokenType t) {
    switch (t) {
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
}

TokenType CheckKeyword(const std::string& text) {
    if (text == "class") return TokenType::Class;
    if (text == "struct") return TokenType::Struct;
    if (text == "static") return TokenType::Static;
    if (text == "const") return TokenType::Const;
    if (text == "return") return TokenType::Return;
    if (text == "if") return TokenType::If;
    if (text == "else") return TokenType::Else;
    if (text == "for") return TokenType::For;
    if (text == "while") return TokenType::While;
    return TokenType::Identifier;
}

// ===== Lexer =====
Lexer::Lexer(const std::string& source)
    : source_(source), index_(0), start_(0), line_(1), column_(1) {}

bool Lexer::IsAtEnd() const { return index_ >= source_.size(); }
char Lexer::Peek() const { return IsAtEnd() ? '\0' : source_[index_]; }
char Lexer::PeekNext() const { return (index_ + 1 < source_.size()) ? source_[index_ + 1] : '\0'; }

char Lexer::Advance() {
    char c = source_[index_++];
    if (c == '\n') { line_++; column_ = 1; } else { column_++; }
    return c;
}

bool Lexer::Match(char expected) {
    if (IsAtEnd() || source_[index_] != expected) return false;
    Advance();
    return true;
}

void Lexer::SkipWhitespaceAndComments() {
    while (!IsAtEnd()) {
        char c = Peek();
        if (std::isspace(c)) { Advance(); }
        else if (c == '/' && PeekNext() == '/') {
            while (!IsAtEnd() && Peek() != '\n') Advance();
        } else if (c == '/' && PeekNext() == '*') {
            Advance(); Advance();
            while (!IsAtEnd() && !(Peek() == '*' && PeekNext() == '/')) Advance();
            if (!IsAtEnd()) { Advance(); Advance(); }
        } else break;
    }
}

Token Lexer::MakeToken(TokenType t) {
    std::string text = source_.substr(start_, index_ - start_);
    return {t, text, line_, column_ - static_cast<int>(index_ - start_)};
}

Token Lexer::ErrorToken(const std::string& msg) {
    return {TokenType::Error, msg, line_, column_};
}

Token Lexer::ScanToken() {
    char c = Advance();
    if (std::isalpha(c) || c == '_') return IdentifierOrKeyword();
    if (std::isdigit(c)) return ScanNumber();

    switch (c) {
        case '(': return MakeToken(TokenType::LeftParen);
        case ')': return MakeToken(TokenType::RightParen);
        case '{': return MakeToken(TokenType::LeftBrace);
        case '}': return MakeToken(TokenType::RightBrace);
        case ',': return MakeToken(TokenType::Comma);
        case ';': return MakeToken(TokenType::Semicolon);
        case ':': return MakeToken(TokenType::Colon);
        case '+': return MakeToken(TokenType::Plus);
        case '-': return MakeToken(TokenType::Minus);
        case '*': return MakeToken(TokenType::Star);
        case '/': return MakeToken(TokenType::Slash);
        case '%': return MakeToken(TokenType::Percent);
        case '=': return Match('=') ? MakeToken(TokenType::EqualEqual) : MakeToken(TokenType::Equal);
        case '!': return Match('=') ? MakeToken(TokenType::BangEqual) : MakeToken(TokenType::Bang);
        case '<': return Match('=') ? MakeToken(TokenType::LessEqual) : MakeToken(TokenType::Less);
        case '>': return Match('=') ? MakeToken(TokenType::GreaterEqual) : MakeToken(TokenType::Greater);
        default: return ErrorToken(std::string("Unexpected character '") + c + "'");
    }
}

Token Lexer::IdentifierOrKeyword() {
    while (!IsAtEnd() && (std::isalnum(Peek()) || Peek() == '_')) Advance();
    std::string text = source_.substr(start_, index_ - start_);
    return {CheckKeyword(text), text, line_, column_ - static_cast<int>(index_ - start_)};
}

Token Lexer::ScanNumber() {
    while (!IsAtEnd() && std::isdigit(Peek())) Advance();
    return MakeToken(TokenType::Number);
}

std::vector<Token> Lexer::Tokenize() {
    std::vector<Token> tokens;
    while (!IsAtEnd()) {
        SkipWhitespaceAndComments();
        if (IsAtEnd()) break;
        start_ = index_;
        tokens.push_back(ScanToken());
    }
    tokens.push_back({TokenType::EndOfFile, "", line_, column_});
    return tokens;
}

// ===== AST Accept =====
void LiteralExpression::Accept(AstVisitor& v) { v.Visit(this); }
void IdentifierExpression::Accept(AstVisitor& v) { v.Visit(this); }
void BinaryExpression::Accept(AstVisitor& v) { v.Visit(this); }
void UnaryExpression::Accept(AstVisitor& v) { v.Visit(this); }
void BlockStatement::Accept(AstVisitor& v) { v.Visit(this); }
void ExpressionStatement::Accept(AstVisitor& v) { v.Visit(this); }
void IfStatement::Accept(AstVisitor& v) { v.Visit(this); }
void WhileStatement::Accept(AstVisitor& v) { v.Visit(this); }
void ForStatement::Accept(AstVisitor& v) { v.Visit(this); }
void ReturnStatement::Accept(AstVisitor& v) { v.Visit(this); }
void VariableDeclaration::Accept(AstVisitor& v) { v.Visit(this); }
void FunctionDeclaration::Accept(AstVisitor& v) { v.Visit(this); }

// ===== Parser =====
Parser::Parser(const std::vector<Token>& tokens) : tokens_(tokens), current_(0) {}

bool Parser::IsAtEnd() const { return Peek().Type == TokenType::EndOfFile; }
const Token& Parser::Peek() const { return tokens_[current_]; }
const Token& Parser::Previous() const { return tokens_[current_ - 1]; }
bool Parser::Check(TokenType t) const { return !IsAtEnd() && Peek().Type == t; }

bool Parser::Match(std::initializer_list<TokenType> types) {
    for (auto t : types) { if (Check(t)) { Advance(); return true; } }
    return false;
}

const Token& Parser::Advance() { if (!IsAtEnd()) current_++; return Previous(); }

const Token& Parser::Consume(TokenType t, const std::string& msg) {
    if (Check(t)) return Advance();
    throw Error(Peek(), msg);
}

std::runtime_error Parser::Error(const Token& tk, const std::string& msg) {
    return std::runtime_error(msg);
}

void Parser::Synchronize() {
    Advance();
    while (!IsAtEnd()) {
        if (Previous().Type == TokenType::Semicolon) return;
        switch (Peek().Type) {
            case TokenType::Class: case TokenType::Struct:
            case TokenType::If: case TokenType::For:
            case TokenType::While: case TokenType::Return: return;
            default: break;
        }
        Advance();
    }
}

std::unique_ptr<Statement> Parser::Parse() {
    auto block = std::make_unique<BlockStatement>();
    while (!IsAtEnd()) {
        try { block->Statements.push_back(ParseDeclaration()); }
        catch (std::runtime_error&) { Synchronize(); }
    }
    return block;
}

std::unique_ptr<Statement> Parser::ParseDeclaration() {
    SkipAccessSpecifier();
    if (Match({TokenType::Class})) return ParseClassDeclaration();
    if (Match({TokenType::Struct})) return ParseStructDeclaration();
    return ParseFunctionOrVariableDeclaration();
}

bool Parser::SkipAccessSpecifier() {
    if (Check(TokenType::Identifier)) {
        std::string kw = Peek().Lexeme;
        if ((kw == "public" || kw == "private" || kw == "protected") &&
            current_ + 1 < tokens_.size() && tokens_[current_ + 1].Type == TokenType::Colon) {
            Advance(); Advance();
            return true;
        }
    }
    return false;
}

std::unique_ptr<Statement> Parser::ParseClassDeclaration() {
    Token name = Consume(TokenType::Identifier, "Expected class name.");
    if (Match({TokenType::Colon})) {
        if (Check(TokenType::Identifier)) Advance();
        if (Check(TokenType::Identifier)) Advance();
    }
    Consume(TokenType::LeftBrace, "Expected '{' after class name.");
    auto classFake = std::make_unique<FunctionDeclaration>(name.Lexeme, "class");
    while (!Check(TokenType::RightBrace) && !IsAtEnd()) {
        classFake->Body = ParseBlockStatementInsideClass();
    }
    Consume(TokenType::RightBrace, "Expected '}' after class body.");
    if (Check(TokenType::Semicolon)) Advance();
    return classFake;
}

std::unique_ptr<Statement> Parser::ParseStructDeclaration() {
    Token name = Consume(TokenType::Identifier, "Expected struct name.");
    Consume(TokenType::LeftBrace, "Expected '{' after struct name.");
    auto structFake = std::make_unique<FunctionDeclaration>(name.Lexeme, "struct");
    while (!Check(TokenType::RightBrace) && !IsAtEnd()) {
        structFake->Body = ParseBlockStatementInsideClass();
    }
    Consume(TokenType::RightBrace, "Expected '}' after struct body.");
    if (Check(TokenType::Semicolon)) Advance();
    return structFake;
}

std::unique_ptr<BlockStatement> Parser::ParseBlockStatementInsideClass() {
    auto blk = std::make_unique<BlockStatement>();
    blk->Statements.push_back(ParseDeclaration());
    return blk;
}

std::unique_ptr<Statement> Parser::ParseFunctionOrVariableDeclaration() {
    bool isStatic = false, isConst = false;
    while (Match({TokenType::Static, TokenType::Const})) {
        if (Previous().Type == TokenType::Static) isStatic = true;
        else if (Previous().Type == TokenType::Const) isConst = true;
    }
    Token typeToken = Consume(TokenType::Identifier, "Expected type.");
    Token nameToken = Consume(TokenType::Identifier, "Expected name.");
    if (Check(TokenType::LeftParen))
        return ParseFunctionDeclaration(typeToken.Lexeme, nameToken.Lexeme, isStatic, isConst);
    return ParseVariableDeclaration(typeToken.Lexeme, nameToken.Lexeme, isStatic, isConst);
}

std::unique_ptr<Statement> Parser::ParseVariableDeclaration(
    const std::string& type, const std::string& name, bool isStatic, bool isConst) {
    auto varDecl = std::make_unique<VariableDeclaration>(name, type);
    varDecl->IsStatic = isStatic;
    varDecl->IsConst = isConst;
    if (Match({TokenType::Equal})) varDecl->Initializer = ParseExpression();
    Consume(TokenType::Semicolon, "Expected ';'.");
    return varDecl;
}

std::unique_ptr<Statement> Parser::ParseFunctionDeclaration(
    const std::string& retType, const std::string& name, bool isStatic, bool isConst) {
    auto func = std::make_unique<FunctionDeclaration>(name, retType);
    func->IsStatic = isStatic;
    func->IsConst = isConst;
    Consume(TokenType::LeftParen, "Expected '('.");
    if (!Check(TokenType::RightParen)) {
        do {
            Token pType = Consume(TokenType::Identifier, "Expected param type.");
            Token pName = Consume(TokenType::Identifier, "Expected param name.");
            func->Parameters.push_back({pType.Lexeme, pName.Lexeme});
        } while (Match({TokenType::Comma}));
    }
    Consume(TokenType::RightParen, "Expected ')'.");
    func->Body = ParseBlockStatement();
    return func;
}

std::unique_ptr<BlockStatement> Parser::ParseBlockStatement() {
    auto block = std::make_unique<BlockStatement>();
    Consume(TokenType::LeftBrace, "Expected '{'.");
    while (!Check(TokenType::RightBrace) && !IsAtEnd()) {
        if (IsDeclarationStart()) block->Statements.push_back(ParseFunctionOrVariableDeclaration());
        else block->Statements.push_back(ParseStatement());
    }
    Consume(TokenType::RightBrace, "Expected '}'.");
    return block;
}

bool Parser::IsDeclarationStart() {
    if (Check(TokenType::Static) || Check(TokenType::Const)) return true;
    if (Check(TokenType::Identifier) && current_ + 1 < tokens_.size() &&
        tokens_[current_ + 1].Type == TokenType::Identifier) return true;
    return false;
}

std::unique_ptr<Statement> Parser::ParseStatement() {
    if (Match({TokenType::Return})) return ParseReturnStatement();
    if (Match({TokenType::If})) return ParseIfStatement();
    if (Match({TokenType::While})) return ParseWhileStatement();
    if (Match({TokenType::For})) return ParseForStatement();
    if (Match({TokenType::LeftBrace})) return ParseBlockStatementNoConsume();
    return ParseExpressionStatement();
}

std::unique_ptr<Statement> Parser::ParseReturnStatement() {
    if (!Check(TokenType::Semicolon)) {
        auto expr = ParseExpression();
        Consume(TokenType::Semicolon, "Expected ';'.");
        return std::make_unique<ReturnStatement>(std::move(expr));
    }
    Consume(TokenType::Semicolon, "Expected ';'.");
    return std::make_unique<ReturnStatement>(nullptr);
}

std::unique_ptr<Statement> Parser::ParseIfStatement() {
    Consume(TokenType::LeftParen, "Expected '('.");
    auto cond = ParseExpression();
    Consume(TokenType::RightParen, "Expected ')'.");
    auto thenStmt = ParseStatement();
    std::unique_ptr<Statement> elseStmt;
    if (Match({TokenType::Else})) elseStmt = ParseStatement();
    return std::make_unique<IfStatement>(std::move(cond), std::move(thenStmt), std::move(elseStmt));
}

std::unique_ptr<Statement> Parser::ParseWhileStatement() {
    Consume(TokenType::LeftParen, "Expected '('.");
    auto cond = ParseExpression();
    Consume(TokenType::RightParen, "Expected ')'.");
    auto body = ParseStatement();
    return std::make_unique<WhileStatement>(std::move(cond), std::move(body));
}

std::unique_ptr<Statement> Parser::ParseForStatement() {
    Consume(TokenType::LeftParen, "Expected '('.");
    std::unique_ptr<Statement> init;
    if (!Check(TokenType::Semicolon)) {
        if (IsDeclarationStart()) init = ParseFunctionOrVariableDeclaration();
        else init = ParseExpressionStatement();
    } else { Consume(TokenType::Semicolon, "Expected ';'."); }

    std::unique_ptr<Expression> cond;
    if (!Check(TokenType::Semicolon)) cond = ParseExpression();
    Consume(TokenType::Semicolon, "Expected ';'.");

    std::unique_ptr<Expression> incr;
    if (!Check(TokenType::RightParen)) incr = ParseExpression();
    Consume(TokenType::RightParen, "Expected ')'.");

    auto body = ParseStatement();
    return std::make_unique<ForStatement>(std::move(init), std::move(cond), std::move(incr), std::move(body));
}

std::unique_ptr<Statement> Parser::ParseBlockStatementNoConsume() {
    auto block = std::make_unique<BlockStatement>();
    while (!Check(TokenType::RightBrace) && !IsAtEnd()) {
        if (IsDeclarationStart()) block->Statements.push_back(ParseFunctionOrVariableDeclaration());
        else block->Statements.push_back(ParseStatement());
    }
    Consume(TokenType::RightBrace, "Expected '}'.");
    return block;
}

std::unique_ptr<Statement> Parser::ParseExpressionStatement() {
    auto expr = ParseExpression();
    Consume(TokenType::Semicolon, "Expected ';'.");
    return std::make_unique<ExpressionStatement>(std::move(expr));
}

std::unique_ptr<Expression> Parser::ParseExpression() { return ParsePrimary(); }

std::unique_ptr<Expression> Parser::ParsePrimary() {
    if (Match({TokenType::Minus})) return std::make_unique<UnaryExpression>("-", ParsePrimary());
    if (Match({TokenType::Bang})) return std::make_unique<UnaryExpression>("!", ParsePrimary());
    if (Match({TokenType::LeftParen})) {
        auto expr = ParseExpression();
        Consume(TokenType::RightParen, "Expected ')'.");
        return expr;
    }
    if (Match({TokenType::Number})) return std::make_unique<LiteralExpression>(Previous().Lexeme);
    if (Match({TokenType::Identifier})) return std::make_unique<IdentifierExpression>(Previous().Lexeme);
    throw Error(Peek(), "Expected expression.");
}

// ===== Semantic Analyzer =====
class SemanticAnalyzer : public AstVisitor {
public:
    SemanticAnalyzer() : errorCount_(0) {}
    void Analyze(AstNode* root) { root->Accept(*this); }
    bool HasErrors() const { return errorCount_ > 0; }

    void Visit(LiteralExpression*) override {}
    void Visit(IdentifierExpression*) override {}
    void Visit(BinaryExpression* n) override { n->Left->Accept(*this); n->Right->Accept(*this); }
    void Visit(UnaryExpression* n) override { n->Operand->Accept(*this); }
    void Visit(BlockStatement* n) override { for (auto& s : n->Statements) s->Accept(*this); }
    void Visit(ExpressionStatement* n) override { n->Expr->Accept(*this); }
    void Visit(IfStatement* n) override {
        n->Condition->Accept(*this); n->ThenBranch->Accept(*this);
        if (n->ElseBranch) n->ElseBranch->Accept(*this);
    }
    void Visit(WhileStatement* n) override { n->Condition->Accept(*this); n->Body->Accept(*this); }
    void Visit(ForStatement* n) override {
        if (n->Initializer) n->Initializer->Accept(*this);
        if (n->Condition) n->Condition->Accept(*this);
        if (n->Increment) n->Increment->Accept(*this);
        n->Body->Accept(*this);
    }
    void Visit(ReturnStatement* n) override { if (n->Expr) n->Expr->Accept(*this); }
    void Visit(VariableDeclaration* n) override { if (n->Initializer) n->Initializer->Accept(*this); }
    void Visit(FunctionDeclaration* n) override { if (n->Body) n->Body->Accept(*this); }

private:
    int errorCount_;
};

// ===== Code Generator =====
class CodeGenerator : public AstVisitor {
public:
    std::string Generate(AstNode* root) { code_.str(""); root->Accept(*this); return code_.str(); }

    void Visit(BlockStatement* n) override { for (auto& s : n->Statements) s->Accept(*this); }
    void Visit(ExpressionStatement*) override { code_ << "; ExpressionStatement\n"; }
    void Visit(IfStatement* n) override {
        code_ << "; IfStatement\n"; n->Condition->Accept(*this);
        n->ThenBranch->Accept(*this);
        if (n->ElseBranch) n->ElseBranch->Accept(*this);
    }
    void Visit(WhileStatement* n) override {
        code_ << "; WhileStatement\n"; n->Condition->Accept(*this); n->Body->Accept(*this);
    }
    void Visit(ForStatement* n) override {
        code_ << "; ForStatement\n";
        if (n->Initializer) n->Initializer->Accept(*this);
        if (n->Condition) n->Condition->Accept(*this);
        if (n->Increment) n->Increment->Accept(*this);
        n->Body->Accept(*this);
    }
    void Visit(ReturnStatement* n) override {
        code_ << "; ReturnStatement\n"; if (n->Expr) n->Expr->Accept(*this);
    }
    void Visit(VariableDeclaration* n) override {
        code_ << "; VarDecl " << n->TypeName << " " << n->Name << "\n";
        if (n->Initializer) { code_ << "; initializer:\n"; n->Initializer->Accept(*this); }
    }
    void Visit(FunctionDeclaration* n) override {
        code_ << "define i32 @" << n->Name << "() {\n";
        if (n->Body) n->Body->Accept(*this);
        code_ << "  ret i32 0\n}\n\n";
    }
    void Visit(LiteralExpression* n) override { code_ << "; literal " << n->Value << "\n"; }
    void Visit(IdentifierExpression* n) override { code_ << "; identifier " << n->Name << "\n"; }
    void Visit(BinaryExpression* n) override {
        code_ << "; binary " << n->Op << "\n"; n->Left->Accept(*this); n->Right->Accept(*this);
    }
    void Visit(UnaryExpression* n) override {
        code_ << "; unary " << n->Op << "\n"; n->Operand->Accept(*this);
    }

private:
    std::stringstream code_;
};

// ===== Compiler Driver =====
Compiler::Compiler(const std::string& source) : source_(source) {}

bool Compiler::Compile() {
    if (!LexicalAnalysis()) return false;
    if (!Parsing()) return false;
    if (!SemanticAnalysis()) return false;
    Optimization();
    if (!CodeGeneration()) return false;
    return true;
}

const std::string& Compiler::GetGeneratedCode() const { return generatedCode_; }

bool Compiler::LexicalAnalysis() {
    Lexer lexer(source_);
    tokens_ = lexer.Tokenize();
    return !tokens_.empty();
}

bool Compiler::Parsing() {
    Parser parser(tokens_);
    try { ast_ = parser.Parse(); }
    catch (const std::exception&) { return false; }
    return true;
}

bool Compiler::SemanticAnalysis() {
    SemanticAnalyzer sem;
    ast_->Accept(sem);
    return !sem.HasErrors();
}

void Compiler::Optimization() { /* placeholder */ }

bool Compiler::CodeGeneration() {
    CodeGenerator cg;
    generatedCode_ = cg.Generate(ast_.get());
    return !generatedCode_.empty();
}

} // namespace CompilerModule
