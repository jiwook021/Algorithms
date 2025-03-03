/**
 * @file Compiler.hpp
 * @brief Simple C++ compiler: lexer, parser, AST, semantic analysis, code generation
 * @details Parses a subset of C++ (variables, functions, classes, control flow)
 *          and generates pseudo-assembly output.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

namespace CompilerModule {

// ===== Token Types =====
enum class TokenType {
    LeftParen, RightParen, LeftBrace, RightBrace,
    Comma, Semicolon, Colon,
    Plus, Minus, Star, Slash, Percent,
    Equal, EqualEqual, Bang, BangEqual,
    Less, LessEqual, Greater, GreaterEqual,
    Identifier, Number, String, Char,
    If, Else, For, While, Return,
    Class, Struct, Static, Const,
    EndOfFile, Error
};

const char* TokenTypeName(TokenType t);

TokenType CheckKeyword(const std::string& text);

struct Token {
    TokenType Type;
    std::string Lexeme;
    int Line;
    int Column;
};

// ===== Lexer =====
class Lexer {
public:
    explicit Lexer(const std::string& source);
    std::vector<Token> Tokenize();

private:
    std::string source_;
    std::size_t index_, start_;
    int line_, column_;

    bool IsAtEnd() const;
    char Peek() const;
    char PeekNext() const;
    char Advance();
    bool Match(char expected);
    void SkipWhitespaceAndComments();
    Token MakeToken(TokenType t);
    Token ErrorToken(const std::string& msg);
    Token ScanToken();
    Token IdentifierOrKeyword();
    Token ScanNumber();
};

// ===== AST Nodes =====
class AstVisitor;

class AstNode {
public:
    virtual ~AstNode() = default;
    virtual void Accept(AstVisitor& visitor) = 0;
};

class Expression : public AstNode {};
class Statement : public AstNode {};

class LiteralExpression : public Expression {
public:
    std::string Value;
    explicit LiteralExpression(const std::string& v) : Value(v) {}
    void Accept(AstVisitor& visitor) override;
};

class IdentifierExpression : public Expression {
public:
    std::string Name;
    explicit IdentifierExpression(const std::string& n) : Name(n) {}
    void Accept(AstVisitor& visitor) override;
};

class BinaryExpression : public Expression {
public:
    std::string Op;
    std::unique_ptr<Expression> Left;
    std::unique_ptr<Expression> Right;
    BinaryExpression(const std::string& op,
                     std::unique_ptr<Expression> left,
                     std::unique_ptr<Expression> right)
        : Op(op), Left(std::move(left)), Right(std::move(right)) {}
    void Accept(AstVisitor& visitor) override;
};

class UnaryExpression : public Expression {
public:
    std::string Op;
    std::unique_ptr<Expression> Operand;
    UnaryExpression(const std::string& op, std::unique_ptr<Expression> operand)
        : Op(op), Operand(std::move(operand)) {}
    void Accept(AstVisitor& visitor) override;
};

class BlockStatement : public Statement {
public:
    std::vector<std::unique_ptr<Statement>> Statements;
    void Accept(AstVisitor& visitor) override;
};

class ExpressionStatement : public Statement {
public:
    std::unique_ptr<Expression> Expr;
    explicit ExpressionStatement(std::unique_ptr<Expression> expr)
        : Expr(std::move(expr)) {}
    void Accept(AstVisitor& visitor) override;
};

class IfStatement : public Statement {
public:
    std::unique_ptr<Expression> Condition;
    std::unique_ptr<Statement> ThenBranch;
    std::unique_ptr<Statement> ElseBranch;
    IfStatement(std::unique_ptr<Expression> cond,
                std::unique_ptr<Statement> thenBr,
                std::unique_ptr<Statement> elseBr)
        : Condition(std::move(cond)), ThenBranch(std::move(thenBr)),
          ElseBranch(std::move(elseBr)) {}
    void Accept(AstVisitor& visitor) override;
};

class WhileStatement : public Statement {
public:
    std::unique_ptr<Expression> Condition;
    std::unique_ptr<Statement> Body;
    WhileStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> body)
        : Condition(std::move(cond)), Body(std::move(body)) {}
    void Accept(AstVisitor& visitor) override;
};

class ForStatement : public Statement {
public:
    std::unique_ptr<Statement> Initializer;
    std::unique_ptr<Expression> Condition;
    std::unique_ptr<Expression> Increment;
    std::unique_ptr<Statement> Body;
    ForStatement(std::unique_ptr<Statement> init,
                 std::unique_ptr<Expression> cond,
                 std::unique_ptr<Expression> incr,
                 std::unique_ptr<Statement> body)
        : Initializer(std::move(init)), Condition(std::move(cond)),
          Increment(std::move(incr)), Body(std::move(body)) {}
    void Accept(AstVisitor& visitor) override;
};

class ReturnStatement : public Statement {
public:
    std::unique_ptr<Expression> Expr;
    explicit ReturnStatement(std::unique_ptr<Expression> expr) : Expr(std::move(expr)) {}
    void Accept(AstVisitor& visitor) override;
};

class VariableDeclaration : public Statement {
public:
    std::string Name;
    std::string TypeName;
    bool IsStatic = false;
    bool IsConst = false;
    std::unique_ptr<Expression> Initializer;
    VariableDeclaration(const std::string& n, const std::string& t)
        : Name(n), TypeName(t) {}
    void Accept(AstVisitor& visitor) override;
};

class FunctionDeclaration : public Statement {
public:
    std::string Name;
    std::string ReturnType;
    std::vector<std::pair<std::string, std::string>> Parameters;
    bool IsStatic = false;
    bool IsConst = false;
    std::unique_ptr<BlockStatement> Body;
    FunctionDeclaration(const std::string& n, const std::string& r)
        : Name(n), ReturnType(r) {}
    void Accept(AstVisitor& visitor) override;
};

// ===== Visitor =====
class AstVisitor {
public:
    virtual ~AstVisitor() = default;
    virtual void Visit(LiteralExpression* node) = 0;
    virtual void Visit(IdentifierExpression* node) = 0;
    virtual void Visit(BinaryExpression* node) = 0;
    virtual void Visit(UnaryExpression* node) = 0;
    virtual void Visit(BlockStatement* node) = 0;
    virtual void Visit(ExpressionStatement* node) = 0;
    virtual void Visit(IfStatement* node) = 0;
    virtual void Visit(WhileStatement* node) = 0;
    virtual void Visit(ForStatement* node) = 0;
    virtual void Visit(ReturnStatement* node) = 0;
    virtual void Visit(VariableDeclaration* node) = 0;
    virtual void Visit(FunctionDeclaration* node) = 0;
};

// ===== Parser =====
class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens);
    std::unique_ptr<Statement> Parse();

private:
    const std::vector<Token>& tokens_;
    std::size_t current_;

    bool IsAtEnd() const;
    const Token& Peek() const;
    const Token& Previous() const;
    bool Check(TokenType t) const;
    bool Match(std::initializer_list<TokenType> types);
    const Token& Advance();
    const Token& Consume(TokenType t, const std::string& msg);
    std::runtime_error Error(const Token& tk, const std::string& msg);
    void Synchronize();

    std::unique_ptr<Statement> ParseDeclaration();
    bool SkipAccessSpecifier();
    std::unique_ptr<Statement> ParseClassDeclaration();
    std::unique_ptr<Statement> ParseStructDeclaration();
    std::unique_ptr<BlockStatement> ParseBlockStatementInsideClass();
    std::unique_ptr<Statement> ParseFunctionOrVariableDeclaration();
    std::unique_ptr<Statement> ParseVariableDeclaration(
        const std::string& type, const std::string& name, bool isStatic, bool isConst);
    std::unique_ptr<Statement> ParseFunctionDeclaration(
        const std::string& retType, const std::string& name, bool isStatic, bool isConst);
    std::unique_ptr<BlockStatement> ParseBlockStatement();
    bool IsDeclarationStart();
    std::unique_ptr<Statement> ParseStatement();
    std::unique_ptr<Statement> ParseReturnStatement();
    std::unique_ptr<Statement> ParseIfStatement();
    std::unique_ptr<Statement> ParseWhileStatement();
    std::unique_ptr<Statement> ParseForStatement();
    std::unique_ptr<Statement> ParseBlockStatementNoConsume();
    std::unique_ptr<Statement> ParseExpressionStatement();
    std::unique_ptr<Expression> ParseExpression();
    std::unique_ptr<Expression> ParsePrimary();
};

// ===== Compiler Driver =====
class Compiler {
public:
    explicit Compiler(const std::string& source);
    bool Compile();
    const std::string& GetGeneratedCode() const;

private:
    std::string source_;
    std::vector<Token> tokens_;
    std::unique_ptr<Statement> ast_;
    std::string generatedCode_;

    bool LexicalAnalysis();
    bool Parsing();
    bool SemanticAnalysis();
    void Optimization();
    bool CodeGeneration();
};

} // namespace CompilerModule
