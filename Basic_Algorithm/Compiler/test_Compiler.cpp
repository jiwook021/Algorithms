/**
 * @file test_Compiler.cpp
 * @brief Unit tests for Compiler (lexer, parser, full pipeline)
 */

#include "Compiler.hpp"
#include <gtest/gtest.h>

using namespace CompilerModule;

// ===== Lexer Tests =====
TEST(Lexer, TokenizeSimpleDeclaration) {
    Lexer lexer("int x = 42;");
    auto tokens = lexer.Tokenize();
    EXPECT_EQ(tokens[0].Type, TokenType::Identifier);
    EXPECT_EQ(tokens[0].Lexeme, "int");
    EXPECT_EQ(tokens[1].Lexeme, "x");
    EXPECT_EQ(tokens[2].Type, TokenType::Equal);
    EXPECT_EQ(tokens[3].Type, TokenType::Number);
    EXPECT_EQ(tokens[3].Lexeme, "42");
    EXPECT_EQ(tokens[4].Type, TokenType::Semicolon);
}

TEST(Lexer, RecognizesKeywords) {
    Lexer lexer("if else while for return class struct");
    auto tokens = lexer.Tokenize();
    EXPECT_EQ(tokens[0].Type, TokenType::If);
    EXPECT_EQ(tokens[1].Type, TokenType::Else);
    EXPECT_EQ(tokens[2].Type, TokenType::While);
    EXPECT_EQ(tokens[3].Type, TokenType::For);
    EXPECT_EQ(tokens[4].Type, TokenType::Return);
    EXPECT_EQ(tokens[5].Type, TokenType::Class);
    EXPECT_EQ(tokens[6].Type, TokenType::Struct);
}

TEST(Lexer, RecognizesOperators) {
    Lexer lexer("== <= < = + - *");
    auto tokens = lexer.Tokenize();
    EXPECT_EQ(tokens[0].Type, TokenType::EqualEqual);
    EXPECT_EQ(tokens[1].Type, TokenType::LessEqual);
    EXPECT_EQ(tokens[2].Type, TokenType::Less);
    EXPECT_EQ(tokens[3].Type, TokenType::Equal);
}

TEST(Lexer, EmptyInput) {
    Lexer lexer("");
    auto tokens = lexer.Tokenize();
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0].Type, TokenType::EndOfFile);
}

TEST(Lexer, RecognizesBraces) {
    Lexer lexer("{ ( ) }");
    auto tokens = lexer.Tokenize();
    EXPECT_EQ(tokens[0].Type, TokenType::LeftBrace);
    EXPECT_EQ(tokens[1].Type, TokenType::LeftParen);
    EXPECT_EQ(tokens[2].Type, TokenType::RightParen);
    EXPECT_EQ(tokens[3].Type, TokenType::RightBrace);
}

TEST(Lexer, SkipsComments) {
    Lexer lexer("int x; // comment\nint y;");
    auto tokens = lexer.Tokenize();
    // Should have: int x ; int y ; EOF
    EXPECT_EQ(tokens[0].Lexeme, "int");
    EXPECT_EQ(tokens[1].Lexeme, "x");
    EXPECT_EQ(tokens[2].Type, TokenType::Semicolon);
    EXPECT_EQ(tokens[3].Lexeme, "int");
    EXPECT_EQ(tokens[4].Lexeme, "y");
}

// ===== Compiler Pipeline Tests =====
TEST(Compiler, CompileSimpleFunction) {
    std::string src = "int main() { return 0; }";
    Compiler compiler(src);
    EXPECT_TRUE(compiler.Compile());
    EXPECT_FALSE(compiler.GetGeneratedCode().empty());
}

TEST(Compiler, CompileVariableDeclaration) {
    std::string src = "int main() { int x = 42; return x; }";
    Compiler compiler(src);
    EXPECT_TRUE(compiler.Compile());
}

TEST(Compiler, CompileClassDeclaration) {
    std::string src = R"(
        class Foo {
        public:
            int GetValue() { return 0; }
        };
    )";
    Compiler compiler(src);
    EXPECT_TRUE(compiler.Compile());
}
