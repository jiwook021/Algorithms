/**
 * SimpleCppCompiler.cpp
 * A simplified C++ compiler that performs lexical analysis, parsing,
 * semantic analysis, optimization, and code generation.
 * 
 * This compiler supports a subset of C++ language features and generates
 * x86-64 assembly code.
 * 
 * Author: Claude
 * Date: March 23, 2025
 */

 #include <algorithm>
 #include <cassert>
 #include <cctype>
 #include <cstdint>
 #include <fstream>
 #include <functional>
 #include <iostream>
 #include <map>
 #include <memory>
 #include <optional>
 #include <set>
 #include <sstream>
 #include <stack>
 #include <stdexcept>
 #include <string>
 #include <string_view>
 #include <unordered_map>
 #include <unordered_set>
 #include <variant>
 #include <vector>
 #include <mutex>
 #include <shared_mutex>
 #include <chrono>
 #include <format>
 
 // Forward declarations
 class Token;
 class Lexer;
 class ASTNode;
 class Parser;
 class SemanticAnalyzer;
 class Optimizer;
 class CodeGenerator;
 class Compiler;
 class SymbolTable;
 class Type;
 class Value;
 class ErrorReporter;
 class Instruction;
 class BasicBlock;
 class Function;
 class Module;
 
 /**
  * @brief Defines the token types used in the lexical analysis phase
  */
 enum class TokenType {
     // Keywords
     INT, CHAR, BOOL, FLOAT, DOUBLE, VOID, AUTO,
     STRUCT, CLASS, ENUM, UNION, TYPEDEF,
     CONST, STATIC, EXTERN, INLINE, VIRTUAL, OVERRIDE, FINAL,
     PUBLIC, PRIVATE, PROTECTED,
     IF, ELSE, WHILE, FOR, DO, SWITCH, CASE, DEFAULT, BREAK, CONTINUE, RETURN,
     NEW, DELETE, TRY, CATCH, THROW, NAMESPACE, USING, TEMPLATE, TYPENAME,
     
     // Operators
     PLUS, MINUS, ASTERISK, SLASH, PERCENT,          // +, -, *, /, %
     AMPERSAND, PIPE, CARET, TILDE, EXCLAMATION,     // &, |, ^, ~, !
     LESS, GREATER, EQUAL, DOT, ARROW,               // <, >, =, ., ->
     PLUS_EQUAL, MINUS_EQUAL, ASTERISK_EQUAL,        // +=, -=, *=
     SLASH_EQUAL, PERCENT_EQUAL, AMPERSAND_EQUAL,    // /=, %=, &=
     PIPE_EQUAL, CARET_EQUAL,                        // |=, ^=
     LESS_LESS, GREATER_GREATER,                     // <<, >>
     LESS_LESS_EQUAL, GREATER_GREATER_EQUAL,         // <<=, >>=
     EQUAL_EQUAL, EXCLAMATION_EQUAL,                 // ==, !=
     LESS_EQUAL, GREATER_EQUAL,                      // <=, >=
     AMPERSAND_AMPERSAND, PIPE_PIPE,                 // &&, ||
     PLUS_PLUS, MINUS_MINUS,                         // ++, --
     COLON_COLON,                                    // ::
 
     // Punctuation
     LEFT_PAREN, RIGHT_PAREN,                        // (, )
     LEFT_BRACKET, RIGHT_BRACKET,                    // [, ]
     LEFT_BRACE, RIGHT_BRACE,                        // {, }
     SEMICOLON, COLON, COMMA, QUESTION,              // ;, :, ,, ?
     
     // Literals
     IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL,
     CHAR_LITERAL, STRING_LITERAL, BOOL_LITERAL,
     
     // Special tokens
     COMMENT, PREPROCESSOR,
     END_OF_FILE, ERROR
 };
 
 /**
  * @brief Main compiler class that coordinates all phases of compilation
  * 
  * Time Complexity: O(n) where n is the size of the source code
  * Space Complexity: O(n) for storing the AST, IR, and generated code
  */
 class Compiler {
 private:
     ErrorReporter errorReporter;
     bool verbose = false;
     
 public:
     Compiler(bool verbose = false) : verbose(verbose) {}
     
     /**
      * @brief Compile a C++ source file to x86-64 assembly
      * @param inputFile The input C++ source file
      * @param outputFile The output assembly file
      * @return True if compilation was successful, false otherwise
      */
     bool compile(const std::string& inputFile, const std::string& outputFile) {
         try {
             // Read the input file
             std::string source = readFile(inputFile);
             
             if (verbose) {
                 std::cout << "Source code loaded (" << source.length() << " bytes)" << std::endl;
             }
             
             // Lexical analysis
             Lexer lexer(source, inputFile, errorReporter);
             std::vector<Token> tokens = lexer.scanTokens();
             
             if (verbose) {
                 std::cout << "Lexical analysis completed (" << tokens.size() << " tokens)" << std::endl;
             }
             
             if (errorReporter.hadError()) {
                 std::cerr << "Compilation failed during lexical analysis" << std::endl;
                 return false;
             }
             
             // Parsing
             Parser parser(tokens, errorReporter);
             auto ast = parser.parse();
             
             if (verbose) {
                 std::cout << "Parsing completed" << std::endl;
             }
             
             if (errorReporter.hadError()) {
                 std::cerr << "Compilation failed during parsing" << std::endl;
                 return false;
             }
             
             // Semantic analysis
             SemanticAnalyzer semanticAnalyzer(errorReporter);
             bool semanticSuccess = semanticAnalyzer.analyze(ast);
             
             if (verbose) {
                 std::cout << "Semantic analysis completed" << std::endl;
             }
             
             if (!semanticSuccess) {
                 std::cerr << "Compilation failed during semantic analysis" << std::endl;
                 return false;
             }
             
             // IR generation (simplified for now)
             std::shared_ptr<Module> module = generateIR(ast);
             
             if (verbose) {
                 std::cout << "IR generation completed" << std::endl;
             }
             
             // Optimization
             Optimizer optimizer(errorReporter);
             optimizer.optimize(module);
             
             if (verbose) {
                 std::cout << "Optimization completed" << std::endl;
             }
             
             // Code generation
             CodeGenerator codeGenerator(errorReporter);
             std::string assembly = codeGenerator.generateCode(module);
             
             if (verbose) {
                 std::cout << "Code generation completed" << std::endl;
             }
             
             // Write the output file
             writeFile(outputFile, assembly);
             
             if (verbose) {
                 std::cout << "Assembly code written to " << outputFile << std::endl;
             }
             
             return true;
         } catch (const std::exception& e) {
             std::cerr << "Error: " << e.what() << std::endl;
             return false;
         }
     }
     
 private:
     /**
      * @brief Read a file into a string
      * @param filename The file to read
      * @return The contents of the file
      */
     std::string readFile(const std::string& filename) {
         std::ifstream file(filename, std::ios::binary | std::ios::ate);
         
         if (!file) {
             throw std::runtime_error("Could not open file: " + filename);
         }
         
         std::streamsize size = file.tellg();
         file.seekg(0, std::ios::beg);
         
         std::string buffer(size, ' ');
         if (!file.read(buffer.data(), size)) {
             throw std::runtime_error("Could not read file: " + filename);
         }
         
         return buffer;
     }
     
     /**
      * @brief Write a string to a file
      * @param filename The file to write to
      * @param content The content to write
      */
     void writeFile(const std::string& filename, const std::string& content) {
         std::ofstream file(filename);
         
         if (!file) {
             throw std::runtime_error("Could not open file for writing: " + filename);
         }
         
         file << content;
         
         if (!file) {
             throw std::runtime_error("Could not write to file: " + filename);
         }
     }
     
     /**
      * @brief Generate IR from the AST
      * @param ast The AST
      * @return The generated IR module
      */
     std::shared_ptr<Module> generateIR(const std::unique_ptr<ProgramNode>& ast) {
         auto module = std::make_shared<Module>("main_module");
         
         // Process each declaration in the AST
         for (const auto& decl : ast->declarations) {
             if (decl->type == ASTNodeType::FUNCTION_DECL) {
                 auto funcDecl = static_cast<const FunctionDeclNode*>(decl.get());
                 generateFunctionIR(module, funcDecl);
             }
             // Other declarations like global variables, classes, etc.
             // would be handled here in a full compiler
         }
         
         // If no main function was found, create a minimal one
         bool hasMain = false;
         for (const auto& func : module->functions) {
             if (func->name == "main") {
                 hasMain = true;
                 break;
             }
         }
         
         if (!hasMain) {
             // Create a main function
             auto mainType = std::make_shared<FunctionType>(
                 std::make_shared<Type>(Type::TypeKind::INT),
                 std::vector<std::shared_ptr<Type>>()
             );
             
             auto mainFunction = std::make_shared<Function>("main", mainType);
             
             // Create entry block
             auto entryBlock = std::make_shared<BasicBlock>("entry");
             
             // Create a return 0 instruction
             auto retInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
             auto returnValue = std::make_shared<Value>(Value::ValueType::INTEGER);
             retInst->addOperand(returnValue);
             
             // Add instruction to block
             entryBlock->addInstruction(retInst);
             
             // Add block to function
             mainFunction->addBlock(entryBlock);
             
             // Add function to module
             module->addFunction(mainFunction);
         }
         
         return module;
     }
     
     /**
      * @brief Generate IR for a function declaration
      * @param module The module to add the function to
      * @param funcDecl The function declaration AST node
      */
     void generateFunctionIR(std::shared_ptr<Module> module, const FunctionDeclNode* funcDecl) {
         // Create function type and function in the IR
         auto function = std::make_shared<Function>(funcDecl->name, funcDecl->type);
         
         // Create entry block
         auto entryBlock = std::make_shared<BasicBlock>("entry");
         function->addBlock(entryBlock);
         
         // Track the current block being generated
         std::shared_ptr<BasicBlock> currentBlock = entryBlock;
         
         // Create a symbol table for this function's scope
         std::unordered_map<std::string, std::shared_ptr<Value>> symbolTable;
         
         // Allocate space for parameters
         for (size_t i = 0; i < funcDecl->parameters.size(); i++) {
             // For simplicity, just create an alloca instruction for each parameter
             auto allocaInst = std::make_shared<Instruction>(Instruction::OpCode::ALLOCA);
             auto paramValue = std::make_shared<Value>(Value::ValueType::POINTER);
             allocaInst->setResult(paramValue);
             
             currentBlock->addInstruction(allocaInst);
             
             // Store the parameter value
             auto storeInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
             auto argValue = std::make_shared<Value>(Value::ValueType::INTEGER); // Simplified
             storeInst->addOperand(argValue);
             storeInst->addOperand(paramValue);
             
             currentBlock->addInstruction(storeInst);
             
             // Add to symbol table
             std::string paramName = "param" + std::to_string(i); // Simplified
             symbolTable[paramName] = paramValue;
         }
         
         // Generate IR for the function body if it exists
         if (funcDecl->body) {
             generateStatementIR(function, currentBlock, funcDecl->body.get(), symbolTable);
         }
         
         // If the function doesn't end with a return, add one
         if (currentBlock->instructions.empty() || 
             currentBlock->instructions.back()->opcode != Instruction::OpCode::RET) {
             
             auto retInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
             
             // If the function has a return type, provide a default return value
             if (funcDecl->type->returnType->kind != Type::TypeKind::VOID) {
                 auto defaultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 retInst->addOperand(defaultValue);
             }
             
             currentBlock->addInstruction(retInst);
         }
         
         // Add function to module
         module->addFunction(function);
     }
     
     /**
      * @brief Generate IR for a statement
      * @param function The current function
      * @param currentBlock The current basic block
      * @param stmt The statement AST node
      * @param symbolTable The symbol table for variable lookup
      * @return The next basic block to use
      */
     std::shared_ptr<BasicBlock> generateStatementIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> currentBlock,
         const ASTNode* stmt,
         std::unordered_map<std::string, std::shared_ptr<Value>>& symbolTable
     ) {
         switch (stmt->type) {
             case ASTNodeType::COMPOUND_STMT: {
                 const CompoundStmtNode* compoundStmt = static_cast<const CompoundStmtNode*>(stmt);
                 
                 // Create a new scope
                 std::unordered_map<std::string, std::shared_ptr<Value>> innerSymbolTable = symbolTable;
                 
                 // Generate IR for each statement in sequence
                 for (const auto& subStmt : compoundStmt->statements) {
                     currentBlock = generateStatementIR(function, currentBlock, subStmt.get(), innerSymbolTable);
                     
                     // If we've ended the block (e.g., with a return), stop
                     if (currentBlock->instructions.size() > 0 &&
                         (currentBlock->instructions.back()->opcode == Instruction::OpCode::RET ||
                          currentBlock->instructions.back()->opcode == Instruction::OpCode::BR)) {
                         break;
                     }
                 }
                 
                 return currentBlock;
             }
             
             case ASTNodeType::EXPRESSION_STMT: {
                 const ExpressionStmtNode* exprStmt = static_cast<const ExpressionStmtNode*>(stmt);
                 
                 // Generate IR for the expression
                 generateExpressionIR(function, currentBlock, exprStmt->expression.get(), symbolTable);
                 
                 return currentBlock;
             }
             
             case ASTNodeType::IF_STMT: {
                 const IfStmtNode* ifStmt = static_cast<const IfStmtNode*>(stmt);
                 
                 // Generate IR for the condition
                 auto condValue = generateExpressionIR(function, currentBlock, ifStmt->condition.get(), symbolTable);
                 
                 // Create then and else blocks
                 auto thenBlock = std::make_shared<BasicBlock>("then" + std::to_string(function->blocks.size()));
                 auto elseBlock = std::make_shared<BasicBlock>("else" + std::to_string(function->blocks.size() + 1));
                 auto mergeBlock = std::make_shared<BasicBlock>("merge" + std::to_string(function->blocks.size() + 2));
                 
                 // Add blocks to function
                 function->addBlock(thenBlock);
                 function->addBlock(elseBlock);
                 function->addBlock(mergeBlock);
                 
                 // Update CFG
                 currentBlock->successors.push_back(thenBlock);
                 currentBlock->successors.push_back(elseBlock);
                 thenBlock->predecessors.push_back(currentBlock);
                 elseBlock->predecessors.push_back(currentBlock);
                 thenBlock->successors.push_back(mergeBlock);
                 elseBlock->successors.push_back(mergeBlock);
                 mergeBlock->predecessors.push_back(thenBlock);
                 mergeBlock->predecessors.push_back(elseBlock);
                 
                 // Create conditional branch
                 auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 brInst->addOperand(condValue);
                 auto thenValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto elseValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 brInst->addOperand(thenValue);
                 brInst->addOperand(elseValue);
                 
                 // Add branch to current block
                 currentBlock->addInstruction(brInst);
                 
                 // Generate IR for then branch
                 auto thenEnd = generateStatementIR(function, thenBlock, ifStmt->thenBranch.get(), symbolTable);
                 
                 // Add branch to merge block if needed
                 if (thenEnd->instructions.empty() ||
                     thenEnd->instructions.back()->opcode != Instruction::OpCode::BR) {
                     auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     auto mergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                     brInst->addOperand(mergeValue);
                     thenEnd->addInstruction(brInst);
                 }
                 
                 // Generate IR for else branch if it exists
                 if (ifStmt->elseBranch) {
                     auto elseEnd = generateStatementIR(function, elseBlock, ifStmt->elseBranch.get(), symbolTable);
                     
                     // Add branch to merge block if needed
                     if (elseEnd->instructions.empty() ||
                         elseEnd->instructions.back()->opcode != Instruction::OpCode::BR) {
                         auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                         auto mergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         brInst->addOperand(mergeValue);
                         elseEnd->addInstruction(brInst);
                     }
                 } else {
                     // Empty else branch, just branch to merge
                     auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     auto mergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                     brInst->addOperand(mergeValue);
                     elseBlock->addInstruction(brInst);
                 }
                 
                 return mergeBlock;
             }
             
             case ASTNodeType::WHILE_STMT: {
                 const WhileStmtNode* whileStmt = static_cast<const WhileStmtNode*>(stmt);
                 
                 // Create loop header, body, and exit blocks
                 auto headerBlock = std::make_shared<BasicBlock>("loop_header" + std::to_string(function->blocks.size()));
                 auto bodyBlock = std::make_shared<BasicBlock>("loop_body" + std::to_string(function->blocks.size() + 1));
                 auto exitBlock = std::make_shared<BasicBlock>("loop_exit" + std::to_string(function->blocks.size() + 2));
                 
                 // Add blocks to function
                 function->addBlock(headerBlock);
                 function->addBlock(bodyBlock);
                 function->addBlock(exitBlock);
                 
                 // Update CFG
                 currentBlock->successors.push_back(headerBlock);
                 headerBlock->predecessors.push_back(currentBlock);
                 headerBlock->successors.push_back(bodyBlock);
                 headerBlock->successors.push_back(exitBlock);
                 bodyBlock->predecessors.push_back(headerBlock);
                 bodyBlock->successors.push_back(headerBlock);
                 headerBlock->predecessors.push_back(bodyBlock);
                 exitBlock->predecessors.push_back(headerBlock);
                 
                 // Branch to loop header
                 auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto headerValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 brInst->addOperand(headerValue);
                 currentBlock->addInstruction(brInst);
                 
                 // Generate IR for condition in header
                 auto condValue = generateExpressionIR(function, headerBlock, whileStmt->condition.get(), symbolTable);
                 
                 // Create conditional branch
                 auto condBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 condBrInst->addOperand(condValue);
                 auto bodyValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto exitValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 condBrInst->addOperand(bodyValue);
                 condBrInst->addOperand(exitValue);
                 
                 // Add branch to header block
                 headerBlock->addInstruction(condBrInst);
                 
                 // Generate IR for loop body
                 auto bodyEnd = generateStatementIR(function, bodyBlock, whileStmt->body.get(), symbolTable);
                 
                 // Branch back to header
                 auto loopBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 loopBrInst->addOperand(headerValue);
                 bodyEnd->addInstruction(loopBrInst);
                 
                 return exitBlock;
             }
             
             case ASTNodeType::FOR_STMT: {
                 const ForStmtNode* forStmt = static_cast<const ForStmtNode*>(stmt);
                 
                 // Generate IR for initializer
                 if (forStmt->initializer) {
                     if (forStmt->initializer->type == ASTNodeType::VARIABLE_DECL) {
                         // Handle variable declaration initializer
                         generateVariableDeclIR(function, currentBlock, 
                                              static_cast<const VariableDeclNode*>(forStmt->initializer.get()), 
                                              symbolTable);
                     } else {
                         // Handle expression initializer
                         generateExpressionIR(function, currentBlock, forStmt->initializer.get(), symbolTable);
                     }
                 }
                 
                 // Create loop header, body, increment, and exit blocks
                 auto headerBlock = std::make_shared<BasicBlock>("for_header" + std::to_string(function->blocks.size()));
                 auto bodyBlock = std::make_shared<BasicBlock>("for_body" + std::to_string(function->blocks.size() + 1));
                 auto incBlock = std::make_shared<BasicBlock>("for_inc" + std::to_string(function->blocks.size() + 2));
                 auto exitBlock = std::make_shared<BasicBlock>("for_exit" + std::to_string(function->blocks.size() + 3));
                 
                 // Add blocks to function
                 function->addBlock(headerBlock);
                 function->addBlock(bodyBlock);
                 function->addBlock(incBlock);
                 function->addBlock(exitBlock);
                 
                 // Update CFG
                 currentBlock->successors.push_back(headerBlock);
                 headerBlock->predecessors.push_back(currentBlock);
                 headerBlock->predecessors.push_back(incBlock);
                 headerBlock->successors.push_back(bodyBlock);
                 headerBlock->successors.push_back(exitBlock);
                 bodyBlock->predecessors.push_back(headerBlock);
                 bodyBlock->successors.push_back(incBlock);
                 incBlock->predecessors.push_back(bodyBlock);
                 incBlock->successors.push_back(headerBlock);
                 exitBlock->predecessors.push_back(headerBlock);
                 
                 // Branch to loop header
                 auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto headerValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 brInst->addOperand(headerValue);
                 currentBlock->addInstruction(brInst);
                 
                 // Generate IR for condition in header
                 std::shared_ptr<Value> condValue;
                 if (forStmt->condition) {
                     condValue = generateExpressionIR(function, headerBlock, forStmt->condition.get(), symbolTable);
                 } else {
                     // If no condition, use true
                     condValue = std::make_shared<Value>(Value::ValueType::BOOLEAN);
                 }
                 
                 // Create conditional branch
                 auto condBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 condBrInst->addOperand(condValue);
                 auto bodyValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto exitValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 condBrInst->addOperand(bodyValue);
                 condBrInst->addOperand(exitValue);
                 
                 // Add branch to header block
                 headerBlock->addInstruction(condBrInst);
                 
                 // Generate IR for loop body
                 auto bodyEnd = generateStatementIR(function, bodyBlock, forStmt->body.get(), symbolTable);
                 
                 // Branch to increment block
                 auto bodyBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto incValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 bodyBrInst->addOperand(incValue);
                 bodyEnd->addInstruction(bodyBrInst);
                 
                 // Generate IR for increment
                 if (forStmt->increment) {
                     generateExpressionIR(function, incBlock, forStmt->increment.get(), symbolTable);
                 }
                 
                 // Branch back to header
                 auto incBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 incBrInst->addOperand(headerValue);
                 incBlock->addInstruction(incBrInst);
                 
                 return exitBlock;
             }
             
             case ASTNodeType::RETURN_STMT: {
                 const ReturnStmtNode* returnStmt = static_cast<const ReturnStmtNode*>(stmt);
                 
                 // Create return instruction
                 auto retInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
                 
                 // If there's a return value, generate IR for it
                 if (returnStmt->value) {
                     auto returnValue = generateExpressionIR(function, currentBlock, returnStmt->value.get(), symbolTable);
                     retInst->addOperand(returnValue);
                 }
                 
                 // Add return instruction to current block
                 currentBlock->addInstruction(retInst);
                 
                 return currentBlock;
             }
             
             case ASTNodeType::VARIABLE_DECL: {
                 generateVariableDeclIR(function, currentBlock, 
                                      static_cast<const VariableDeclNode*>(stmt), 
                                      symbolTable);
                 return currentBlock;
             }
             
             default:
                 // For other statement types, we would add similar logic
                 return currentBlock;
         }
     }
     
     /**
      * @brief Generate IR for a variable declaration
      * @param function The current function
      * @param currentBlock The current basic block
      * @param varDecl The variable declaration AST node
      * @param symbolTable The symbol table for variable lookup
      */
     void generateVariableDeclIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> currentBlock,
         const VariableDeclNode* varDecl,
         std::unordered_map<std::string, std::shared_ptr<Value>>& symbolTable
     ) {
         // Create alloca instruction
         auto allocaInst = std::make_shared<Instruction>(Instruction::OpCode::ALLOCA);
         auto varValue = std::make_shared<Value>(Value::ValueType::POINTER);
         allocaInst->setResult(varValue);
         
         // Add to symbol table
         symbolTable[varDecl->name] = varValue;
         
         // Add alloca instruction to current block
         currentBlock->addInstruction(allocaInst);
         
         // If there's an initializer, generate IR for it and store the result
         if (varDecl->initializer) {
             auto initValue = generateExpressionIR(function, currentBlock, varDecl->initializer.get(), symbolTable);
             
             // Create store instruction
             auto storeInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
             storeInst->addOperand(initValue);
             storeInst->addOperand(varValue);
             
             // Add store instruction to current block
             currentBlock->addInstruction(storeInst);
         }
     }
     
     /**
      * @brief Generate IR for an expression
      * @param function The current function
      * @param currentBlock The current basic block
      * @param expr The expression AST node
      * @param symbolTable The symbol table for variable lookup
      * @return The value produced by the expression
      */
     std::shared_ptr<Value> generateExpressionIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> currentBlock,
         const ASTNode* expr,
         std::unordered_map<std::string, std::shared_ptr<Value>>& symbolTable
     ) {
         switch (expr->type) {
             case ASTNodeType::BINARY_EXPR: {
                 const BinaryExprNode* binExpr = static_cast<const BinaryExprNode*>(expr);
                 
                 // Generate IR for left and right operands
                 auto leftValue = generateExpressionIR(function, currentBlock, binExpr->left.get(), symbolTable);
                 auto rightValue = generateExpressionIR(function, currentBlock, binExpr->right.get(), symbolTable);
                 
                 // Create instruction based on operator
                 std::shared_ptr<Instruction> inst;
                 
                 switch (binExpr->op) {
                     case BinaryExprNode::Operator::ADD:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                         break;
                     case BinaryExprNode::Operator::SUBTRACT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         break;
                     case BinaryExprNode::Operator::MULTIPLY:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::MUL);
                         break;
                     case BinaryExprNode::Operator::DIVIDE:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::DIV);
                         break;
                     case BinaryExprNode::Operator::MODULO:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::MOD);
                         break;
                     case BinaryExprNode::Operator::EQUAL:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::EQ);
                         break;
                     case BinaryExprNode::Operator::NOT_EQUAL:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::NE);
                         break;
                     case BinaryExprNode::Operator::LESS:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::LT);
                         break;
                     case BinaryExprNode::Operator::LESS_EQUAL:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::LE);
                         break;
                     case BinaryExprNode::Operator::GREATER:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::GT);
                         break;
                     case BinaryExprNode::Operator::GREATER_EQUAL:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::GE);
                         break;
                     case BinaryExprNode::Operator::AND:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                         break;
                     case BinaryExprNode::Operator::OR:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                         break;
                     case BinaryExprNode::Operator::BITWISE_AND:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                         break;
                     case BinaryExprNode::Operator::BITWISE_OR:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                         break;
                     case BinaryExprNode::Operator::BITWISE_XOR:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         break;
                     case BinaryExprNode::Operator::LEFT_SHIFT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::SHL);
                         break;
                     case BinaryExprNode::Operator::RIGHT_SHIFT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::SHR);
                         break;
                     default:
                         // Unsupported operator
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 // Add operands and set result
                 inst->addOperand(leftValue);
                 inst->addOperand(rightValue);
                 auto resultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 inst->setResult(resultValue);
                 
                 // Add instruction to current block
                 currentBlock->addInstruction(inst);
                 
                 return resultValue;
             }
             
             case ASTNodeType::UNARY_EXPR: {
                 const UnaryExprNode* unaryExpr = static_cast<const UnaryExprNode*>(expr);
                 
                 // Generate IR for operand
                 auto operandValue = generateExpressionIR(function, currentBlock, unaryExpr->operand.get(), symbolTable);
                 
                 // Create instruction based on operator
                 std::shared_ptr<Instruction> inst;
                 
                 switch (unaryExpr->op) {
                     case UnaryExprNode::Operator::NEGATE:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         {
                             // Create a zero value for negation (0 - operand)
                             auto zeroValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             inst->addOperand(zeroValue);
                             inst->addOperand(operandValue);
                         }
                         break;
                     case UnaryExprNode::Operator::NOT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         {
                             // Create a one value for logical NOT (operand XOR 1)
                             auto oneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             inst->addOperand(operandValue);
                             inst->addOperand(oneValue);
                         }
                         break;
                     case UnaryExprNode::Operator::BITWISE_NOT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         {
                             // Create a -1 value for bitwise NOT (operand XOR -1)
                             auto allOnesValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             inst->addOperand(operandValue);
                             inst->addOperand(allOnesValue);
                         }
                         break;
                     case UnaryExprNode::Operator::ADDRESS_OF:
                         // No instruction needed, the operand itself is the address
                         return operandValue;
                     case UnaryExprNode::Operator::DEREFERENCE:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::LOAD);
                         inst->addOperand(operandValue);
                         break;
                     case UnaryExprNode::Operator::PRE_INCREMENT:
                     case UnaryExprNode::Operator::POST_INCREMENT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                         {
                             // Create a one value for increment (operand + 1)
                             auto oneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             inst->addOperand(operandValue);
                             inst->addOperand(oneValue);
                         }
                         break;
                     case UnaryExprNode::Operator::PRE_DECREMENT:
                     case UnaryExprNode::Operator::POST_DECREMENT:
                         inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         {
                             // Create a one value for decrement (operand - 1)
                             auto oneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             inst->addOperand(operandValue);
                             inst->addOperand(oneValue);
                         }
                         break;
                     default:
                         // Unsupported operator
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 // Set result
                 auto resultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 inst->setResult(resultValue);
                 
                 // Add instruction to current block
                 currentBlock->addInstruction(inst);
                 
                 // For increment/decrement, store the result back to the variable
                 if (unaryExpr->op == UnaryExprNode::Operator::PRE_INCREMENT ||
                     unaryExpr->op == UnaryExprNode::Operator::PRE_DECREMENT ||
                     unaryExpr->op == UnaryExprNode::Operator::POST_INCREMENT ||
                     unaryExpr->op == UnaryExprNode::Operator::POST_DECREMENT) {
                     
                     // Create store instruction
                     auto storeInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
                     storeInst->addOperand(resultValue);
                     
                     // For simplicity, assume the operand is a variable expression
                     // In a real compiler, we would handle more complex lvalues
                     if (unaryExpr->operand->type == ASTNodeType::VARIABLE_EXPR) {
                         const VariableExprNode* varExpr = static_cast<const VariableExprNode*>(unaryExpr->operand.get());
                         auto varValue = symbolTable[varExpr->name];
                         storeInst->addOperand(varValue);
                         
                         // Add store instruction to current block
                         currentBlock->addInstruction(storeInst);
                     }
                     
                     // For post-increment/decrement, return the original value
                     if (unaryExpr->op == UnaryExprNode::Operator::POST_INCREMENT ||
                         unaryExpr->op == UnaryExprNode::Operator::POST_DECREMENT) {
                         return operandValue;
                     }
                 }
                 
                 return resultValue;
             }
             
             case ASTNodeType::LITERAL_EXPR: {
                 const LiteralExprNode* literalExpr = static_cast<const LiteralExprNode*>(expr);
                 
                 // Create a value for the literal
                 std::shared_ptr<Value> literalValue;
                 
                 switch (literalExpr->literalType) {
                     case LiteralExprNode::LiteralType::INTEGER:
                         literalValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                         break;
                     case LiteralExprNode::LiteralType::FLOAT:
                         literalValue = std::make_shared<Value>(Value::ValueType::FLOAT);
                         break;
                     case LiteralExprNode::LiteralType::CHARACTER:
                         literalValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                         break;
                     case LiteralExprNode::LiteralType::STRING:
                         literalValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         break;
                     case LiteralExprNode::LiteralType::BOOLEAN:
                         literalValue = std::make_shared<Value>(Value::ValueType::BOOLEAN);
                         break;
                     case LiteralExprNode::LiteralType::NULL_LITERAL:
                         literalValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         break;
                     default:
                         // Unsupported literal type
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 return literalValue;
             }
             
             case ASTNodeType::VARIABLE_EXPR: {
                 const VariableExprNode* varExpr = static_cast<const VariableExprNode*>(expr);
                 
                 // Look up variable in symbol table
                 auto it = symbolTable.find(varExpr->name);
                 if (it != symbolTable.end()) {
                     // Create load instruction to get the value
                     auto loadInst = std::make_shared<Instruction>(Instruction::OpCode::LOAD);
                     loadInst->addOperand(it->second);
                     auto resultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     loadInst->setResult(resultValue);
                     
                     // Add load instruction to current block
                     currentBlock->addInstruction(loadInst);
                     
                     return resultValue;
                 } else {
                     // Variable not found
                     return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
             }
             
             case ASTNodeType::ASSIGNMENT_EXPR: {
                 const AssignmentExprNode* assignExpr = static_cast<const AssignmentExprNode*>(expr);
                 
                 // Generate IR for right operand
                 auto rightValue = generateExpressionIR(function, currentBlock, assignExpr->right.get(), symbolTable);
                 
                 // Handle special assignment operators (+=, -=, etc.)
                 if (assignExpr->op != AssignmentExprNode::Operator::ASSIGN) {
                     // Generate IR for left operand
                     auto leftValue = generateExpressionIR(function, currentBlock, assignExpr->left.get(), symbolTable);
                     
                     // Create instruction based on operator
                     std::shared_ptr<Instruction> inst;
                     
                     switch (assignExpr->op) {
                         case AssignmentExprNode::Operator::ADD_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                             break;
                         case AssignmentExprNode::Operator::SUBTRACT_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                             break;
                         case AssignmentExprNode::Operator::MULTIPLY_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::MUL);
                             break;
                         case AssignmentExprNode::Operator::DIVIDE_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::DIV);
                             break;
                         case AssignmentExprNode::Operator::MODULO_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::MOD);
                             break;
                         case AssignmentExprNode::Operator::AND_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                             break;
                         case AssignmentExprNode::Operator::OR_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                             break;
                         case AssignmentExprNode::Operator::XOR_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                             break;
                         case AssignmentExprNode::Operator::LEFT_SHIFT_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::SHL);
                             break;
                         case AssignmentExprNode::Operator::RIGHT_SHIFT_ASSIGN:
                             inst = std::make_shared<Instruction>(Instruction::OpCode::SHR);
                             break;
                         default:
                             // Should not happen
                             inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                             break;
                     }
                     
                     // Add operands and set result
                     inst->addOperand(leftValue);
                     inst->addOperand(rightValue);
                     auto resultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     inst->setResult(resultValue);
                     
                     // Add instruction to current block
                     currentBlock->addInstruction(inst);
                     
                     // Update rightValue to be the result of the operation
                     rightValue = resultValue;
                 }
                 
                 // Create store instruction
                 auto storeInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
                 storeInst->addOperand(rightValue);
                 
                 // Handling different types of left operands (variable, dereference, array access, etc.)
                 if (assignExpr->left->type == ASTNodeType::VARIABLE_EXPR) {
                     const VariableExprNode* varExpr = static_cast<const VariableExprNode*>(assignExpr->left.get());
                     
                     // Look up variable in symbol table
                     auto it = symbolTable.find(varExpr->name);
                     if (it != symbolTable.end()) {
                         storeInst->addOperand(it->second);
                     } else {
                         // Variable not found, create a new one
                         auto varValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         symbolTable[varExpr->name] = varValue;
                         storeInst->addOperand(varValue);
                     }
                 } else {
                     // For more complex left operands (dereferenced pointers, array accesses, etc.)
                     // we would need to generate the appropriate IR
                     // For simplicity, just generate a dummy location
                     auto dummyLoc = std::make_shared<Value>(Value::ValueType::POINTER);
                     storeInst->addOperand(dummyLoc);
                 }
                 
                 // Add store instruction to current block
                 currentBlock->addInstruction(storeInst);
                 
                 return rightValue;
             }
             
             case ASTNodeType::CALL_EXPR: {
                 const CallExprNode* callExpr = static_cast<const CallExprNode*>(expr);
                 
                 // Generate IR for arguments
                 std::vector<std::shared_ptr<Value>> argValues;
                 for (const auto& arg : callExpr->arguments) {
                     argValues.push_back(generateExpressionIR(function, currentBlock, arg.get(), symbolTable));
                 }
                 
                 // Create call instruction
                 auto callInst = std::make_shared<Instruction>(Instruction::OpCode::CALL);
                 
                 // Add callee and arguments
                 // For simplicity, assume the callee is a variable expression
                 if (callExpr->callee->type == ASTNodeType::VARIABLE_EXPR) {
                     const VariableExprNode* varExpr = static_cast<const VariableExprNode*>(callExpr->callee.get());
                     
                     // Create a dummy value for the function
                     auto funcValue = std::make_shared<Value>(Value::ValueType::FUNCTION);
                     callInst->addOperand(funcValue);
                     
                     // Add arguments
                     for (auto& argValue : argValues) {
                         callInst->addOperand(argValue);
                     }
                     
                     // Set result
                     auto resultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     callInst->setResult(resultValue);
                     
                     // Add call instruction to current block
                     currentBlock->addInstruction(callInst);
                     
                     return resultValue;
                 } else {
                     // For more complex callees (function pointers, member functions, etc.)
                     // we would need to generate the appropriate IR
                     // For simplicity, just return a dummy value
                     return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
             }
             
             default:
                 // For other expression types, we would add similar logic
                 return std::make_shared<Value>(Value::ValueType::INTEGER);
         }
     }
 };
 
 /**
  * @brief Main function to run the compiler
  * @param argc Argument count
  * @param argv Argument values
  * @return 0 on success, non-zero on failure
  */
 int main(int argc, char* argv[]) {
     try {
         // Default options
         std::string inputFile;
         std::string outputFile;
         bool verbose = false;
         bool optimize = true;
         int optimizationLevel = 1;
         bool showHelp = false;
         bool showVersion = false;
         
         // Parse command line arguments
         for (int i = 1; i < argc; i++) {
             std::string arg = argv[i];
             
             if (arg == "-h" || arg == "--help") {
                 showHelp = true;
             } else if (arg == "-v" || arg == "--verbose") {
                 verbose = true;
             } else if (arg == "--version") {
                 showVersion = true;
             } else if (arg == "-O0") {
                 optimize = false;
                 optimizationLevel = 0;
             } else if (arg == "-O1") {
                 optimize = true;
                 optimizationLevel = 1;
             } else if (arg == "-O2") {
                 optimize = true;
                 optimizationLevel = 2;
             } else if (arg == "-O3") {
                 optimize = true;
                 optimizationLevel = 3;
             } else if (arg == "-o" && i + 1 < argc) {
                 outputFile = argv[++i];
             } else if (arg[0] == '-') {
                 std::cerr << "Unknown option: " << arg << std::endl;
                 showHelp = true;
             } else {
                 if (inputFile.empty()) {
                     inputFile = arg;
                 } else if (outputFile.empty()) {
                     outputFile = arg;
                 } else {
                     std::cerr << "Too many arguments" << std::endl;
                     showHelp = true;
                 }
             }
         }
         
         // Show help message
         if (showHelp) {
             std::cout << "SimpleCppCompiler - A simple C++ compiler" << std::endl;
             std::cout << "Usage: " << (argc > 0 ? argv[0] : "compiler") << " [options] input.cpp [output.asm]" << std::endl;
             std::cout << "Options:" << std::endl;
             std::cout << "  -h, --help      Show this help message" << std::endl;
             std::cout << "  -v, --verbose   Enable verbose output" << std::endl;
             std::cout << "  --version       Show version information" << std::endl;
             std::cout << "  -O0             Disable optimizations" << std::endl;
             std::cout << "  -O1             Enable basic optimizations (default)" << std::endl;
             std::cout << "  -O2             Enable more aggressive optimizations" << std::endl;
             std::cout << "  -O3             Enable all optimizations" << std::endl;
             std::cout << "  -o <file>       Specify output file" << std::endl;
             return 0;
         }
         
         // Show version information
         if (showVersion) {
             std::cout << "SimpleCppCompiler version 1.0.0" << std::endl;
             std::cout << "Built on " << __DATE__ << " " << __TIME__ << std::endl;
             return 0;
         }
         
         // Check for required arguments
         if (inputFile.empty()) {
             std::cerr << "Error: No input file specified" << std::endl;
             std::cerr << "Use --help for more information" << std::endl;
             return 1;
         }
         
         // Set default output file if not specified
         if (outputFile.empty()) {
             size_t dotPos = inputFile.find_last_of('.');
             if (dotPos != std::string::npos) {
                 outputFile = inputFile.substr(0, dotPos) + ".asm";
             } else {
                 outputFile = inputFile + ".asm";
             }
         }
         
         // Create compiler with options
         Compiler compiler(verbose);
         
         // Additional compiler options based on command line arguments
         // In a real compiler, we would set more options here
         
         // Print compilation options if verbose
         if (verbose) {
             std::cout << "Input file: " << inputFile << std::endl;
             std::cout << "Output file: " << outputFile << std::endl;
             std::cout << "Optimization level: " << optimizationLevel << std::endl;
         }
         
         // Run the compiler
         auto startTime = std::chrono::high_resolution_clock::now();
         bool success = compiler.compile(inputFile, outputFile);
         auto endTime = std::chrono::high_resolution_clock::now();
         
         // Calculate compilation time
         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
         
         if (success) {
             std::cout << "Compilation successful: " << inputFile << " -> " << outputFile << std::endl;
             
             if (verbose) {
                 std::cout << "Compilation time: " << duration << " ms" << std::endl;
             }
             
             return 0;
         } else {
             std::cerr << "Compilation failed" << std::endl;
             return 1;
         }
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     } catch (...) {
         std::cerr << "Unknown error occurred" << std::endl;
         return 1;
     }
 }
 
 /**
  * @brief Code generator for x86-64 assembly
  * 
  * Time Complexity: O(n) where n is the number of instructions in the IR
  * Space Complexity: O(n) for storing the generated assembly code
  */
 class CodeGenerator {
 private:
     ErrorReporter& errorReporter;
     std::stringstream output;
     std::unordered_map<std::string, size_t> localVars;
     size_t stackSize = 0;
     size_t labelCounter = 0;
     
 public:
     CodeGenerator(ErrorReporter& errorReporter) : errorReporter(errorReporter) {}
     
     /**
      * @brief Generate assembly code from the intermediate representation
      * @param module The module to generate code for
      * @return The generated assembly code
      */
     std::string generateCode(std::shared_ptr<Module> module) {
         output.str("");
         
         // Generate assembly header
         generateHeader(module->name);
         
         // Generate code for each function
         for (auto& function : module->functions) {
             generateFunction(function);
         }
         
         return output.str();
     }
     
 private:
     /**
      * @brief Generate assembly header
      * @param moduleName The name of the module
      */
     void generateHeader(const std::string& moduleName) {
         output << "; Generated assembly for module: " << moduleName << "\n";
         output << "; Generated by SimpleCppCompiler\n\n";
         
         output << "section .text\n";
         output << "global main\n\n";
         
         // Import external functions
         output << "extern printf\n";
         output << "extern scanf\n";
         output << "extern malloc\n";
         output << "extern free\n\n";
         
         // String literals
         output << "section .data\n";
         output << "format_int db \"%d\", 0\n";
         output << "format_float db \"%f\", 0\n";
         output << "format_char db \"%c\", 0\n";
         output << "format_string db \"%s\", 0\n";
         output << "format_bool_true db \"true\", 0\n";
         output << "format_bool_false db \"false\", 0\n\n";
         
         output << "section .text\n\n";
     }
     
     /**
      * @brief Generate assembly code for a function
      * @param function The function to generate code for
      */
     void generateFunction(std::shared_ptr<Function> function) {
         // Reset local variables and stack size
         localVars.clear();
         stackSize = 0;
         
         // Function label
         output << function->name << ":\n";
         
         // Function prologue
         output << "    push rbp\n";
         output << "    mov rbp, rsp\n";
         
         // Allocate stack space for local variables
         // In a real compiler, we would calculate this based on the variables used
         output << "    sub rsp, 64\n";  // Allocate 64 bytes for local variables
         
         // Generate code for each basic block
         for (auto& block : function->blocks) {
             generateBasicBlock(block);
         }
         
         // Function epilogue
         // This is just a default epilogue; in a real compiler, the actual return point
         // would depend on the control flow
         output << "    mov rsp, rbp\n";
         output << "    pop rbp\n";
         output << "    ret\n\n";
     }
     
     /**
      * @brief Generate assembly code for a basic block
      * @param block The basic block to generate code for
      */
     void generateBasicBlock(std::shared_ptr<BasicBlock> block) {
         output << block->label << ":\n";
         
         // Generate code for each instruction
         for (auto& instruction : block->instructions) {
             generateInstruction(instruction);
         }
     }
     
     /**
      * @brief Generate assembly code for an instruction
      * @param instruction The instruction to generate code for
      */
     void generateInstruction(std::shared_ptr<Instruction> instruction) {
         switch (instruction->opcode) {
             case Instruction::OpCode::ADD:
                 generateAdd(instruction);
                 break;
             case Instruction::OpCode::SUB:
                 generateSub(instruction);
                 break;
             case Instruction::OpCode::MUL:
                 generateMul(instruction);
                 break;
             case Instruction::OpCode::DIV:
                 generateDiv(instruction);
                 break;
             case Instruction::OpCode::MOD:
                 generateMod(instruction);
                 break;
             case Instruction::OpCode::AND:
                 generateAnd(instruction);
                 break;
             case Instruction::OpCode::OR:
                 generateOr(instruction);
                 break;
             case Instruction::OpCode::XOR:
                 generateXor(instruction);
                 break;
             case Instruction::OpCode::SHL:
                 generateShl(instruction);
                 break;
             case Instruction::OpCode::SHR:
                 generateShr(instruction);
                 break;
             case Instruction::OpCode::EQ:
                 generateEq(instruction);
                 break;
             case Instruction::OpCode::NE:
                 generateNe(instruction);
                 break;
             case Instruction::OpCode::LT:
                 generateLt(instruction);
                 break;
             case Instruction::OpCode::LE:
                 generateLe(instruction);
                 break;
             case Instruction::OpCode::GT:
                 generateGt(instruction);
                 break;
             case Instruction::OpCode::GE:
                 generateGe(instruction);
                 break;
             case Instruction::OpCode::ALLOCA:
                 generateAlloca(instruction);
                 break;
             case Instruction::OpCode::LOAD:
                 generateLoad(instruction);
                 break;
             case Instruction::OpCode::STORE:
                 generateStore(instruction);
                 break;
             case Instruction::OpCode::BR:
                 generateBr(instruction);
                 break;
             case Instruction::OpCode::BR_COND:
                 generateBrCond(instruction);
                 break;
             case Instruction::OpCode::CALL:
                 generateCall(instruction);
                 break;
             case Instruction::OpCode::RET:
                 generateRet(instruction);
                 break;
             case Instruction::OpCode::PHI:
                 generatePhi(instruction);
                 break;
             case Instruction::OpCode::CAST:
                 generateCast(instruction);
                 break;
             default:
                 errorReporter.reportError("Unknown instruction opcode", SourceLocation());
                 break;
         }
     }
     
     /**
      * @brief Generate assembly code for an ADD instruction
      * @param instruction The ADD instruction
      */
     void generateAdd(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("ADD instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Add second operand
         output << "    add rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a SUB instruction
      * @param instruction The SUB instruction
      */
     void generateSub(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("SUB instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Subtract second operand
         output << "    sub rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a MUL instruction
      * @param instruction The MUL instruction
      */
     void generateMul(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("MUL instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Multiply by second operand
         output << "    imul rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a DIV instruction
      * @param instruction The DIV instruction
      */
     void generateDiv(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("DIV instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Clear RDX for division
         output << "    xor rdx, rdx\n";
         
         // Load second operand into RCX
         output << "    mov rcx, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Divide
         output << "    div rcx\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a MOD instruction
      * @param instruction The MOD instruction
      */
     void generateMod(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("MOD instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Clear RDX for division
         output << "    xor rdx, rdx\n";
         
         // Load second operand into RCX
         output << "    mov rcx, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Divide
         output << "    div rcx\n";
         
         // Store remainder (modulo)
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rdx\n";
     }
     
     /**
      * @brief Generate assembly code for an AND instruction
      * @param instruction The AND instruction
      */
     void generateAnd(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("AND instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // AND with second operand
         output << "    and rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an OR instruction
      * @param instruction The OR instruction
      */
     void generateOr(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("OR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // OR with second operand
         output << "    or rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an XOR instruction
      * @param instruction The XOR instruction
      */
     void generateXor(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("XOR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // XOR with second operand
         output << "    xor rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an SHL instruction
      * @param instruction The SHL instruction
      */
     void generateShl(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("SHL instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Load second operand into RCX (shift count)
         output << "    mov rcx, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Shift left
         output << "    shl rax, cl\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an SHR instruction
      * @param instruction The SHR instruction
      */
     void generateShr(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("SHR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Load second operand into RCX (shift count)
         output << "    mov rcx, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Shift right
         output << "    shr rax, cl\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an EQ instruction
      * @param instruction The EQ instruction
      */
     void generateEq(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("EQ instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    sete al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an NE instruction
      * @param instruction The NE instruction
      */
     void generateNe(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("NE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    setne al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an LT instruction
      * @param instruction The LT instruction
      */
     void generateLt(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("LT instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    setl al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an LE instruction
      * @param instruction The LE instruction
      */
     void generateLe(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("LE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    setle al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a GT instruction
      * @param instruction The GT instruction
      */
     void generateGt(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("GT instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    setg al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a GE instruction
      * @param instruction The GE instruction
      */
     void generateGe(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("GE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with second operand
         output << "    cmp rax, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Set result based on comparison
         output << "    setge al\n";
         output << "    movzx rax, al\n";
         
         // Store result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an ALLOCA instruction
      * @param instruction The ALLOCA instruction
      */
     void generateAlloca(std::shared_ptr<Instruction> instruction) {
         // Allocate space on the stack for a variable
         // In a real compiler, we would use the size operand to determine how much space to allocate
         
         // For now, just reserve 8 bytes
         stackSize += 8;
         
         // Store the offset for the result
         if (instruction->result) {
             localVars[std::to_string(reinterpret_cast<uintptr_t>(instruction->result.get()))] = stackSize;
         }
     }
     
     /**
      * @brief Generate assembly code for a LOAD instruction
      * @param instruction The LOAD instruction
      */
     void generateLoad(std::shared_ptr<Instruction> instruction) {
         // Ensure we have one operand
         if (instruction->operands.size() != 1) {
             errorReporter.reportError("LOAD instruction requires one operand", SourceLocation());
             return;
         }
         
         // Load the value from the address in the operand
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         output << "    mov rax, [rax]\n";
         
         // Store the loaded value
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a STORE instruction
      * @param instruction The STORE instruction
      */
     void generateStore(std::shared_ptr<Instruction> instruction) {
         // Ensure we have two operands
         if (instruction->operands.size() != 2) {
             errorReporter.reportError("STORE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load the value to be stored
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Load the address to store to
         output << "    mov rcx, [rbp - " << getOperandOffset(instruction->operands[1]) << "]\n";
         
         // Store the value
         output << "    mov [rcx], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a BR instruction
      * @param instruction The BR instruction
      */
     void generateBr(std::shared_ptr<Instruction> instruction) {
         // Ensure we have one operand
         if (instruction->operands.size() != 1) {
             errorReporter.reportError("BR instruction requires one operand", SourceLocation());
             return;
         }
         
         // Branch to the specified label
         // In a real compiler, we would extract the label from the operand
         std::string label = "label" + std::to_string(labelCounter++);
         
         output << "    jmp " << label << "\n";
     }
     
     /**
      * @brief Generate assembly code for a BR_COND instruction
      * @param instruction The BR_COND instruction
      */
     void generateBrCond(std::shared_ptr<Instruction> instruction) {
         // Ensure we have three operands
         if (instruction->operands.size() != 3) {
             errorReporter.reportError("BR_COND instruction requires three operands", SourceLocation());
             return;
         }
         
         // Load the condition
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // Compare with zero
         output << "    cmp rax, 0\n";
         
         // Branch based on the condition
         // In a real compiler, we would extract the labels from the operands
         std::string trueLabel = "label" + std::to_string(labelCounter++);
         std::string falseLabel = "label" + std::to_string(labelCounter++);
         
         output << "    jne " << trueLabel << "\n";
         output << "    jmp " << falseLabel << "\n";
     }
     
     /**
      * @brief Generate assembly code for a CALL instruction
      * @param instruction The CALL instruction
      */
     void generateCall(std::shared_ptr<Instruction> instruction) {
         // Ensure we have at least one operand
         if (instruction->operands.size() < 1) {
             errorReporter.reportError("CALL instruction requires at least one operand", SourceLocation());
             return;
         }
         
         // The first operand is the function to call
         // In a real compiler, we would extract the function name from the operand
         std::string functionName = "function" + std::to_string(labelCounter++);
         
         // Save caller-saved registers
         output << "    push rcx\n";
         output << "    push rdx\n";
         output << "    push rsi\n";
         output << "    push rdi\n";
         output << "    push r8\n";
         output << "    push r9\n";
         output << "    push r10\n";
         output << "    push r11\n";
         
         // Load arguments into registers according to the x86-64 calling convention
         // In a real compiler, we would extract the arguments from the operands
         
         // First 6 arguments go in registers: RDI, RSI, RDX, RCX, R8, R9
         // Additional arguments are pushed on the stack in reverse order
         
         for (size_t i = 1; i < instruction->operands.size() && i <= 6; i++) {
             std::string reg;
             
             switch (i) {
                 case 1: reg = "rdi"; break;
                 case 2: reg = "rsi"; break;
                 case 3: reg = "rdx"; break;
                 case 4: reg = "rcx"; break;
                 case 5: reg = "r8"; break;
                 case 6: reg = "r9"; break;
                 default: reg = ""; break; // Shouldn't happen
             }
             
             if (!reg.empty()) {
                 output << "    mov " << reg << ", [rbp - " << 
                     getOperandOffset(instruction->operands[i]) << "]\n";
             }
         }
         
         // Call the function
         output << "    call " << functionName << "\n";
         
         // Store the return value
         if (instruction->result) {
             output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
         }
         
         // Restore caller-saved registers
         output << "    pop r11\n";
         output << "    pop r10\n";
         output << "    pop r9\n";
         output << "    pop r8\n";
         output << "    pop rdi\n";
         output << "    pop rsi\n";
         output << "    pop rdx\n";
         output << "    pop rcx\n";
     }
     
     /**
      * @brief Generate assembly code for a RET instruction
      * @param instruction The RET instruction
      */
     void generateRet(std::shared_ptr<Instruction> instruction) {
         // If we have an operand, it's the return value
         if (instruction->operands.size() >= 1) {
             output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         }
         
         // Function epilogue and return
         output << "    mov rsp, rbp\n";
         output << "    pop rbp\n";
         output << "    ret\n";
     }
     
     /**
      * @brief Generate assembly code for a PHI instruction
      * @param instruction The PHI instruction
      */
     void generatePhi(std::shared_ptr<Instruction> instruction) {
         // PHI instructions are handled during basic block generation
         // For simplicity, we'll just set the result to the first operand's value
         
         if (instruction->operands.size() >= 1) {
             output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
             output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
         }
     }
     
     /**
      * @brief Generate assembly code for a CAST instruction
      * @param instruction The CAST instruction
      */
     void generateCast(std::shared_ptr<Instruction> instruction) {
         // Ensure we have at least one operand
         if (instruction->operands.size() < 1) {
             errorReporter.reportError("CAST instruction requires at least one operand", SourceLocation());
             return;
         }
         
         // Load the value to cast
         output << "    mov rax, [rbp - " << getOperandOffset(instruction->operands[0]) << "]\n";
         
         // For now, we'll just copy the value
         // In a real compiler, we would handle different types of casts
         
         // Store the result
         output << "    mov [rbp - " << getResultOffset(instruction->result) << "], rax\n";
     }
     
     /**
      * @brief Get the stack offset for an operand
      * @param operand The operand
      * @return The stack offset
      */
     size_t getOperandOffset(std::shared_ptr<Value> operand) {
         // In a real compiler, we would track variable locations
         // For now, just use a map from operand addresses to stack offsets
         
         std::string key = std::to_string(reinterpret_cast<uintptr_t>(operand.get()));
         
         auto it = localVars.find(key);
         if (it != localVars.end()) {
             return it->second;
         }
         
         // If we don't have an offset for this operand, allocate one
         stackSize += 8;
         localVars[key] = stackSize;
         
         return stackSize;
     }
     
     /**
      * @brief Get the stack offset for a result
      * @param result The result
      * @return The stack offset
      */
     size_t getResultOffset(std::shared_ptr<Value> result) {
         return getOperandOffset(result); // Same as operand offset
     }
 };
 
 /**
  * @brief Intermediate representation for code generation and optimization
  */
 class Value {
 public:
     enum class ValueType {
         INTEGER, FLOAT, BOOLEAN, POINTER, ARRAY, STRUCT, FUNCTION
     };
     
     ValueType type;
     
     Value(ValueType type) : type(type) {}
     virtual ~Value() = default;
 };
 
 /**
  * @brief Instruction in the intermediate representation
  */
 class Instruction {
 public:
     enum class OpCode {
         // Arithmetic
         ADD, SUB, MUL, DIV, MOD,
         // Bitwise
         AND, OR, XOR, SHL, SHR,
         // Comparison
         EQ, NE, LT, LE, GT, GE,
         // Memory
         ALLOCA, LOAD, STORE,
         // Control flow
         BR, BR_COND, CALL, RET,
         // Other
         PHI, CAST
     };
     
     OpCode opcode;
     std::vector<std::shared_ptr<Value>> operands;
     std::shared_ptr<Value> result;
     
     Instruction(OpCode opcode) : opcode(opcode) {}
     
     void addOperand(std::shared_ptr<Value> operand) {
         operands.push_back(operand);
     }
     
     void setResult(std::shared_ptr<Value> result) {
         this->result = result;
     }
 };
 
 /**
  * @brief Basic block in the intermediate representation
  */
 class BasicBlock {
 public:
     std::string label;
     std::vector<std::shared_ptr<Instruction>> instructions;
     std::vector<std::shared_ptr<BasicBlock>> predecessors;
     std::vector<std::shared_ptr<BasicBlock>> successors;
     
     BasicBlock(const std::string& label) : label(label) {}
     
     void addInstruction(std::shared_ptr<Instruction> instruction) {
         instructions.push_back(instruction);
     }
     
     void addPredecessor(std::shared_ptr<BasicBlock> predecessor) {
         predecessors.push_back(predecessor);
     }
     
     void addSuccessor(std::shared_ptr<BasicBlock> successor) {
         successors.push_back(successor);
     }
 };
 
 /**
  * @brief Function in the intermediate representation
  */
 class Function {
 public:
     std::string name;
     std::shared_ptr<FunctionType> type;
     std::vector<std::shared_ptr<BasicBlock>> blocks;
     
     Function(const std::string& name, std::shared_ptr<FunctionType> type)
         : name(name), type(type) {}
     
     void addBlock(std::shared_ptr<BasicBlock> block) {
         blocks.push_back(block);
     }
 };
 
 /**
  * @brief Module representing a compilation unit
  */
 class Module {
 public:
     std::string name;
     std::vector<std::shared_ptr<Function>> functions;
     
     Module(const std::string& name) : name(name) {}
     
     void addFunction(std::shared_ptr<Function> function) {
         functions.push_back(function);
     }
 };
 
 /**
  * @brief Optimizer for the intermediate representation
  * 
  * Time Complexity: O(n) where n is the number of instructions
  * Space Complexity: O(n) for storing the optimized code
  */
 class Optimizer {
 private:
     ErrorReporter& errorReporter;
     
 public:
     Optimizer(ErrorReporter& errorReporter) : errorReporter(errorReporter) {}
     
     /**
      * @brief Optimize a module
      * @param module The module to optimize
      */
     void optimize(std::shared_ptr<Module> module) {
         for (auto& function : module->functions) {
             optimizeFunction(function);
         }
     }
     
 private:
     /**
      * @brief Optimize a function
      * @param function The function to optimize
      */
     void optimizeFunction(std::shared_ptr<Function> function) {
         // Apply various optimization passes
         eliminateDeadCode(function);
         constantFolding(function);
         constantPropagation(function);
         simplifyControlFlow(function);
     }
     
     /**
      * @brief Eliminate dead code in a function
      * @param function The function to optimize
      */
     void eliminateDeadCode(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Mark all instructions as potentially dead
             std::unordered_map<std::shared_ptr<Instruction>, bool> isLive;
             
             // First pass: mark instructions with side effects as live
             for (auto& block : function->blocks) {
                 for (auto& instruction : block->instructions) {
                     // Instructions with side effects
                     if (instruction->opcode == Instruction::OpCode::STORE ||
                         instruction->opcode == Instruction::OpCode::CALL ||
                         instruction->opcode == Instruction::OpCode::RET) {
                         isLive[instruction] = true;
                     } else {
                         isLive[instruction] = false;
                     }
                 }
             }
             
             // Second pass: propagate liveness backward
             bool localChanged = true;
             while (localChanged) {
                 localChanged = false;
                 
                 for (auto& block : function->blocks) {
                     for (auto& instruction : block->instructions) {
                         if (isLive[instruction]) continue;
                         
                         // Check if this instruction's result is used by a live instruction
                         bool used = false;
                         
                         for (auto& otherBlock : function->blocks) {
                             for (auto& otherInstruction : otherBlock->instructions) {
                                 if (!isLive[otherInstruction]) continue;
                                 
                                 for (auto& operand : otherInstruction->operands) {
                                     if (operand == instruction->result) {
                                         used = true;
                                         break;
                                     }
                                 }
                                 
                                 if (used) break;
                             }
                             
                             if (used) break;
                         }
                         
                         if (used) {
                             isLive[instruction] = true;
                             localChanged = true;
                             changed = true;
                         }
                     }
                 }
             }
             
             // Third pass: remove dead instructions
             for (auto& block : function->blocks) {
                 auto it = block->instructions.begin();
                 while (it != block->instructions.end()) {
                     if (!isLive[*it]) {
                         it = block->instructions.erase(it);
                         changed = true;
                     } else {
                         ++it;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform constant folding in a function
      * @param function The function to optimize
      */
     void constantFolding(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             for (auto& block : function->blocks) {
                 auto it = block->instructions.begin();
                 while (it != block->instructions.end()) {
                     auto& instruction = *it;
                     
                     // Check if all operands are constants
                     bool allConstant = true;
                     
                     // This is a simplified version that doesn't actually check for constants
                     // In a real optimizer, we would have a way to identify constant values
                     
                     if (allConstant) {
                         // Evaluate the instruction at compile time
                         // In a real optimizer, we would actually compute the result
                         
                         // Replace uses of this instruction with the constant result
                         // In a real optimizer, we would update all uses
                         
                         // Remove the instruction
                         it = block->instructions.erase(it);
                         changed = true;
                     } else {
                         ++it;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform constant propagation in a function
      * @param function The function to optimize
      */
     void constantPropagation(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Map of values to their constant values (if known)
             std::unordered_map<std::shared_ptr<Value>, std::shared_ptr<Value>> constants;
             
             // First pass: identify constant values
             for (auto& block : function->blocks) {
                 for (auto& instruction : block->instructions) {
                     // Check for constant assignment instructions
                     if (instruction->opcode == Instruction::OpCode::ALLOCA && 
                         instruction->operands.size() >= 1) {
                         
                         // Check if the operand is a constant value
                         // In a real optimizer, we would have proper constant detection
                         if (instruction->operands[0]->type == Value::ValueType::INTEGER ||
                             instruction->operands[0]->type == Value::ValueType::FLOAT ||
                             instruction->operands[0]->type == Value::ValueType::BOOLEAN) {
                             constants[instruction->result] = instruction->operands[0];
                         }
                     }
                 }
             }
             
             // Second pass: propagate constants
             for (auto& block : function->blocks) {
                 for (auto& instruction : block->instructions) {
                     // Replace operands with constants if known
                     for (size_t i = 0; i < instruction->operands.size(); i++) {
                         auto it = constants.find(instruction->operands[i]);
                         if (it != constants.end()) {
                             instruction->operands[i] = it->second;
                             changed = true;
                         }
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform common subexpression elimination in a function
      * @param function The function to optimize
      */
     void commonSubexpressionElimination(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Map to track expressions that compute the same value
             std::unordered_map<std::string, std::shared_ptr<Value>> expressionMap;
             
             for (auto& block : function->blocks) {
                 // Reset the map for each basic block (local CSE)
                 expressionMap.clear();
                 
                 auto it = block->instructions.begin();
                 while (it != block->instructions.end()) {
                     auto& instruction = *it;
                     
                     // Skip instructions with side effects
                     if (instruction->opcode == Instruction::OpCode::STORE ||
                         instruction->opcode == Instruction::OpCode::CALL ||
                         instruction->opcode == Instruction::OpCode::RET) {
                         ++it;
                         continue;
                     }
                     
                     // Create a key for the instruction
                     std::string key = std::to_string(static_cast<int>(instruction->opcode));
                     for (auto& operand : instruction->operands) {
                         // In a real compiler, we would have a proper way to identify values
                         key += "_" + std::to_string(reinterpret_cast<uintptr_t>(operand.get()));
                     }
                     
                     // Check if we've seen this expression before
                     auto mapIt = expressionMap.find(key);
                     if (mapIt != expressionMap.end()) {
                         // Replace all uses of this instruction's result with the previous result
                         for (auto& otherBlock : function->blocks) {
                             for (auto& otherInstruction : otherBlock->instructions) {
                                 for (size_t i = 0; i < otherInstruction->operands.size(); i++) {
                                     if (otherInstruction->operands[i] == instruction->result) {
                                         otherInstruction->operands[i] = mapIt->second;
                                         changed = true;
                                     }
                                 }
                             }
                         }
                         
                         // Remove the redundant instruction
                         it = block->instructions.erase(it);
                     } else {
                         // Add this expression to the map
                         expressionMap[key] = instruction->result;
                         ++it;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform loop invariant code motion in a function
      * @param function The function to optimize
      */
     void loopInvariantCodeMotion(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Identify loops in the function
             std::vector<std::vector<std::shared_ptr<BasicBlock>>> loops;
             identifyLoops(function, loops);
             
             // For each loop
             for (auto& loop : loops) {
                 if (loop.empty()) continue;
                 
                 // Find the loop header
                 auto header = loop[0];
                 
                 // Find loop pre-header (entry block to the loop)
                 std::shared_ptr<BasicBlock> preHeader = nullptr;
                 for (auto& block : function->blocks) {
                     if (std::find(block->successors.begin(), block->successors.end(), header) != block->successors.end() &&
                         std::find(loop.begin(), loop.end(), block) == loop.end()) {
                         preHeader = block;
                         break;
                     }
                 }
                 
                 if (!preHeader) {
                     // Create a pre-header if it doesn't exist
                     preHeader = std::make_shared<BasicBlock>("loop_preheader");
                     
                     // Update the function's blocks
                     auto it = std::find(function->blocks.begin(), function->blocks.end(), header);
                     if (it != function->blocks.end()) {
                         function->blocks.insert(it, preHeader);
                     } else {
                         function->blocks.push_back(preHeader);
                     }
                     
                     // Update the CFG
                     for (auto& block : function->blocks) {
                         auto it = std::find(block->successors.begin(), block->successors.end(), header);
                         if (it != block->successors.end() && std::find(loop.begin(), loop.end(), block) == loop.end()) {
                             // Replace header with pre-header in the successor list
                             *it = preHeader;
                             
                             // Add block to pre-header's predecessors
                             preHeader->predecessors.push_back(block);
                         }
                     }
                     
                     // Add header to pre-header's successors
                     preHeader->successors.push_back(header);
                     
                     // Add pre-header to header's predecessors
                     auto it = std::find(header->predecessors.begin(), header->predecessors.end(), preHeader);
                     if (it == header->predecessors.end()) {
                         header->predecessors.push_back(preHeader);
                     }
                     
                     // Generate a branch instruction to the header
                     auto brInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     brInst->addOperand(std::make_shared<Value>(Value::ValueType::POINTER)); // dummy operand
                     preHeader->addInstruction(brInst);
                 }
                 
                 // Find loop-invariant instructions
                 std::vector<std::shared_ptr<Instruction>> invariantInsts;
                 
                 for (auto& block : loop) {
                     auto it = block->instructions.begin();
                     while (it != block->instructions.end()) {
                         auto& instruction = *it;
                         
                         // Skip instructions with side effects
                         if (instruction->opcode == Instruction::OpCode::STORE ||
                             instruction->opcode == Instruction::OpCode::CALL ||
                             instruction->opcode == Instruction::OpCode::RET) {
                             ++it;
                             continue;
                         }
                         
                         // Check if all operands are loop-invariant
                         bool isInvariant = true;
                         for (auto& operand : instruction->operands) {
                             // Check if the operand is defined outside the loop or is loop-invariant
                             bool operandInvariant = false;
                             
                             // In a real compiler, we would have proper data flow analysis
                             // For now, just assume all operands are not invariant
                             operandInvariant = false;
                             
                             if (!operandInvariant) {
                                 isInvariant = false;
                                 break;
                             }
                         }
                         
                         if (isInvariant) {
                             // Add to the list of invariant instructions
                             invariantInsts.push_back(instruction);
                             
                             // Move the instruction to the pre-header
                             preHeader->instructions.insert(preHeader->instructions.end() - 1, instruction);
                             
                             // Remove from the current block
                             it = block->instructions.erase(it);
                             changed = true;
                         } else {
                             ++it;
                         }
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Inline small functions into their call sites
      * @param module The module to optimize
      */
     void inlineFunctions(std::shared_ptr<Module> module) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Find candidate functions for inlining
             std::unordered_set<std::shared_ptr<Function>> inlineCandidates;
             
             for (auto& function : module->functions) {
                 // Skip the main function
                 if (function->name == "main") continue;
                 
                 // Check if the function is small enough to inline
                 // This is a simplified heuristic; in a real compiler, we would consider
                 // function size, call frequency, etc.
                 if (function->blocks.size() <= 3) {
                     inlineCandidates.insert(function);
                 }
             }
             
             // For each function in the module
             for (auto& function : module->functions) {
                 // For each block in the function
                 for (auto& block : function->blocks) {
                     auto it = block->instructions.begin();
                     while (it != block->instructions.end()) {
                         auto& instruction = *it;
                         
                         // Check if it's a call instruction
                         if (instruction->opcode == Instruction::OpCode::CALL && 
                             instruction->operands.size() >= 1) {
                             
                             // Find the called function
                             std::shared_ptr<Function> calledFunction = nullptr;
                             
                             // In a real compiler, we would have a proper way to resolve function references
                             // For now, just assume we can't find the function
                             
                             // Check if the function is a candidate for inlining
                             if (calledFunction && inlineCandidates.find(calledFunction) != inlineCandidates.end()) {
                                 // Perform function inlining
                                 // In a real compiler, we would clone the callee's body, rewrite variable
                                 // references, handle returns, etc.
                                 
                                 // Mark as changed
                                 changed = true;
                             } else {
                                 ++it;
                             }
                         } else {
                             ++it;
                         }
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Simplify control flow in a function
      * @param function The function to optimize
      */
     void simplifyControlFlow(std::shared_ptr<Function> function) {
         bool changed = true;
         
         while (changed) {
             changed = false;
             
             // Remove empty blocks
             auto blockIt = function->blocks.begin();
             while (blockIt != function->blocks.end()) {
                 auto& block = *blockIt;
                 
                 // Skip blocks with instructions
                 if (!block->instructions.empty()) {
                     ++blockIt;
                     continue;
                 }
                 
                 // Skip blocks with multiple predecessors or successors
                 if (block->predecessors.size() != 1 || block->successors.size() != 1) {
                     ++blockIt;
                     continue;
                 }
                 
                 // Get the predecessor and successor
                 auto pred = block->predecessors[0];
                 auto succ = block->successors[0];
                 
                 // Update the CFG
                 auto it = std::find(pred->successors.begin(), pred->successors.end(), block);
                 if (it != pred->successors.end()) {
                     *it = succ;
                 }
                 
                 it = std::find(succ->predecessors.begin(), succ->predecessors.end(), block);
                 if (it != succ->predecessors.end()) {
                     *it = pred;
                 }
                 
                 // Remove the empty block
                 blockIt = function->blocks.erase(blockIt);
                 changed = true;
             }
             
             // Merge blocks with a single predecessor and successor
             blockIt = function->blocks.begin();
             while (blockIt != function->blocks.end()) {
                 auto& block = *blockIt;
                 
                 // Skip blocks with multiple predecessors or successors
                 if (block->predecessors.size() != 1 || block->successors.size() != 1) {
                     ++blockIt;
                     continue;
                 }
                 
                 // Get the predecessor
                 auto pred = block->predecessors[0];
                 
                 // Skip if the predecessor has multiple successors
                 if (pred->successors.size() != 1) {
                     ++blockIt;
                     continue;
                 }
                 
                 // Merge the blocks
                 // Move instructions from block to the end of pred
                 pred->instructions.insert(pred->instructions.end(),
                                           block->instructions.begin(),
                                           block->instructions.end());
                 
                 // Update pred's successors
                 pred->successors = block->successors;
                 
                 // Update the successors' predecessors
                 for (auto& succ : block->successors) {
                     auto it = std::find(succ->predecessors.begin(), succ->predecessors.end(), block);
                     if (it != succ->predecessors.end()) {
                         *it = pred;
                     }
                 }
                 
                 // Remove the merged block
                 blockIt = function->blocks.erase(blockIt);
                 changed = true;
             }
         }
     }
     
     /**
      * @brief Identify loops in a function
      * @param function The function
      * @param loops Output parameter to store the identified loops
      */
     void identifyLoops(std::shared_ptr<Function> function, 
                        std::vector<std::vector<std::shared_ptr<BasicBlock>>>& loops) {
         // Build a map of dominators
         std::unordered_map<std::shared_ptr<BasicBlock>, std::unordered_set<std::shared_ptr<BasicBlock>>> dominators;
         buildDominators(function, dominators);
         
         // For each block in the function
         for (auto& block : function->blocks) {
             // For each successor of the block
             for (auto& succ : block->successors) {
                 // If the successor dominates the block, it's a back edge
                 if (dominators[block].find(succ) != dominators[block].end()) {
                     // Identify the loop
                     std::vector<std::shared_ptr<BasicBlock>> loop;
                     loop.push_back(succ); // Header
                     
                     // Add all blocks in the loop
                     std::function<void(std::shared_ptr<BasicBlock>)> addToLoop =
                         [&](std::shared_ptr<BasicBlock> current) {
                             if (current != succ && 
                                 std::find(loop.begin(), loop.end(), current) == loop.end()) {
                                 loop.push_back(current);
                                 
                                 for (auto& pred : current->predecessors) {
                                     addToLoop(pred);
                                 }
                             }
                         };
                     
                     addToLoop(block);
                     
                     loops.push_back(loop);
                 }
             }
         }
     }
     
     /**
      * @brief Build the dominator sets for a function
      * @param function The function
      * @param dominators Output parameter to store the dominators
      */
     void buildDominators(std::shared_ptr<Function> function,
                          std::unordered_map<std::shared_ptr<BasicBlock>, 
                                            std::unordered_set<std::shared_ptr<BasicBlock>>>& dominators) {
         // Initialize all blocks to be dominated by all blocks
         for (auto& block : function->blocks) {
             dominators[block] = std::unordered_set<std::shared_ptr<BasicBlock>>();
             
             // Add all blocks as potential dominators, except for the entry block
             for (auto& otherBlock : function->blocks) {
                 if (block != function->blocks[0]) {
                     dominators[block].insert(otherBlock);
                 }
             }
         }
         
         // The entry block is only dominated by itself
         dominators[function->blocks[0]].clear();
         dominators[function->blocks[0]].insert(function->blocks[0]);
         
         bool changed = true;
         while (changed) {
             changed = false;
             
             // For each block (except the entry block)
             for (size_t i = 1; i < function->blocks.size(); i++) {
                 auto block = function->blocks[i];
                 std::unordered_set<std::shared_ptr<BasicBlock>> newDominators = dominators[block];
                 
                 // Intersect the dominators of all predecessors
                 for (auto& pred : block->predecessors) {
                     std::unordered_set<std::shared_ptr<BasicBlock>> intersection;
                     
                     for (auto& dom : dominators[pred]) {
                         if (dominators[block].find(dom) != dominators[block].end()) {
                             intersection.insert(dom);
                         }
                     }
                     
                     newDominators = intersection;
                 }
                 
                 // Add the block itself to its dominators
                 newDominators.insert(block);
                 
                 // If the dominators changed, we need to continue iterating
                 if (newDominators != dominators[block]) {
                     dominators[block] = newDominators;
                     changed = true;
                 }
             }
         }
     }
 };
 /**
  * @brief The semantic analyzer performs type checking and other semantic validations
  * 
  * Time Complexity: O(n) where n is the number of nodes in the AST
  * Space Complexity: O(d) where d is the maximum depth of the AST
  */
 class SemanticAnalyzer {
 private:
     std::shared_ptr<SymbolTable> symbolTable;
     ErrorReporter& errorReporter;
     
 public:
     SemanticAnalyzer(ErrorReporter& errorReporter)
         : errorReporter(errorReporter) {
         symbolTable = std::make_shared<SymbolTable>();
     }
     
     /**
      * @brief Analyze the AST for semantic correctness
      * @param root The root node of the AST
      * @return True if semantic analysis was successful, false otherwise
      */
     bool analyze(const std::unique_ptr<ASTNode>& root) {
         try {
             // Define built-in types
             defineBuiltInTypes();
             
             // Visit the AST
             visitNode(root);
             
             return !errorReporter.hadError();
         } catch (const std::exception& e) {
             errorReporter.reportError(e.what(), SourceLocation());
             return false;
         }
     }
 
 private:
     /**
      * @brief Define built-in types in the symbol table
      */
     void defineBuiltInTypes() {
         // Define primitive types
         symbolTable->defineType("void", std::make_shared<Type>(Type::TypeKind::VOID), SourceLocation());
         symbolTable->defineType("bool", std::make_shared<Type>(Type::TypeKind::BOOL), SourceLocation());
         symbolTable->defineType("char", std::make_shared<Type>(Type::TypeKind::CHAR), SourceLocation());
         symbolTable->defineType("int", std::make_shared<Type>(Type::TypeKind::INT), SourceLocation());
         symbolTable->defineType("float", std::make_shared<Type>(Type::TypeKind::FLOAT), SourceLocation());
         symbolTable->defineType("double", std::make_shared<Type>(Type::TypeKind::DOUBLE), SourceLocation());
         symbolTable->defineType("auto", std::make_shared<Type>(Type::TypeKind::AUTO), SourceLocation());
     }
     
     /**
      * @brief Visit a node in the AST
      * @param node The node to visit
      * @return The type of the node
      */
     std::shared_ptr<Type> visitNode(const std::unique_ptr<ASTNode>& node) {
         if (!node) return nullptr;
         
         switch (node->type) {
             case ASTNodeType::PROGRAM:
                 return visitProgram(static_cast<const ProgramNode&>(*node));
             case ASTNodeType::VARIABLE_DECL:
                 return visitVariableDecl(static_cast<const VariableDeclNode&>(*node));
             case ASTNodeType::FUNCTION_DECL:
                 return visitFunctionDecl(static_cast<const FunctionDeclNode&>(*node));
             case ASTNodeType::COMPOUND_STMT:
                 return visitCompoundStmt(static_cast<const CompoundStmtNode&>(*node));
             case ASTNodeType::EXPRESSION_STMT:
                 return visitExpressionStmt(static_cast<const ExpressionStmtNode&>(*node));
             case ASTNodeType::IF_STMT:
                 return visitIfStmt(static_cast<const IfStmtNode&>(*node));
             case ASTNodeType::WHILE_STMT:
                 return visitWhileStmt(static_cast<const WhileStmtNode&>(*node));
             case ASTNodeType::FOR_STMT:
                 return visitForStmt(static_cast<const ForStmtNode&>(*node));
             case ASTNodeType::RETURN_STMT:
                 return visitReturnStmt(static_cast<const ReturnStmtNode&>(*node));
             case ASTNodeType::BINARY_EXPR:
                 return visitBinaryExpr(static_cast<const BinaryExprNode&>(*node));
             case ASTNodeType::UNARY_EXPR:
                 return visitUnaryExpr(static_cast<const UnaryExprNode&>(*node));
             case ASTNodeType::LITERAL_EXPR:
                 return visitLiteralExpr(static_cast<const LiteralExprNode&>(*node));
             case ASTNodeType::VARIABLE_EXPR:
                 return visitVariableExpr(static_cast<const VariableExprNode&>(*node));
             case ASTNodeType::ASSIGNMENT_EXPR:
                 return visitAssignmentExpr(static_cast<const AssignmentExprNode&>(*node));
             case ASTNodeType::CALL_EXPR:
                 return visitCallExpr(static_cast<const CallExprNode&>(*node));
             default:
                 // For nodes that don't contribute to type checking
                 return nullptr;
         }
     }
     
     /**
      * @brief Visit a program node
      * @param node The program node
      * @return nullptr (program nodes don't have a type)
      */
     std::shared_ptr<Type> visitProgram(const ProgramNode& node) {
         for (const auto& declaration : node.declarations) {
             visitNode(declaration);
         }
         return nullptr;
     }
     
     /**
      * @brief Visit a variable declaration node
      * @param node The variable declaration node
      * @return The type of the variable
      */
     std::shared_ptr<Type> visitVariableDecl(const VariableDeclNode& node) {
         std::shared_ptr<Type> initializerType = nullptr;
         
         if (node.initializer) {
             initializerType = visitNode(node.initializer);
             
             // Check if the initializer type is compatible with the variable type
             if (initializerType && !initializerType->isCompatibleWith(*node.type)) {
                 errorReporter.reportError(
                     std::format("Cannot initialize variable of type '{}' with value of type '{}'",
                                node.type->toString(), initializerType->toString()),
                     node.location
                 );
             }
         }
         
         // Add the variable to the symbol table
         if (!symbolTable->defineVariable(node.name, node.type, node.isConst, node.location)) {
             errorReporter.reportError(
                 std::format("Variable '{}' already defined", node.name),
                 node.location
             );
         }
         
         return node.type;
     }
     
     /**
      * @brief Visit a function declaration node
      * @param node The function declaration node
      * @return The type of the function
      */
     std::shared_ptr<Type> visitFunctionDecl(const FunctionDeclNode& node) {
         // Add the function to the symbol table
         if (!symbolTable->defineFunction(node.name, node.type, node.isInline, node.isVirtual, node.location)) {
             errorReporter.reportError(
                 std::format("Function '{}' already defined", node.name),
                 node.location
             );
         }
         
         // Create a new scope for the function body
         auto functionScope = std::make_shared<SymbolTable>(symbolTable);
         auto outerScope = symbolTable;
         symbolTable = functionScope;
         
         // Add parameters to the function's scope
         for (size_t i = 0; i < node.parameters.size(); i++) {
             // For simplicity, assume parameters are of the form TYPE NAME
             // In a real compiler, we would properly extract this information
             auto paramType = node.type->parameterTypes[i];
             std::string paramName = "param" + std::to_string(i); // Simplified
             
             symbolTable->defineVariable(paramName, paramType, false, node.location);
         }
         
         // Visit the function body
         if (node.body) {
             visitNode(node.body);
         }
         
         // Restore the outer scope
         symbolTable = outerScope;
         
         return node.type;
     }
     
     /**
      * @brief Visit a compound statement node
      * @param node The compound statement node
      * @return nullptr (compound statements don't have a type)
      */
     std::shared_ptr<Type> visitCompoundStmt(const CompoundStmtNode& node) {
         // Create a new scope for the compound statement
         auto blockScope = std::make_shared<SymbolTable>(symbolTable);
         auto outerScope = symbolTable;
         symbolTable = blockScope;
         
         for (const auto& statement : node.statements) {
             visitNode(statement);
         }
         
         // Restore the outer scope
         symbolTable = outerScope;
         
         return nullptr;
     }
     
     /**
      * @brief Visit an expression statement node
      * @param node The expression statement node
      * @return The type of the expression
      */
     std::shared_ptr<Type> visitExpressionStmt(const ExpressionStmtNode& node) {
         return visitNode(node.expression);
     }
     
     /**
      * @brief Visit an if statement node
      * @param node The if statement node
      * @return nullptr (if statements don't have a type)
      */
     std::shared_ptr<Type> visitIfStmt(const IfStmtNode& node) {
         auto conditionType = visitNode(node.condition);
         
         // Check if the condition is a boolean expression
         if (conditionType && conditionType->kind != Type::TypeKind::BOOL) {
             errorReporter.reportError(
                 std::format("Condition must be a boolean expression, got '{}'", 
                            conditionType->toString()),
                 node.location
             );
         }
         
         visitNode(node.thenBranch);
         
         if (node.elseBranch) {
             visitNode(node.elseBranch);
         }
         
         return nullptr;
     }
     
     /**
      * @brief Visit a while statement node
      * @param node The while statement node
      * @return nullptr (while statements don't have a type)
      */
     std::shared_ptr<Type> visitWhileStmt(const WhileStmtNode& node) {
         auto conditionType = visitNode(node.condition);
         
         // Check if the condition is a boolean expression
         if (conditionType && conditionType->kind != Type::TypeKind::BOOL) {
             errorReporter.reportError(
                 std::format("Condition must be a boolean expression, got '{}'", 
                            conditionType->toString()),
                 node.location
             );
         }
         
         visitNode(node.body);
         
         return nullptr;
     }
     
     /**
      * @brief Visit a for statement node
      * @param node The for statement node
      * @return nullptr (for statements don't have a type)
      */
     std::shared_ptr<Type> visitForStmt(const ForStmtNode& node) {
         // Create a new scope for the for statement
         auto forScope = std::make_shared<SymbolTable>(symbolTable);
         auto outerScope = symbolTable;
         symbolTable = forScope;
         
         if (node.initializer) {
             visitNode(node.initializer);
         }
         
         if (node.condition) {
             auto conditionType = visitNode(node.condition);
             
             // Check if the condition is a boolean expression
             if (conditionType && conditionType->kind != Type::TypeKind::BOOL) {
                 errorReporter.reportError(
                     std::format("Condition must be a boolean expression, got '{}'", 
                                conditionType->toString()),
                     node.location
                 );
             }
         }
         
         if (node.increment) {
             visitNode(node.increment);
         }
         
         visitNode(node.body);
         
         // Restore the outer scope
         symbolTable = outerScope;
         
         return nullptr;
     }
     
     /**
      * @brief Visit a return statement node
      * @param node The return statement node
      * @return nullptr (return statements don't have a type)
      */
     std::shared_ptr<Type> visitReturnStmt(const ReturnStmtNode& node) {
         // In a full implementation, we would check if the return type matches the function's return type
         if (node.value) {
             visitNode(node.value);
         }
         
         return nullptr;
     }
     
     /**
      * @brief Visit a binary expression node
      * @param node The binary expression node
      * @return The type of the binary expression
      */
     std::shared_ptr<Type> visitBinaryExpr(const BinaryExprNode& node) {
         auto leftType = visitNode(node.left);
         auto rightType = visitNode(node.right);
         
         if (!leftType || !rightType) {
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Type checking for binary operators
         switch (node.op) {
             case BinaryExprNode::Operator::ADD:
             case BinaryExprNode::Operator::SUBTRACT:
             case BinaryExprNode::Operator::MULTIPLY:
             case BinaryExprNode::Operator::DIVIDE:
             case BinaryExprNode::Operator::MODULO:
                 // Arithmetic operators require numeric operands
                 if (!leftType->isNumeric() || !rightType->isNumeric()) {
                     errorReporter.reportError(
                         std::format("Arithmetic operator requires numeric operands, got '{}' and '{}'",
                                    leftType->toString(), rightType->toString()),
                         node.location
                     );
                 }
                 
                 // If either operand is floating-point, the result is floating-point
                 if (leftType->isFloatingPoint() || rightType->isFloatingPoint()) {
                     return std::make_shared<Type>(Type::TypeKind::DOUBLE);
                 } else {
                     return std::make_shared<Type>(Type::TypeKind::INT);
                 }
                 
             case BinaryExprNode::Operator::EQUAL:
             case BinaryExprNode::Operator::NOT_EQUAL:
                 // Equality operators can compare any types, but they must be compatible
                 if (!leftType->isCompatibleWith(*rightType)) {
                     errorReporter.reportError(
                         std::format("Cannot compare '{}' and '{}'",
                                    leftType->toString(), rightType->toString()),
                         node.location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::LESS:
             case BinaryExprNode::Operator::LESS_EQUAL:
             case BinaryExprNode::Operator::GREATER:
             case BinaryExprNode::Operator::GREATER_EQUAL:
                 // Comparison operators require numeric operands
                 if (!leftType->isNumeric() || !rightType->isNumeric()) {
                     errorReporter.reportError(
                         std::format("Comparison operator requires numeric operands, got '{}' and '{}'",
                                    leftType->toString(), rightType->toString()),
                         node.location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::AND:
             case BinaryExprNode::Operator::OR:
                 // Logical operators require boolean operands
                 if (leftType->kind != Type::TypeKind::BOOL || rightType->kind != Type::TypeKind::BOOL) {
                     errorReporter.reportError(
                         std::format("Logical operator requires boolean operands, got '{}' and '{}'",
                                    leftType->toString(), rightType->toString()),
                         node.location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::BITWISE_AND:
             case BinaryExprNode::Operator::BITWISE_OR:
             case BinaryExprNode::Operator::BITWISE_XOR:
             case BinaryExprNode::Operator::LEFT_SHIFT:
             case BinaryExprNode::Operator::RIGHT_SHIFT:
                 // Bitwise operators require integer operands
                 if (!leftType->isInteger() || !rightType->isInteger()) {
                     errorReporter.reportError(
                         std::format("Bitwise operator requires integer operands, got '{}' and '{}'",
                                    leftType->toString(), rightType->toString()),
                         node.location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::INT);
                 
             default:
                 errorReporter.reportError("Unknown binary operator", node.location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a unary expression node
      * @param node The unary expression node
      * @return The type of the unary expression
      */
     std::shared_ptr<Type> visitUnaryExpr(const UnaryExprNode& node) {
         auto operandType = visitNode(node.operand);
         
         if (!operandType) {
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Type checking for unary operators
         switch (node.op) {
             case UnaryExprNode::Operator::NEGATE:
                 // Negation requires a numeric operand
                 if (!operandType->isNumeric()) {
                     errorReporter.reportError(
                         std::format("Unary negation requires a numeric operand, got '{}'",
                                    operandType->toString()),
                         node.location
                     );
                 }
                 return operandType;
                 
             case UnaryExprNode::Operator::NOT:
                 // Logical NOT requires a boolean operand
                 if (operandType->kind != Type::TypeKind::BOOL) {
                     errorReporter.reportError(
                         std::format("Logical NOT requires a boolean operand, got '{}'",
                                    operandType->toString()),
                         node.location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case UnaryExprNode::Operator::BITWISE_NOT:
                 // Bitwise NOT requires an integer operand
                 if (!operandType->isInteger()) {
                     errorReporter.reportError(
                         std::format("Bitwise NOT requires an integer operand, got '{}'",
                                    operandType->toString()),
                         node.location
                     );
                 }
                 return operandType;
                 
             case UnaryExprNode::Operator::ADDRESS_OF:
                 // Address-of operator returns a pointer to the operand's type
                 return std::make_shared<PointerType>(operandType);
                 
             case UnaryExprNode::Operator::DEREFERENCE:
                 // Dereference operator requires a pointer operand
                 if (operandType->kind != Type::TypeKind::POINTER) {
                     errorReporter.reportError(
                         std::format("Dereference operator requires a pointer operand, got '{}'",
                                    operandType->toString()),
                         node.location
                     );
                     return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
                 }
                 
                 return static_cast<PointerType*>(operandType.get())->baseType;
                 
             case UnaryExprNode::Operator::PRE_INCREMENT:
             case UnaryExprNode::Operator::PRE_DECREMENT:
             case UnaryExprNode::Operator::POST_INCREMENT:
             case UnaryExprNode::Operator::POST_DECREMENT:
                 // Increment and decrement operators require a numeric operand
                 if (!operandType->isNumeric()) {
                     errorReporter.reportError(
                         std::format("Increment/decrement operator requires a numeric operand, got '{}'",
                                    operandType->toString()),
                         node.location
                     );
                 }
                 return operandType;
                 
             default:
                 errorReporter.reportError("Unknown unary operator", node.location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a literal expression node
      * @param node The literal expression node
      * @return The type of the literal
      */
     std::shared_ptr<Type> visitLiteralExpr(const LiteralExprNode& node) {
         switch (node.literalType) {
             case LiteralExprNode::LiteralType::INTEGER:
                 return std::make_shared<Type>(Type::TypeKind::INT);
             case LiteralExprNode::LiteralType::FLOAT:
                 return std::make_shared<Type>(Type::TypeKind::DOUBLE);
             case LiteralExprNode::LiteralType::CHARACTER:
                 return std::make_shared<Type>(Type::TypeKind::CHAR);
             case LiteralExprNode::LiteralType::STRING:
                 return std::make_shared<ArrayType>(
                     std::make_shared<Type>(Type::TypeKind::CHAR),
                     node.value.length() - 2 + 1 // -2 for quotes, +1 for null terminator
                 );
             case LiteralExprNode::LiteralType::BOOLEAN:
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
             case LiteralExprNode::LiteralType::NULL_LITERAL:
                 return std::make_shared<PointerType>(std::make_shared<Type>(Type::TypeKind::VOID));
             default:
                 errorReporter.reportError("Unknown literal type", node.location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a variable expression node
      * @param node The variable expression node
      * @return The type of the variable
      */
     std::shared_ptr<Type> visitVariableExpr(const VariableExprNode& node) {
         auto variable = symbolTable->resolveVariable(node.name);
         
         if (!variable) {
             errorReporter.reportError(
                 std::format("Undefined variable '{}'", node.name),
                 node.location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         return variable->type;
     }
     
     /**
      * @brief Visit an assignment expression node
      * @param node The assignment expression node
      * @return The type of the assigned value
      */
     std::shared_ptr<Type> visitAssignmentExpr(const AssignmentExprNode& node) {
         auto leftType = visitNode(node.left);
         auto rightType = visitNode(node.right);
         
         if (!leftType || !rightType) {
             return leftType ? leftType : rightType;
         }
         
         // Check if the left operand is an lvalue
         if (node.left->type != ASTNodeType::VARIABLE_EXPR && 
             node.left->type != ASTNodeType::MEMBER_ACCESS_EXPR &&
             node.left->type != ASTNodeType::ARRAY_ACCESS_EXPR) {
             errorReporter.reportError(
                 "Left-hand side of assignment must be an lvalue",
                 node.location
             );
         }
         
         // Check if the right operand is compatible with the left operand
         if (!rightType->isCompatibleWith(*leftType)) {
             errorReporter.reportError(
                 std::format("Cannot assign value of type '{}' to variable of type '{}'",
                            rightType->toString(), leftType->toString()),
                 node.location
             );
         }
         
         return leftType;
     }
     
     /**
      * @brief Visit a function call expression node
      * @param node The function call expression node
      * @return The return type of the function
      */
     std::shared_ptr<Type> visitCallExpr(const CallExprNode& node) {
         // For now, we'll assume the callee is a variable expression (function name)
         if (node.callee->type != ASTNodeType::VARIABLE_EXPR) {
             errorReporter.reportError(
                 "Function call on non-function expression",
                 node.location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         const VariableExprNode& callee = static_cast<const VariableExprNode&>(*node.callee);
         auto function = symbolTable->resolveFunction(callee.name);
         
         if (!function) {
             errorReporter.reportError(
                 std::format("Undefined function '{}'", callee.name),
                 node.location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Check argument count
         if (node.arguments.size() != function->type->parameterTypes.size()) {
             errorReporter.reportError(
                 std::format("Function '{}' expects {} arguments, got {}",
                            callee.name, function->type->parameterTypes.size(), node.arguments.size()),
                 node.location
             );
         } else {
             // Check argument types
             for (size_t i = 0; i < node.arguments.size(); i++) {
                 auto argType = visitNode(node.arguments[i]);
                 
                 if (argType && !argType->isCompatibleWith(*function->type->parameterTypes[i])) {
                     errorReporter.reportError(
                         std::format("Function '{}' expects argument {} of type '{}', got '{}'",
                                    callee.name, i + 1, function->type->parameterTypes[i]->toString(),
                                    argType->toString()),
                         node.location
                     );
                 }
             }
         }
         
         return function->type->returnType;
     }
 };
 
 /**
  * @brief Converts a TokenType to a string for debugging and error reporting
  * @param type The token type to convert
  * @return A string representation of the token type
  */
 std::string tokenTypeToString(TokenType type) {
     static const std::unordered_map<TokenType, std::string> tokenNames = {
         {TokenType::INT, "INT"},
         {TokenType::CHAR, "CHAR"},
         {TokenType::BOOL, "BOOL"},
         {TokenType::FLOAT, "FLOAT"},
         {TokenType::DOUBLE, "DOUBLE"},
         {TokenType::VOID, "VOID"},
         {TokenType::AUTO, "AUTO"},
         {TokenType::STRUCT, "STRUCT"},
         {TokenType::CLASS, "CLASS"},
         {TokenType::ENUM, "ENUM"},
         {TokenType::UNION, "UNION"},
         {TokenType::TYPEDEF, "TYPEDEF"},
         {TokenType::CONST, "CONST"},
         {TokenType::STATIC, "STATIC"},
         {TokenType::EXTERN, "EXTERN"},
         {TokenType::INLINE, "INLINE"},
         {TokenType::VIRTUAL, "VIRTUAL"},
         {TokenType::OVERRIDE, "OVERRIDE"},
         {TokenType::FINAL, "FINAL"},
         {TokenType::PUBLIC, "PUBLIC"},
         {TokenType::PRIVATE, "PRIVATE"},
         {TokenType::PROTECTED, "PROTECTED"},
         {TokenType::IF, "IF"},
         {TokenType::ELSE, "ELSE"},
         {TokenType::WHILE, "WHILE"},
         {TokenType::FOR, "FOR"},
         {TokenType::DO, "DO"},
         {TokenType::SWITCH, "SWITCH"},
         {TokenType::CASE, "CASE"},
         {TokenType::DEFAULT, "DEFAULT"},
         {TokenType::BREAK, "BREAK"},
         {TokenType::CONTINUE, "CONTINUE"},
         {TokenType::RETURN, "RETURN"},
         {TokenType::NEW, "NEW"},
         {TokenType::DELETE, "DELETE"},
         {TokenType::TRY, "TRY"},
         {TokenType::CATCH, "CATCH"},
         {TokenType::THROW, "THROW"},
         {TokenType::NAMESPACE, "NAMESPACE"},
         {TokenType::USING, "USING"},
         {TokenType::TEMPLATE, "TEMPLATE"},
         {TokenType::TYPENAME, "TYPENAME"},
         {TokenType::PLUS, "PLUS"},
         {TokenType::MINUS, "MINUS"},
         {TokenType::ASTERISK, "ASTERISK"},
         {TokenType::SLASH, "SLASH"},
         {TokenType::PERCENT, "PERCENT"},
         {TokenType::AMPERSAND, "AMPERSAND"},
         {TokenType::PIPE, "PIPE"},
         {TokenType::CARET, "CARET"},
         {TokenType::TILDE, "TILDE"},
         {TokenType::EXCLAMATION, "EXCLAMATION"},
         {TokenType::LESS, "LESS"},
         {TokenType::GREATER, "GREATER"},
         {TokenType::EQUAL, "EQUAL"},
         {TokenType::DOT, "DOT"},
         {TokenType::ARROW, "ARROW"},
         {TokenType::PLUS_EQUAL, "PLUS_EQUAL"},
         {TokenType::MINUS_EQUAL, "MINUS_EQUAL"},
         {TokenType::ASTERISK_EQUAL, "ASTERISK_EQUAL"},
         {TokenType::SLASH_EQUAL, "SLASH_EQUAL"},
         {TokenType::PERCENT_EQUAL, "PERCENT_EQUAL"},
         {TokenType::AMPERSAND_EQUAL, "AMPERSAND_EQUAL"},
         {TokenType::PIPE_EQUAL, "PIPE_EQUAL"},
         {TokenType::CARET_EQUAL, "CARET_EQUAL"},
         {TokenType::LESS_LESS, "LESS_LESS"},
         {TokenType::GREATER_GREATER, "GREATER_GREATER"},
         {TokenType::LESS_LESS_EQUAL, "LESS_LESS_EQUAL"},
         {TokenType::GREATER_GREATER_EQUAL, "GREATER_GREATER_EQUAL"},
         {TokenType::EQUAL_EQUAL, "EQUAL_EQUAL"},
         {TokenType::EXCLAMATION_EQUAL, "EXCLAMATION_EQUAL"},
         {TokenType::LESS_EQUAL, "LESS_EQUAL"},
         {TokenType::GREATER_EQUAL, "GREATER_EQUAL"},
         {TokenType::AMPERSAND_AMPERSAND, "AMPERSAND_AMPERSAND"},
         {TokenType::PIPE_PIPE, "PIPE_PIPE"},
         {TokenType::PLUS_PLUS, "PLUS_PLUS"},
         {TokenType::MINUS_MINUS, "MINUS_MINUS"},
         {TokenType::COLON_COLON, "COLON_COLON"},
         {TokenType::LEFT_PAREN, "LEFT_PAREN"},
         {TokenType::RIGHT_PAREN, "RIGHT_PAREN"},
         {TokenType::LEFT_BRACKET, "LEFT_BRACKET"},
         {TokenType::RIGHT_BRACKET, "RIGHT_BRACKET"},
         {TokenType::LEFT_BRACE, "LEFT_BRACE"},
         {TokenType::RIGHT_BRACE, "RIGHT_BRACE"},
         {TokenType::SEMICOLON, "SEMICOLON"},
         {TokenType::COLON, "COLON"},
         {TokenType::COMMA, "COMMA"},
         {TokenType::QUESTION, "QUESTION"},
         {TokenType::IDENTIFIER, "IDENTIFIER"},
         {TokenType::INTEGER_LITERAL, "INTEGER_LITERAL"},
         {TokenType::FLOAT_LITERAL, "FLOAT_LITERAL"},
         {TokenType::CHAR_LITERAL, "CHAR_LITERAL"},
         {TokenType::STRING_LITERAL, "STRING_LITERAL"},
         {TokenType::BOOL_LITERAL, "BOOL_LITERAL"},
         {TokenType::COMMENT, "COMMENT"},
         {TokenType::PREPROCESSOR, "PREPROCESSOR"},
         {TokenType::END_OF_FILE, "END_OF_FILE"},
         {TokenType::ERROR, "ERROR"}
     };
     
     auto it = tokenNames.find(type);
     if (it != tokenNames.end()) {
         return it->second;
     } else {
         return "UNKNOWN_TOKEN";
     }
 }
 
 /**
  * @brief Location information for error reporting
  */
 struct SourceLocation {
     std::string filename;
     int line;
     int column;
     
     SourceLocation(const std::string& file = "", int l = 1, int c = 1)
         : filename(file), line(l), column(c) {}
     
     std::string toString() const {
         return std::format("{}:{}:{}", filename, line, column);
     }
 };
 
 /**
  * @brief Token class representing lexical units from the source code
  */
 class Token {
 public:
     TokenType type;
     std::string lexeme;
     SourceLocation location;
     
     Token(TokenType t, const std::string& lex, const SourceLocation& loc)
         : type(t), lexeme(lex), location(loc) {}
     
     std::string toString() const {
         return std::format("Token({}, '{}', {})", 
                           tokenTypeToString(type), 
                           lexeme, 
                           location.toString());
     }
 };
 
 /**
  * @brief Error handling class for reporting and tracking compilation errors
  */
 class ErrorReporter {
 private:
     std::vector<std::string> errors;
     std::vector<std::string> warnings;
     bool hasError = false;
     
     // Mutex for thread-safe error reporting
     // This is needed because error reporting might be called from different compilation stages
     // running in parallel, or from different threads processing different files
     mutable std::shared_mutex mutex;
 
 public:
     void reportError(const std::string& message, const SourceLocation& location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         std::string errorMsg = std::format("Error at {}: {}", location.toString(), message);
         errors.push_back(errorMsg);
         hasError = true;
         std::cerr << errorMsg << std::endl;
     }
     
     void reportWarning(const std::string& message, const SourceLocation& location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         std::string warningMsg = std::format("Warning at {}: {}", location.toString(), message);
         warnings.push_back(warningMsg);
         std::cerr << warningMsg << std::endl;
     }
     
     bool hadError() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return hasError;
     }
     
     const std::vector<std::string>& getErrors() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return errors;
     }
     
     const std::vector<std::string>& getWarnings() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return warnings;
     }
     
     void reset() {
         std::unique_lock lock(mutex); // Write lock for thread safety
         errors.clear();
         warnings.clear();
         hasError = false;
     }
 };
 
 /**
  * @brief Lexical analyzer to convert source code into tokens
  * 
  * Time Complexity: O(n) where n is the number of characters in the source code
  * Space Complexity: O(n) for storing tokens and internal buffers
  */
 class Lexer {
 private:
     std::string source;
     std::string filename;
     size_t start = 0;
     size_t current = 0;
     int line = 1;
     int column = 1;
     std::vector<Token> tokens;
     ErrorReporter& errorReporter;
     
     // Map of keywords to their corresponding token types
     static const std::unordered_map<std::string, TokenType> keywords;
 
 public:
     Lexer(const std::string& source, const std::string& filename, ErrorReporter& reporter)
         : source(source), filename(filename), errorReporter(reporter) {}
     
     /**
      * @brief Scan all tokens from the source code
      * @return Vector of tokens
      */
     std::vector<Token> scanTokens() {
         while (!isAtEnd()) {
             // Beginning of the next lexeme
             start = current;
             scanToken();
         }
         
         // Add EOF token
         tokens.emplace_back(TokenType::END_OF_FILE, "", SourceLocation(filename, line, column));
         return tokens;
     }
 
 private:
     /**
      * @brief Scan a single token from the source code
      */
     void scanToken() {
         char c = advance();
         
         switch (c) {
             // Single-character tokens
             case '(': addToken(TokenType::LEFT_PAREN); break;
             case ')': addToken(TokenType::RIGHT_PAREN); break;
             case '[': addToken(TokenType::LEFT_BRACKET); break;
             case ']': addToken(TokenType::RIGHT_BRACKET); break;
             case '{': addToken(TokenType::LEFT_BRACE); break;
             case '}': addToken(TokenType::RIGHT_BRACE); break;
             case ',': addToken(TokenType::COMMA); break;
             case '.': addToken(TokenType::DOT); break;
             case ';': addToken(TokenType::SEMICOLON); break;
             case '?': addToken(TokenType::QUESTION); break;
             case '~': addToken(TokenType::TILDE); break;
             
             // Operators that could be part of multi-character operators
             case '+': 
                 if (match('+')) addToken(TokenType::PLUS_PLUS);
                 else if (match('=')) addToken(TokenType::PLUS_EQUAL);
                 else addToken(TokenType::PLUS);
                 break;
                 
             case '-': 
                 if (match('>')) addToken(TokenType::ARROW);
                 else if (match('-')) addToken(TokenType::MINUS_MINUS);
                 else if (match('=')) addToken(TokenType::MINUS_EQUAL);
                 else addToken(TokenType::MINUS);
                 break;
                 
             case '*': 
                 if (match('=')) addToken(TokenType::ASTERISK_EQUAL);
                 else addToken(TokenType::ASTERISK);
                 break;
                 
             case '/': 
                 if (match('/')) {
                     // Single-line comment
                     while (peek() != '\n' && !isAtEnd()) advance();
                     // Don't add comment tokens for now
                 } else if (match('*')) {
                     // Multi-line comment
                     while (!(peek() == '*' && peekNext() == '/') && !isAtEnd()) {
                         if (peek() == '\n') {
                             line++;
                             column = 1;
                         }
                         advance();
                     }
                     
                     if (isAtEnd()) {
                         errorReporter.reportError("Unterminated comment", 
                                                  SourceLocation(filename, line, column));
                     } else {
                         // Consume the closing */
                         advance();
                         advance();
                     }
                     // Don't add comment tokens for now
                 } else if (match('=')) {
                     addToken(TokenType::SLASH_EQUAL);
                 } else {
                     addToken(TokenType::SLASH);
                 }
                 break;
                 
             case '%': 
                 if (match('=')) addToken(TokenType::PERCENT_EQUAL);
                 else addToken(TokenType::PERCENT);
                 break;
                 
             case '&': 
                 if (match('&')) addToken(TokenType::AMPERSAND_AMPERSAND);
                 else if (match('=')) addToken(TokenType::AMPERSAND_EQUAL);
                 else addToken(TokenType::AMPERSAND);
                 break;
                 
             case '|': 
                 if (match('|')) addToken(TokenType::PIPE_PIPE);
                 else if (match('=')) addToken(TokenType::PIPE_EQUAL);
                 else addToken(TokenType::PIPE);
                 break;
                 
             case '^': 
                 if (match('=')) addToken(TokenType::CARET_EQUAL);
                 else addToken(TokenType::CARET);
                 break;
                 
             case '!': 
                 if (match('=')) addToken(TokenType::EXCLAMATION_EQUAL);
                 else addToken(TokenType::EXCLAMATION);
                 break;
                 
             case '=': 
                 if (match('=')) addToken(TokenType::EQUAL_EQUAL);
                 else addToken(TokenType::EQUAL);
                 break;
                 
             case '<': 
                 if (match('<')) {
                     if (match('=')) addToken(TokenType::LESS_LESS_EQUAL);
                     else addToken(TokenType::LESS_LESS);
                 } else if (match('=')) {
                     addToken(TokenType::LESS_EQUAL);
                 } else {
                     addToken(TokenType::LESS);
                 }
                 break;
                 
             case '>': 
                 if (match('>')) {
                     if (match('=')) addToken(TokenType::GREATER_GREATER_EQUAL);
                     else addToken(TokenType::GREATER_GREATER);
                 } else if (match('=')) {
                     addToken(TokenType::GREATER_EQUAL);
                 } else {
                     addToken(TokenType::GREATER);
                 }
                 break;
                 
             case ':': 
                 if (match(':')) addToken(TokenType::COLON_COLON);
                 else addToken(TokenType::COLON);
                 break;
                 
             // Whitespace handling
             case ' ':
             case '\r':
             case '\t':
                 // Ignore whitespace
                 break;
                 
             case '\n':
                 line++;
                 column = 1;
                 break;
                 
             // Literals
             case '"': stringLiteral(); break;
             case '\'': charLiteral(); break;
                 
             // Preprocessor directive
             case '#': 
                 // Handle preprocessor directives
                 while (peek() != '\n' && !isAtEnd()) advance();
                 // Currently just skipping preprocessor directives
                 break;
                 
             default:
                 if (isDigit(c)) {
                     number();
                 } else if (isAlpha(c) || c == '_') {
                     identifier();
                 } else {
                     errorReporter.reportError(
                         std::format("Unexpected character: {}", c),
                         SourceLocation(filename, line, column - 1)
                     );
                 }
                 break;
         }
     }
     
     /**
      * @brief Process an identifier or keyword
      */
     void identifier() {
         while (isAlphaNumeric(peek())) advance();
         
         // See if the identifier is a reserved word
         std::string text = source.substr(start, current - start);
         
         auto it = keywords.find(text);
         TokenType type = it != keywords.end() ? it->second : TokenType::IDENTIFIER;
         
         // Handle boolean literals
         if (text == "true" || text == "false") {
             type = TokenType::BOOL_LITERAL;
         }
         
         addToken(type);
     }
     
     /**
      * @brief Process a numeric literal
      */
     void number() {
         bool isFloat = false;
         
         // Consume integers
         while (isDigit(peek())) advance();
         
         // Look for decimal point
         if (peek() == '.' && isDigit(peekNext())) {
             isFloat = true;
             advance(); // Consume the '.'
             
             // Consume fractional part
             while (isDigit(peek())) advance();
         }
         
         // Look for exponent
         if (peek() == 'e' || peek() == 'E') {
             isFloat = true;
             advance(); // Consume the 'e' or 'E'
             
             // Optional sign
             if (peek() == '+' || peek() == '-') advance();
             
             // Exponent digits
             if (!isDigit(peek())) {
                 errorReporter.reportError(
                     "Expected digits after exponent",
                     SourceLocation(filename, line, column)
                 );
             }
             
             while (isDigit(peek())) advance();
         }
         
         // Look for suffixes
         if (peek() == 'f' || peek() == 'F' || peek() == 'l' || peek() == 'L') {
             isFloat = true;
             advance();
         } else if ((peek() == 'u' || peek() == 'U') && !isFloat) {
             advance();
             // Optional size suffix
             if (peek() == 'l' || peek() == 'L') {
                 advance();
                 if (peek() == 'l' || peek() == 'L') advance();
             }
         } else if ((peek() == 'l' || peek() == 'L') && !isFloat) {
             advance();
             if (peek() == 'l' || peek() == 'L') advance();
             // Optional unsigned suffix
             if (peek() == 'u' || peek() == 'U') advance();
         }
         
         addToken(isFloat ? TokenType::FLOAT_LITERAL : TokenType::INTEGER_LITERAL);
     }
     
     /**
      * @brief Process a string literal
      */
     void stringLiteral() {
         while (peek() != '"' && !isAtEnd()) {
             if (peek() == '\n') {
                 errorReporter.reportError(
                     "Unterminated string literal",
                     SourceLocation(filename, line, column)
                 );
                 break;
             }
             
             // Handle escape sequences
             if (peek() == '\\') {
                 advance();
                 if (peek() == 'n' || peek() == 't' || peek() == 'r' || peek() == '"' || peek() == '\\') {
                     advance();
                 } else {
                     // Other escape sequences not handled for simplicity
                     advance();
                 }
             } else {
                 advance();
             }
         }
         
         if (isAtEnd()) {
             errorReporter.reportError(
                 "Unterminated string literal",
                 SourceLocation(filename, line, column)
             );
             return;
         }
         
         // Consume the closing "
         advance();
         
         // Extract the string content (without the quotes)
         addToken(TokenType::STRING_LITERAL);
     }
     
     /**
      * @brief Process a character literal
      */
     void charLiteral() {
         if (isAtEnd() || peek() == '\'') {
             errorReporter.reportError(
                 "Empty character literal",
                 SourceLocation(filename, line, column)
             );
             if (!isAtEnd()) advance(); // Consume the closing '
             addToken(TokenType::CHAR_LITERAL);
             return;
         }
         
         if (peek() == '\\') {
             advance(); // Consume the backslash
             if (isAtEnd()) {
                 errorReporter.reportError(
                     "Unterminated character literal",
                     SourceLocation(filename, line, column)
                 );
                 return;
             }
             advance(); // Consume the escaped character
         } else {
             advance(); // Consume the character
         }
         
         if (isAtEnd() || peek() != '\'') {
             errorReporter.reportError(
                 "Unterminated character literal",
                 SourceLocation(filename, line, column)
             );
             return;
         }
         
         advance(); // Consume the closing '
         addToken(TokenType::CHAR_LITERAL);
     }
     
     /**
      * @brief Check if we're at the end of the source code
      * @return True if at the end, false otherwise
      */
     bool isAtEnd() const {
         return current >= source.length();
     }
     
     /**
      * @brief Consume the current character and return it
      * @return The current character
      */
     char advance() {
         char c = source[current++];
         column++;
         return c;
     }
     
     /**
      * @brief Add a token to the token list
      * @param type The type of token to add
      */
     void addToken(TokenType type) {
         std::string lexeme = source.substr(start, current - start);
         tokens.emplace_back(type, lexeme, SourceLocation(filename, line, column - lexeme.length()));
     }
     
     /**
      * @brief Check if the current character matches the expected character
      * @param expected The character to check against
      * @return True if the characters match, false otherwise
      */
     bool match(char expected) {
         if (isAtEnd()) return false;
         if (source[current] != expected) return false;
         
         current++;
         column++;
         return true;
     }
     
     /**
      * @brief Look at the current character without consuming it
      * @return The current character, or '\0' if at the end
      */
     char peek() const {
         if (isAtEnd()) return '\0';
         return source[current];
     }
     
     /**
      * @brief Look at the next character without consuming it
      * @return The next character, or '\0' if at the end
      */
     char peekNext() const {
         if (current + 1 >= source.length()) return '\0';
         return source[current + 1];
     }
     
     /**
      * @brief Check if a character is a digit
      * @param c The character to check
      * @return True if the character is a digit, false otherwise
      */
     static bool isDigit(char c) {
         return c >= '0' && c <= '9';
     }
     
     /**
      * @brief Check if a character is alphabetic
      * @param c The character to check
      * @return True if the character is alphabetic, false otherwise
      */
     static bool isAlpha(char c) {
         return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
     }
     
     /**
      * @brief Check if a character is alphanumeric
      * @param c The character to check
      * @return True if the character is alphanumeric, false otherwise
      */
     static bool isAlphaNumeric(char c) {
         return isAlpha(c) || isDigit(c);
     }
 };
 
 // Static initialization of keywords map
 const std::unordered_map<std::string, TokenType> Lexer::keywords = {
     {"int", TokenType::INT},
     {"char", TokenType::CHAR},
     {"bool", TokenType::BOOL},
     {"float", TokenType::FLOAT},
     {"double", TokenType::DOUBLE},
     {"void", TokenType::VOID},
     {"auto", TokenType::AUTO},
     {"struct", TokenType::STRUCT},
     {"class", TokenType::CLASS},
     {"enum", TokenType::ENUM},
     {"union", TokenType::UNION},
     {"typedef", TokenType::TYPEDEF},
     {"const", TokenType::CONST},
     {"static", TokenType::STATIC},
     {"extern", TokenType::EXTERN},
     {"inline", TokenType::INLINE},
     {"virtual", TokenType::VIRTUAL},
     {"override", TokenType::OVERRIDE},
     {"final", TokenType::FINAL},
     {"public", TokenType::PUBLIC},
     {"private", TokenType::PRIVATE},
     {"protected", TokenType::PROTECTED},
     {"if", TokenType::IF},
     {"else", TokenType::ELSE},
     {"while", TokenType::WHILE},
     {"for", TokenType::FOR},
     {"do", TokenType::DO},
     {"switch", TokenType::SWITCH},
     {"case", TokenType::CASE},
     {"default", TokenType::DEFAULT},
     {"break", TokenType::BREAK},
     {"continue", TokenType::CONTINUE},
     {"return", TokenType::RETURN},
     {"new", TokenType::NEW},
     {"delete", TokenType::DELETE},
     {"try", TokenType::TRY},
     {"catch", TokenType::CATCH},
     {"throw", TokenType::THROW},
     {"namespace", TokenType::NAMESPACE},
     {"using", TokenType::USING},
     {"template", TokenType::TEMPLATE},
     {"typename", TokenType::TYPENAME}
 };
 
 /**
  * @brief Enumeration of AST node types
  */
 enum class ASTNodeType {
     // Program structure
     PROGRAM, NAMESPACE_DECL, USING_DIRECTIVE,
     
     // Declarations
     VARIABLE_DECL, FUNCTION_DECL, CLASS_DECL, STRUCT_DECL, 
     ENUM_DECL, TYPEDEF_DECL, TEMPLATE_DECL,
     
     // Statements
     COMPOUND_STMT, EXPRESSION_STMT, IF_STMT, WHILE_STMT, FOR_STMT,
     DO_WHILE_STMT, SWITCH_STMT, CASE_STMT, DEFAULT_STMT,
     BREAK_STMT, CONTINUE_STMT, RETURN_STMT, TRY_STMT, CATCH_STMT, THROW_STMT,
     
     // Expressions
     BINARY_EXPR, UNARY_EXPR, LITERAL_EXPR, VARIABLE_EXPR, 
     ASSIGNMENT_EXPR, CALL_EXPR, MEMBER_ACCESS_EXPR, ARRAY_ACCESS_EXPR,
     NEW_EXPR, DELETE_EXPR, CAST_EXPR, CONDITIONAL_EXPR,
     
     // Types
     TYPE_SPECIFIER, ARRAY_TYPE, POINTER_TYPE, REFERENCE_TYPE,
     FUNCTION_TYPE, QUALIFIED_TYPE, TEMPLATE_TYPE,
     
     // Other
     PARAMETER, INITIALIZER, MEMBER, TEMPLATE_PARAMETER,
     TEMPLATE_ARGUMENT, ACCESS_SPECIFIER
 };
 
 /**
  * @brief Base class for all nodes in the Abstract Syntax Tree (AST)
  */
 class ASTNode {
 public:
     ASTNodeType type;
     SourceLocation location;
     
     ASTNode(ASTNodeType type, const SourceLocation& location)
         : type(type), location(location) {}
     
     virtual ~ASTNode() = default;
     
     virtual std::string toString() const {
         return "ASTNode";
     }
 };
 
 /**
  * @brief Type system for representing C++ types
  */
 class Type {
 public:
     enum class TypeKind {
         VOID, BOOL, CHAR, INT, FLOAT, DOUBLE,
         POINTER, REFERENCE, ARRAY, FUNCTION,
         CLASS, STRUCT, ENUM, UNION, TYPEDEF, TEMPLATE, AUTO
     };
     
     TypeKind kind;
     bool isConst = false;
     bool isVolatile = false;
     
     Type(TypeKind kind) : kind(kind) {}
     
     virtual ~Type() = default;
     
     virtual std::string toString() const {
         std::string result;
         
         if (isConst) result += "const ";
         if (isVolatile) result += "volatile ";
         
         switch (kind) {
             case TypeKind::VOID: result += "void"; break;
             case TypeKind::BOOL: result += "bool"; break;
             case TypeKind::CHAR: result += "char"; break;
             case TypeKind::INT: result += "int"; break;
             case TypeKind::FLOAT: result += "float"; break;
             case TypeKind::DOUBLE: result += "double"; break;
             case TypeKind::AUTO: result += "auto"; break;
             default: result += "unknown"; break;
         }
         
         return result;
     }
     
     virtual bool isCompatibleWith(const Type& other) const {
         return kind == other.kind;
     }
     
     virtual size_t getSize() const {
         switch (kind) {
             case TypeKind::VOID: return 0;
             case TypeKind::BOOL: return 1;
             case TypeKind::CHAR: return 1;
             case TypeKind::INT: return 4;
             case TypeKind::FLOAT: return 4;
             case TypeKind::DOUBLE: return 8;
             default: return 0;
         }
     }
     
     virtual bool isNumeric() const {
         return kind == TypeKind::INT || kind == TypeKind::FLOAT || 
                kind == TypeKind::DOUBLE || kind == TypeKind::CHAR;
     }
     
     virtual bool isInteger() const {
         return kind == TypeKind::INT || kind == TypeKind::CHAR;
     }
     
     virtual bool isFloatingPoint() const {
         return kind == TypeKind::FLOAT || kind == TypeKind::DOUBLE;
     }
 };
 
 /**
  * @brief Pointer type in the type system
  */
 class PointerType : public Type {
 public:
     std::shared_ptr<Type> baseType;
     
     PointerType(std::shared_ptr<Type> baseType)
         : Type(TypeKind::POINTER), baseType(baseType) {}
     
     std::string toString() const override {
         return baseType->toString() + "*";
     }
     
     bool isCompatibleWith(const Type& other) const override {
         if (other.kind != TypeKind::POINTER) return false;
         
         const PointerType& otherPtr = static_cast<const PointerType&>(other);
         return baseType->isCompatibleWith(*otherPtr.baseType);
     }
     
     size_t getSize() const override {
         return 8; // Assume 64-bit pointers
     }
 };
 
 /**
  * @brief Reference type in the type system
  */
 class ReferenceType : public Type {
 public:
     std::shared_ptr<Type> baseType;
     
     ReferenceType(std::shared_ptr<Type> baseType)
         : Type(TypeKind::REFERENCE), baseType(baseType) {}
     
     std::string toString() const override {
         return baseType->toString() + "&";
     }
     
     bool isCompatibleWith(const Type& other) const override {
         if (other.kind != TypeKind::REFERENCE) return false;
         
         const ReferenceType& otherRef = static_cast<const ReferenceType&>(other);
         return baseType->isCompatibleWith(*otherRef.baseType);
     }
     
     size_t getSize() const override {
         return 8; // Assume 64-bit references
     }
 };
 
 /**
  * @brief Array type in the type system
  */
 class ArrayType : public Type {
 public:
     std::shared_ptr<Type> elementType;
     int size; // -1 for unknown size
     
     ArrayType(std::shared_ptr<Type> elementType, int size = -1)
         : Type(TypeKind::ARRAY), elementType(elementType), size(size) {}
     
     std::string toString() const override {
         if (size >= 0) {
             return elementType->toString() + "[" + std::to_string(size) + "]";
         } else {
             return elementType->toString() + "[]";
         }
     }
     
     bool isCompatibleWith(const Type& other) const override {
         if (other.kind != TypeKind::ARRAY) return false;
         
         const ArrayType& otherArray = static_cast<const ArrayType&>(other);
         return elementType->isCompatibleWith(*otherArray.elementType);
     }
     
     size_t getSize() const override {
         if (size < 0) return 0;
         return size * elementType->getSize();
     }
 };
 
 /**
  * @brief Function type in the type system
  */
 class FunctionType : public Type {
 public:
     std::shared_ptr<Type> returnType;
     std::vector<std::shared_ptr<Type>> parameterTypes;
     
     FunctionType(std::shared_ptr<Type> returnType,
                  std::vector<std::shared_ptr<Type>> parameterTypes)
         : Type(TypeKind::FUNCTION), returnType(returnType), parameterTypes(parameterTypes) {}
     
     std::string toString() const override {
         std::string result = returnType->toString() + " (";
         
         for (size_t i = 0; i < parameterTypes.size(); i++) {
             if (i > 0) result += ", ";
             result += parameterTypes[i]->toString();
         }
         
         result += ")";
         return result;
     }
     
     bool isCompatibleWith(const Type& other) const override {
         if (other.kind != TypeKind::FUNCTION) return false;
         
         const FunctionType& otherFunc = static_cast<const FunctionType&>(other);
         
         if (!returnType->isCompatibleWith(*otherFunc.returnType)) return false;
         if (parameterTypes.size() != otherFunc.parameterTypes.size()) return false;
         
         for (size_t i = 0; i < parameterTypes.size(); i++) {
             if (!parameterTypes[i]->isCompatibleWith(*otherFunc.parameterTypes[i])) {
                 return false;
             }
         }
         
         return true;
     }
     
     size_t getSize() const override {
         return 8; // Function pointers are typically 8 bytes on 64-bit systems
     }
 };
 
 /**
  * @brief Class/Struct type in the type system
  */
 class CompositeType : public Type {
 public:
     std::string name;
     std::unordered_map<std::string, std::shared_ptr<Type>> members;
     
     CompositeType(TypeKind kind, const std::string& name)
         : Type(kind), name(name) {
         assert(kind == TypeKind::CLASS || kind == TypeKind::STRUCT || 
                kind == TypeKind::UNION);
     }
     
     std::string toString() const override {
         std::string kindStr;
         switch (kind) {
             case TypeKind::CLASS: kindStr = "class"; break;
             case TypeKind::STRUCT: kindStr = "struct"; break;
             case TypeKind::UNION: kindStr = "union"; break;
             default: kindStr = "unknown"; break;
         }
         
         return kindStr + " " + name;
     }
     
     bool isCompatibleWith(const Type& other) const override {
         if (other.kind != kind) return false;
         
         const CompositeType& otherType = static_cast<const CompositeType&>(other);
         return name == otherType.name;
     }
     
     size_t getSize() const override {
         // This is a simplified calculation that doesn't account for padding
         size_t totalSize = 0;
         
         if (kind == TypeKind::UNION) {
             // For unions, the size is the size of the largest member
             for (const auto& [memberName, memberType] : members) {
                 totalSize = std::max(totalSize, memberType->getSize());
             }
         } else {
             // For classes and structs, the size is the sum of the members
             for (const auto& [memberName, memberType] : members) {
                 totalSize += memberType->getSize();
             }
         }
         
         return totalSize;
     }
     
     void addMember(const std::string& name, std::shared_ptr<Type> type) {
         members[name] = type;
     }
     
     std::shared_ptr<Type> getMember(const std::string& name) const {
         auto it = members.find(name);
         if (it != members.end()) {
             return it->second;
         }
         return nullptr;
     }
 };
 
 /**
  * @brief A symbol table entry for variable declarations
  */
 struct VariableSymbol {
     std::string name;
     std::shared_ptr<Type> type;
     bool isConst;
     SourceLocation location;
     
     VariableSymbol(const std::string& name, std::shared_ptr<Type> type, 
                    bool isConst, const SourceLocation& location)
         : name(name), type(type), isConst(isConst), location(location) {}
 };
 
 /**
  * @brief A symbol table entry for function declarations
  */
 struct FunctionSymbol {
     std::string name;
     std::shared_ptr<FunctionType> type;
     bool isInline;
     bool isVirtual;
     SourceLocation location;
     
     FunctionSymbol(const std::string& name, std::shared_ptr<FunctionType> type,
                   bool isInline, bool isVirtual, const SourceLocation& location)
         : name(name), type(type), isInline(isInline), isVirtual(isVirtual), location(location) {}
 };
 
 /**
  * @brief A symbol table entry for type declarations
  */
 struct TypeSymbol {
     std::string name;
     std::shared_ptr<Type> type;
     SourceLocation location;
     
     TypeSymbol(const std::string& name, std::shared_ptr<Type> type, const SourceLocation& location)
         : name(name), type(type), location(location) {}
 };
 
 /**
  * @brief A symbol table for managing variable, function, and type declarations
  * 
  * Thread-safe to allow parallel processing of different scopes.
  */
 class SymbolTable {
 private:
     std::unordered_map<std::string, VariableSymbol> variables;
     std::unordered_map<std::string, FunctionSymbol> functions;
     std::unordered_map<std::string, TypeSymbol> types;
     std::shared_ptr<SymbolTable> parent;
     mutable std::shared_mutex mutex; // For thread safety
 
 public:
     SymbolTable(std::shared_ptr<SymbolTable> parent = nullptr) : parent(parent) {}
     
     /**
      * @brief Define a variable in the current scope
      * @param name The variable name
      * @param type The variable type
      * @param isConst Whether the variable is const
      * @param location Source location for error reporting
      * @return True if successfully defined, false if already defined
      */
     bool defineVariable(const std::string& name, std::shared_ptr<Type> type, 
                         bool isConst, const SourceLocation& location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (variables.find(name) != variables.end()) {
             return false; // Already defined in this scope
         }
         
         variables.emplace(name, VariableSymbol(name, type, isConst, location));
         return true;
     }
     
     /**
      * @brief Define a function in the current scope
      * @param name The function name
      * @param type The function type
      * @param isInline Whether the function is inline
      * @param isVirtual Whether the function is virtual
      * @param location Source location for error reporting
      * @return True if successfully defined, false if already defined
      */
     bool defineFunction(const std::string& name, std::shared_ptr<FunctionType> type,
                        bool isInline, bool isVirtual, const SourceLocation& location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (functions.find(name) != functions.end()) {
             return false; // Already defined in this scope
         }
         
         functions.emplace(name, FunctionSymbol(name, type, isInline, isVirtual, location));
         return true;
     }
     
     /**
      * @brief Define a type in the current scope
      * @param name The type name
      * @param type The type definition
      * @param location Source location for error reporting
      * @return True if successfully defined, false if already defined
      */
     bool defineType(const std::string& name, std::shared_ptr<Type> type, 
                     const SourceLocation& location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (types.find(name) != types.end()) {
             return false; // Already defined in this scope
         }
         
         types.emplace(name, TypeSymbol(name, type, location));
         return true;
     }
     
     /**
      * @brief Resolve a variable from the current or parent scopes
      * @param name The variable name to resolve
      * @return The variable symbol if found, nullptr otherwise
      */
     std::optional<VariableSymbol> resolveVariable(const std::string& name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto it = variables.find(name);
         if (it != variables.end()) {
             return it->second;
         }
         
         if (parent) {
             return parent->resolveVariable(name);
         }
         
         return std::nullopt;
     }
     
     /**
      * @brief Resolve a function from the current or parent scopes
      * @param name The function name to resolve
      * @return The function symbol if found, nullptr otherwise
      */
     std::optional<FunctionSymbol> resolveFunction(const std::string& name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto it = functions.find(name);
         if (it != functions.end()) {
             return it->second;
         }
         
         if (parent) {
             return parent->resolveFunction(name);
         }
         
         return std::nullopt;
     }
     
     /**
      * @brief Resolve a type from the current or parent scopes
      * @param name The type name to resolve
      * @return The type symbol if found, nullptr otherwise
      */
     std::optional<TypeSymbol> resolveType(const std::string& name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto it = types.find(name);
         if (it != types.end()) {
             return it->second;
         }
         
         if (parent) {
             return parent->resolveType(name);
         }
         
         return std::nullopt;
     }
 };
 
 /**
  * @brief Node representing a program in the AST
  */
 class ProgramNode : public ASTNode {
 public:
     std::vector<std::unique_ptr<ASTNode>> declarations;
     
     ProgramNode(const SourceLocation& location)
         : ASTNode(ASTNodeType::PROGRAM, location) {}
     
     std::string toString() const override {
         return "Program";
     }
 };
 
 /**
  * @brief Node representing a variable declaration in the AST
  */
 class VariableDeclNode : public ASTNode {
 public:
     std::string name;
     std::shared_ptr<Type> type;
     std::unique_ptr<ASTNode> initializer;
     bool isConst;
     
     VariableDeclNode(const std::string& name, std::shared_ptr<Type> type,
                     std::unique_ptr<ASTNode> initializer, bool isConst,
                     const SourceLocation& location)
         : ASTNode(ASTNodeType::VARIABLE_DECL, location), name(name), type(type),
           initializer(std::move(initializer)), isConst(isConst) {}
     
     std::string toString() const override {
         std::string result = "VariableDecl: " + name + " : " + type->toString();
         if (isConst) result += " (const)";
         return result;
     }
 };
 
 /**
  * @brief Node representing a function declaration in the AST
  */
 class FunctionDeclNode : public ASTNode {
 public:
     std::string name;
     std::shared_ptr<FunctionType> type;
     std::vector<std::unique_ptr<ASTNode>> parameters;
     std::unique_ptr<ASTNode> body;
     bool isInline;
     bool isVirtual;
     
     FunctionDeclNode(const std::string& name, std::shared_ptr<FunctionType> type,
                     std::vector<std::unique_ptr<ASTNode>> parameters,
                     std::unique_ptr<ASTNode> body, bool isInline, bool isVirtual,
                     const SourceLocation& location)
         : ASTNode(ASTNodeType::FUNCTION_DECL, location), name(name), type(type),
           parameters(std::move(parameters)), body(std::move(body)),
           isInline(isInline), isVirtual(isVirtual) {}
     
     std::string toString() const override {
         std::string result = "FunctionDecl: " + name + " : " + type->toString();
         if (isInline) result += " (inline)";
         if (isVirtual) result += " (virtual)";
         return result;
     }
 };
 
 /**
  * @brief Node representing a compound statement (block) in the AST
  */
 class CompoundStmtNode : public ASTNode {
 public:
     std::vector<std::unique_ptr<ASTNode>> statements;
     
     CompoundStmtNode(const SourceLocation& location)
         : ASTNode(ASTNodeType::COMPOUND_STMT, location) {}
     
     std::string toString() const override {
         return "CompoundStmt";
     }
 };
 
 /**
  * @brief Node representing an expression statement in the AST
  */
 class ExpressionStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> expression;
     
     ExpressionStmtNode(std::unique_ptr<ASTNode> expression, const SourceLocation& location)
         : ASTNode(ASTNodeType::EXPRESSION_STMT, location), expression(std::move(expression)) {}
     
     std::string toString() const override {
         return "ExpressionStmt";
     }
 };
 
 /**
  * @brief Node representing an if statement in the AST
  */
 class IfStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> condition;
     std::unique_ptr<ASTNode> thenBranch;
     std::unique_ptr<ASTNode> elseBranch;
     
     IfStmtNode(std::unique_ptr<ASTNode> condition, std::unique_ptr<ASTNode> thenBranch,
               std::unique_ptr<ASTNode> elseBranch, const SourceLocation& location)
         : ASTNode(ASTNodeType::IF_STMT, location), condition(std::move(condition)),
           thenBranch(std::move(thenBranch)), elseBranch(std::move(elseBranch)) {}
     
     std::string toString() const override {
         return "IfStmt";
     }
 };
 
 /**
  * @brief Node representing a while statement in the AST
  */
 class WhileStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> condition;
     std::unique_ptr<ASTNode> body;
     
     WhileStmtNode(std::unique_ptr<ASTNode> condition, std::unique_ptr<ASTNode> body,
                  const SourceLocation& location)
         : ASTNode(ASTNodeType::WHILE_STMT, location), condition(std::move(condition)),
           body(std::move(body)) {}
     
     std::string toString() const override {
         return "WhileStmt";
     }
 };
 
 /**
  * @brief Node representing a for statement in the AST
  */
 class ForStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> initializer;
     std::unique_ptr<ASTNode> condition;
     std::unique_ptr<ASTNode> increment;
     std::unique_ptr<ASTNode> body;
     
     ForStmtNode(std::unique_ptr<ASTNode> initializer, std::unique_ptr<ASTNode> condition,
                std::unique_ptr<ASTNode> increment, std::unique_ptr<ASTNode> body,
                const SourceLocation& location)
         : ASTNode(ASTNodeType::FOR_STMT, location), initializer(std::move(initializer)),
           condition(std::move(condition)), increment(std::move(increment)),
           body(std::move(body)) {}
     
     std::string toString() const override {
         return "ForStmt";
     }
 };
 
 /**
  * @brief Node representing a return statement in the AST
  */
 class ReturnStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> value;
     
     ReturnStmtNode(std::unique_ptr<ASTNode> value, const SourceLocation& location)
         : ASTNode(ASTNodeType::RETURN_STMT, location), value(std::move(value)) {}
     
     std::string toString() const override {
         return "ReturnStmt";
     }
 };
 
 /**
  * @brief Node representing a binary expression in the AST
  */
 class BinaryExprNode : public ASTNode {
 public:
     enum class Operator {
         ADD, SUBTRACT, MULTIPLY, DIVIDE, MODULO,
         EQUAL, NOT_EQUAL, LESS, LESS_EQUAL, GREATER, GREATER_EQUAL,
         AND, OR, BITWISE_AND, BITWISE_OR, BITWISE_XOR,
         LEFT_SHIFT, RIGHT_SHIFT
     };
     
     Operator op;
     std::unique_ptr<ASTNode> left;
     std::unique_ptr<ASTNode> right;
     
     BinaryExprNode(Operator op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right,
                   const SourceLocation& location)
         : ASTNode(ASTNodeType::BINARY_EXPR, location), op(op), left(std::move(left)),
           right(std::move(right)) {}
     
     std::string toString() const override {
         std::string opStr;
         switch (op) {
             case Operator::ADD: opStr = "+"; break;
             case Operator::SUBTRACT: opStr = "-"; break;
             case Operator::MULTIPLY: opStr = "*"; break;
             case Operator::DIVIDE: opStr = "/"; break;
             case Operator::MODULO: opStr = "%"; break;
             case Operator::EQUAL: opStr = "=="; break;
             case Operator::NOT_EQUAL: opStr = "!="; break;
             case Operator::LESS: opStr = "<"; break;
             case Operator::LESS_EQUAL: opStr = "<="; break;
             case Operator::GREATER: opStr = ">"; break;
             case Operator::GREATER_EQUAL: opStr = ">="; break;
             case Operator::AND: opStr = "&&"; break;
             case Operator::OR: opStr = "||"; break;
             case Operator::BITWISE_AND: opStr = "&"; break;
             case Operator::BITWISE_OR: opStr = "|"; break;
             case Operator::BITWISE_XOR: opStr = "^"; break;
             case Operator::LEFT_SHIFT: opStr = "<<"; break;
             case Operator::RIGHT_SHIFT: opStr = ">>"; break;
         }
         
         return "BinaryExpr: " + opStr;
     }
 };
 
 /**
  * @brief Node representing a unary expression in the AST
  */
 class UnaryExprNode : public ASTNode {
 public:
     enum class Operator {
         NEGATE, NOT, BITWISE_NOT, ADDRESS_OF, DEREFERENCE,
         PRE_INCREMENT, PRE_DECREMENT, POST_INCREMENT, POST_DECREMENT
     };
     
     Operator op;
     std::unique_ptr<ASTNode> operand;
     
     UnaryExprNode(Operator op, std::unique_ptr<ASTNode> operand, const SourceLocation& location)
         : ASTNode(ASTNodeType::UNARY_EXPR, location), op(op), operand(std::move(operand)) {}
     
     std::string toString() const override {
         std::string opStr;
         switch (op) {
             case Operator::NEGATE: opStr = "-"; break;
             case Operator::NOT: opStr = "!"; break;
             case Operator::BITWISE_NOT: opStr = "~"; break;
             case Operator::ADDRESS_OF: opStr = "&"; break;
             case Operator::DEREFERENCE: opStr = "*"; break;
             case Operator::PRE_INCREMENT: opStr = "++"; break;
             case Operator::PRE_DECREMENT: opStr = "--"; break;
             case Operator::POST_INCREMENT: opStr = "++ (post)"; break;
             case Operator::POST_DECREMENT: opStr = "-- (post)"; break;
         }
         
         return "UnaryExpr: " + opStr;
     }
 };
 
 /**
  * @brief Node representing a literal expression in the AST
  */
 class LiteralExprNode : public ASTNode {
 public:
     enum class LiteralType {
         INTEGER, FLOAT, CHARACTER, STRING, BOOLEAN, NULL_LITERAL
     };
     
     LiteralType literalType;
     std::string value;
     
     LiteralExprNode(LiteralType literalType, const std::string& value, 
                    const SourceLocation& location)
         : ASTNode(ASTNodeType::LITERAL_EXPR, location), literalType(literalType), value(value) {}
     
     std::string toString() const override {
         std::string typeStr;
         switch (literalType) {
             case LiteralType::INTEGER: typeStr = "Integer"; break;
             case LiteralType::FLOAT: typeStr = "Float"; break;
             case LiteralType::CHARACTER: typeStr = "Character"; break;
             case LiteralType::STRING: typeStr = "String"; break;
             case LiteralType::BOOLEAN: typeStr = "Boolean"; break;
             case LiteralType::NULL_LITERAL: typeStr = "Null"; break;
         }
         
         return "LiteralExpr: " + typeStr + " " + value;
     }
 };
 
 /**
  * @brief Node representing a variable expression in the AST
  */
 class VariableExprNode : public ASTNode {
 public:
     std::string name;
     
     VariableExprNode(const std::string& name, const SourceLocation& location)
         : ASTNode(ASTNodeType::VARIABLE_EXPR, location), name(name) {}
     
     std::string toString() const override {
         return "VariableExpr: " + name;
     }
 };
 
 /**
  * @brief Node representing an assignment expression in the AST
  */
 class AssignmentExprNode : public ASTNode {
 public:
     enum class Operator {
         ASSIGN, ADD_ASSIGN, SUBTRACT_ASSIGN, MULTIPLY_ASSIGN,
         DIVIDE_ASSIGN, MODULO_ASSIGN, AND_ASSIGN, OR_ASSIGN,
         XOR_ASSIGN, LEFT_SHIFT_ASSIGN, RIGHT_SHIFT_ASSIGN
     };
     
     Operator op;
     std::unique_ptr<ASTNode> left;
     std::unique_ptr<ASTNode> right;
     
     AssignmentExprNode(Operator op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right,
                       const SourceLocation& location)
         : ASTNode(ASTNodeType::ASSIGNMENT_EXPR, location), op(op), left(std::move(left)),
           right(std::move(right)) {}
     
     std::string toString() const override {
         std::string opStr;
         switch (op) {
             case Operator::ASSIGN: opStr = "="; break;
             case Operator::ADD_ASSIGN: opStr = "+="; break;
             case Operator::SUBTRACT_ASSIGN: opStr = "-="; break;
             case Operator::MULTIPLY_ASSIGN: opStr = "*="; break;
             case Operator::DIVIDE_ASSIGN: opStr = "/="; break;
             case Operator::MODULO_ASSIGN: opStr = "%="; break;
             case Operator::AND_ASSIGN: opStr = "&="; break;
             case Operator::OR_ASSIGN: opStr = "|="; break;
             case Operator::XOR_ASSIGN: opStr = "^="; break;
             case Operator::LEFT_SHIFT_ASSIGN: opStr = "<<="; break;
             case Operator::RIGHT_SHIFT_ASSIGN: opStr = ">>="; break;
         }
         
         return "AssignmentExpr: " + opStr;
     }
 };
 
 /**
  * @brief Node representing a function call expression in the AST
  */
 class CallExprNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> callee;
     std::vector<std::unique_ptr<ASTNode>> arguments;
     
     CallExprNode(std::unique_ptr<ASTNode> callee, std::vector<std::unique_ptr<ASTNode>> arguments,
                 const SourceLocation& location)
         : ASTNode(ASTNodeType::CALL_EXPR, location), callee(std::move(callee)),
           arguments(std::move(arguments)) {}
     
     std::string toString() const override {
         return "CallExpr";
     }
 };
 
 /**
  * @brief Parser class for generating the AST from tokens
  * 
  * Time Complexity: O(n) where n is the number of tokens
  * Space Complexity: O(d) where d is the maximum depth of the AST
  */
 class Parser {
 private:
     const std::vector<Token>& tokens;
     size_t current = 0;
     ErrorReporter& errorReporter;
     std::shared_ptr<SymbolTable> globalSymbols;
 
 public:
     Parser(const std::vector<Token>& tokens, ErrorReporter& errorReporter)
         : tokens(tokens), errorReporter(errorReporter) {
         globalSymbols = std::make_shared<SymbolTable>();
     }
     
     /**
      * @brief Parse the tokens into an AST
      * @return The root node of the AST
      */
     std::unique_ptr<ProgramNode> parse() {
         auto program = std::make_unique<ProgramNode>(SourceLocation());
         
         while (!isAtEnd()) {
             try {
                 program->declarations.push_back(parseDeclaration());
             } catch (const std::exception& e) {
                 errorReporter.reportError(e.what(), peek().location);
                 synchronize();
             }
         }
         
         return program;
     }
 
 private:
     /**
      * @brief Parse a declaration
      * @return A node representing the declaration
      */
     std::unique_ptr<ASTNode> parseDeclaration() {
         if (match(TokenType::CLASS)) {
             return parseClassDeclaration();
         } else if (match(TokenType::STRUCT)) {
             return parseStructDeclaration();
         } else if (match(TokenType::ENUM)) {
             return parseEnumDeclaration();
         } else if (check(TokenType::INT) || check(TokenType::CHAR) || check(TokenType::BOOL) ||
                   check(TokenType::FLOAT) || check(TokenType::DOUBLE) || check(TokenType::VOID) ||
                   check(TokenType::AUTO) || check(TokenType::CONST)) {
             return parseVariableOrFunctionDeclaration();
         } else if (match(TokenType::NAMESPACE)) {
             return parseNamespaceDeclaration();
         } else if (match(TokenType::USING)) {
             return parseUsingDirective();
         } else if (match(TokenType::TEMPLATE)) {
             return parseTemplateDeclaration();
         } else if (match(TokenType::TYPEDEF)) {
             return parseTypedefDeclaration();
         }
         
         errorReporter.reportError("Expected declaration", peek().location);
         throw std::runtime_error("Expected declaration");
     }
     
     /**
      * @brief Parse a class declaration
      * @return A node representing the class declaration
      */
     std::unique_ptr<ASTNode> parseClassDeclaration() {
         // This is a simplified implementation
         // In a full compiler, this would handle inheritance, access specifiers, etc.
         
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected class name", peek().location);
             throw std::runtime_error("Expected class name");
         }
         
         std::string className = previous().lexeme;
         
         if (!match(TokenType::LEFT_BRACE)) {
             errorReporter.reportError("Expected '{' after class name", peek().location);
             throw std::runtime_error("Expected '{' after class name");
         }
         
         // Skip to the end of the class declaration for now
         int braceCount = 1;
         while (braceCount > 0 && !isAtEnd()) {
             if (peek().type == TokenType::LEFT_BRACE) {
                 braceCount++;
             } else if (peek().type == TokenType::RIGHT_BRACE) {
                 braceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::CLASS_DECL, previous().location);
     }
     
     /**
      * @brief Parse a struct declaration
      * @return A node representing the struct declaration
      */
     std::unique_ptr<ASTNode> parseStructDeclaration() {
         // Similar to class declaration for now
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected struct name", peek().location);
             throw std::runtime_error("Expected struct name");
         }
         
         std::string structName = previous().lexeme;
         
         if (!match(TokenType::LEFT_BRACE)) {
             errorReporter.reportError("Expected '{' after struct name", peek().location);
             throw std::runtime_error("Expected '{' after struct name");
         }
         
         // Skip to the end of the struct declaration for now
         int braceCount = 1;
         while (braceCount > 0 && !isAtEnd()) {
             if (peek().type == TokenType::LEFT_BRACE) {
                 braceCount++;
             } else if (peek().type == TokenType::RIGHT_BRACE) {
                 braceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::STRUCT_DECL, previous().location);
     }
     
     /**
      * @brief Parse an enum declaration
      * @return A node representing the enum declaration
      */
     std::unique_ptr<ASTNode> parseEnumDeclaration() {
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected enum name", peek().location);
             throw std::runtime_error("Expected enum name");
         }
         
         std::string enumName = previous().lexeme;
         
         if (!match(TokenType::LEFT_BRACE)) {
             errorReporter.reportError("Expected '{' after enum name", peek().location);
             throw std::runtime_error("Expected '{' after enum name");
         }
         
         // Skip to the end of the enum declaration for now
         int braceCount = 1;
         while (braceCount > 0 && !isAtEnd()) {
             if (peek().type == TokenType::LEFT_BRACE) {
                 braceCount++;
             } else if (peek().type == TokenType::RIGHT_BRACE) {
                 braceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::ENUM_DECL, previous().location);
     }
     
     /**
      * @brief Parse a variable or function declaration
      * @return A node representing the variable or function declaration
      */
     std::unique_ptr<ASTNode> parseVariableOrFunctionDeclaration() {
         // Check for const qualifier
         bool isConst = match(TokenType::CONST);
         
         // Parse type specifier
         auto typeSpecifier = parseTypeSpecifier();
         
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected identifier", peek().location);
             throw std::runtime_error("Expected identifier");
         }
         
         std::string name = previous().lexeme;
         
         // If next token is '(', it's a function declaration
         if (match(TokenType::LEFT_PAREN)) {
             // Function parameters
             std::vector<std::unique_ptr<ASTNode>> parameters;
             
             if (!check(TokenType::RIGHT_PAREN)) {
                 do {
                     // Parse parameter
                     bool paramConst = match(TokenType::CONST);
                     auto paramType = parseTypeSpecifier();
                     
                     if (!match(TokenType::IDENTIFIER)) {
                         errorReporter.reportError("Expected parameter name", peek().location);
                         throw std::runtime_error("Expected parameter name");
                     }
                     
                     std::string paramName = previous().lexeme;
                     
                     // Create parameter node (simplified for now)
                     parameters.push_back(std::make_unique<ASTNode>(ASTNodeType::PARAMETER, previous().location));
                     
                 } while (match(TokenType::COMMA));
             }
             
             consume(TokenType::RIGHT_PAREN, "Expected ')' after function parameters");
             
             // Function body
             std::unique_ptr<ASTNode> body = nullptr;
             if (match(TokenType::SEMICOLON)) {
                 // Function declaration without body
             } else {
                 body = parseCompoundStatement();
             }
             
             // Create function node (simplified for now)
             return std::make_unique<ASTNode>(ASTNodeType::FUNCTION_DECL, previous().location);
             
         } else {
             // It's a variable declaration
             std::unique_ptr<ASTNode> initializer = nullptr;
             
             if (match(TokenType::EQUAL)) {
                 initializer = parseExpression();
             }
             
             consume(TokenType::SEMICOLON, "Expected ';' after variable declaration");
             
             // Create variable node (simplified for now)
             return std::make_unique<ASTNode>(ASTNodeType::VARIABLE_DECL, previous().location);
         }
     }
     
     /**
      * @brief Parse a namespace declaration
      * @return A node representing the namespace declaration
      */
     std::unique_ptr<ASTNode> parseNamespaceDeclaration() {
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected namespace name", peek().location);
             throw std::runtime_error("Expected namespace name");
         }
         
         std::string namespaceName = previous().lexeme;
         
         if (!match(TokenType::LEFT_BRACE)) {
             errorReporter.reportError("Expected '{' after namespace name", peek().location);
             throw std::runtime_error("Expected '{' after namespace name");
         }
         
         // Skip to the end of the namespace declaration for now
         int braceCount = 1;
         while (braceCount > 0 && !isAtEnd()) {
             if (peek().type == TokenType::LEFT_BRACE) {
                 braceCount++;
             } else if (peek().type == TokenType::RIGHT_BRACE) {
                 braceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::NAMESPACE_DECL, previous().location);
     }
     
     /**
      * @brief Parse a using directive
      * @return A node representing the using directive
      */
     std::unique_ptr<ASTNode> parseUsingDirective() {
         if (!match(TokenType::NAMESPACE)) {
             errorReporter.reportError("Expected 'namespace' in using directive", peek().location);
             throw std::runtime_error("Expected 'namespace' in using directive");
         }
         
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected namespace name", peek().location);
             throw std::runtime_error("Expected namespace name");
         }
         
         std::string namespaceName = previous().lexeme;
         
         consume(TokenType::SEMICOLON, "Expected ';' after using directive");
         
         return std::make_unique<ASTNode>(ASTNodeType::USING_DIRECTIVE, previous().location);
     }
     
     /**
      * @brief Parse a template declaration
      * @return A node representing the template declaration
      */
     std::unique_ptr<ASTNode> parseTemplateDeclaration() {
         consume(TokenType::LESS, "Expected '<' after 'template'");
         
         // Parse template parameters
         do {
             if (match(TokenType::CLASS) || match(TokenType::TYPENAME)) {
                 if (!match(TokenType::IDENTIFIER)) {
                     errorReporter.reportError("Expected template parameter name", peek().location);
                     throw std::runtime_error("Expected template parameter name");
                 }
             } else {
                 // Parse non-type template parameter
                 auto paramType = parseTypeSpecifier();
                 
                 if (!match(TokenType::IDENTIFIER)) {
                     errorReporter.reportError("Expected template parameter name", peek().location);
                     throw std::runtime_error("Expected template parameter name");
                 }
                 
                 if (match(TokenType::EQUAL)) {
                     // Parse default value
                     parseExpression();
                 }
             }
         } while (match(TokenType::COMMA));
         
         consume(TokenType::GREATER, "Expected '>' after template parameters");
         
         // Parse the templated declaration
         auto declaration = parseDeclaration();
         
         return std::make_unique<ASTNode>(ASTNodeType::TEMPLATE_DECL, previous().location);
     }
     
     /**
      * @brief Parse a typedef declaration
      * @return A node representing the typedef declaration
      */
     std::unique_ptr<ASTNode> parseTypedefDeclaration() {
         auto type = parseTypeSpecifier();
         
         if (!match(TokenType::IDENTIFIER)) {
             errorReporter.reportError("Expected type alias name", peek().location);
             throw std::runtime_error("Expected type alias name");
         }
         
         std::string aliasName = previous().lexeme;
         
         consume(TokenType::SEMICOLON, "Expected ';' after typedef declaration");
         
         return std::make_unique<ASTNode>(ASTNodeType::TYPEDEF_DECL, previous().location);
     }
     
     /**
      * @brief Parse a type specifier
      * @return A shared pointer to the type
      */
     std::shared_ptr<Type> parseTypeSpecifier() {
         TokenType typeToken = peek().type;
         advance();
         
         Type::TypeKind kind;
         switch (typeToken) {
             case TokenType::VOID: kind = Type::TypeKind::VOID; break;
             case TokenType::BOOL: kind = Type::TypeKind::BOOL; break;
             case TokenType::CHAR: kind = Type::TypeKind::CHAR; break;
             case TokenType::INT: kind = Type::TypeKind::INT; break;
             case TokenType::FLOAT: kind = Type::TypeKind::FLOAT; break;
             case TokenType::DOUBLE: kind = Type::TypeKind::DOUBLE; break;
             case TokenType::AUTO: kind = Type::TypeKind::AUTO; break;
             default:
                 errorReporter.reportError("Expected type specifier", previous().location);
                 throw std::runtime_error("Expected type specifier");
         }
         
         auto baseType = std::make_shared<Type>(kind);
         
         // Handle pointers, references, and arrays
         while (match(TokenType::ASTERISK) || match(TokenType::AMPERSAND) || 
               check(TokenType::LEFT_BRACKET)) {
             if (previous().type == TokenType::ASTERISK) {
                 baseType = std::make_shared<PointerType>(baseType);
             } else if (previous().type == TokenType::AMPERSAND) {
                 baseType = std::make_shared<ReferenceType>(baseType);
             } else if (peek().type == TokenType::LEFT_BRACKET) {
                 advance();
                 int size = -1;
                 
                 if (match(TokenType::INTEGER_LITERAL)) {
                     size = std::stoi(previous().lexeme);
                 }
                 
                 consume(TokenType::RIGHT_BRACKET, "Expected ']' after array size");
                 baseType = std::make_shared<ArrayType>(baseType, size);
             }
         }
         
         return baseType;
     }
     
     /**
      * @brief Parse a compound statement (block)
      * @return A node representing the compound statement
      */
     std::unique_ptr<ASTNode> parseCompoundStatement() {
         consume(TokenType::LEFT_BRACE, "Expected '{' at the beginning of a block");
         
         auto compoundStmt = std::make_unique<CompoundStmtNode>(previous().location);
         
         while (!check(TokenType::RIGHT_BRACE) && !isAtEnd()) {
             compoundStmt->statements.push_back(parseStatement());
         }
         
         consume(TokenType::RIGHT_BRACE, "Expected '}' at the end of a block");
         
         return compoundStmt;
     }
     
     /**
      * @brief Parse a statement
      * @return A node representing the statement
      */
     std::unique_ptr<ASTNode> parseStatement() {
         if (match(TokenType::IF)) {
             return parseIfStatement();
         } else if (match(TokenType::WHILE)) {
             return parseWhileStatement();
         } else if (match(TokenType::FOR)) {
             return parseForStatement();
         } else if (match(TokenType::RETURN)) {
             return parseReturnStatement();
         } else if (match(TokenType::BREAK)) {
             auto node = std::make_unique<ASTNode>(ASTNodeType::BREAK_STMT, previous().location);
             consume(TokenType::SEMICOLON, "Expected ';' after break statement");
             return node;
         } else if (match(TokenType::CONTINUE)) {
             auto node = std::make_unique<ASTNode>(ASTNodeType::CONTINUE_STMT, previous().location);
             consume(TokenType::SEMICOLON, "Expected ';' after continue statement");
             return node;
         } else if (match(TokenType::LEFT_BRACE)) {
             // Parse a block
             current--; // Backtrack to the '{'
             return parseCompoundStatement();
         } else {
             // Expression statement or variable declaration
             if (check(TokenType::INT) || check(TokenType::CHAR) || check(TokenType::BOOL) ||
                 check(TokenType::FLOAT) || check(TokenType::DOUBLE) || check(TokenType::VOID) ||
                 check(TokenType::AUTO) || check(TokenType::CONST)) {
                 return parseVariableOrFunctionDeclaration();
             } else {
                 return parseExpressionStatement();
             }
         }
     }
     
     /**
      * @brief Parse an if statement
      * @return A node representing the if statement
      */
     std::unique_ptr<ASTNode> parseIfStatement() {
         consume(TokenType::LEFT_PAREN, "Expected '(' after 'if'");
         auto condition = parseExpression();
         consume(TokenType::RIGHT_PAREN, "Expected ')' after if condition");
         
         auto thenBranch = parseStatement();
         std::unique_ptr<ASTNode> elseBranch = nullptr;
         
         if (match(TokenType::ELSE)) {
             elseBranch = parseStatement();
         }
         
         return std::make_unique<IfStmtNode>(std::move(condition), std::move(thenBranch),
                                           std::move(elseBranch), previous().location);
     }
     
     /**
      * @brief Parse a while statement
      * @return A node representing the while statement
      */
     std::unique_ptr<ASTNode> parseWhileStatement() {
         consume(TokenType::LEFT_PAREN, "Expected '(' after 'while'");
         auto condition = parseExpression();
         consume(TokenType::RIGHT_PAREN, "Expected ')' after while condition");
         
         auto body = parseStatement();
         
         return std::make_unique<WhileStmtNode>(std::move(condition), std::move(body),
                                              previous().location);
     }
     
     /**
      * @brief Parse a for statement
      * @return A node representing the for statement
      */
     std::unique_ptr<ASTNode> parseForStatement() {
         consume(TokenType::LEFT_PAREN, "Expected '(' after 'for'");
         
         std::unique_ptr<ASTNode> initializer = nullptr;
         if (!check(TokenType::SEMICOLON)) {
             if (check(TokenType::INT) || check(TokenType::CHAR) || check(TokenType::BOOL) ||
                 check(TokenType::FLOAT) || check(TokenType::DOUBLE) || check(TokenType::VOID) ||
                 check(TokenType::AUTO) || check(TokenType::CONST)) {
                 initializer = parseVariableOrFunctionDeclaration();
             } else {
                 initializer = parseExpressionStatement();
             }
         } else {
             consume(TokenType::SEMICOLON, "Expected ';'");
         }
         
         std::unique_ptr<ASTNode> condition = nullptr;
         if (!check(TokenType::SEMICOLON)) {
             condition = parseExpression();
         }
         consume(TokenType::SEMICOLON, "Expected ';' after for condition");
         
         std::unique_ptr<ASTNode> increment = nullptr;
         if (!check(TokenType::RIGHT_PAREN)) {
             increment = parseExpression();
         }
         consume(TokenType::RIGHT_PAREN, "Expected ')' after for clauses");
         
         auto body = parseStatement();
         
         return std::make_unique<ForStmtNode>(std::move(initializer), std::move(condition),
                                            std::move(increment), std::move(body),
                                            previous().location);
     }
     
     /**
      * @brief Parse a return statement
      * @return A node representing the return statement
      */
     std::unique_ptr<ASTNode> parseReturnStatement() {
         auto location = previous().location;
         
         std::unique_ptr<ASTNode> value = nullptr;
         if (!check(TokenType::SEMICOLON)) {
             value = parseExpression();
         }
         
         consume(TokenType::SEMICOLON, "Expected ';' after return value");
         
         return std::make_unique<ReturnStmtNode>(std::move(value), location);
     }
     
     /**
      * @brief Parse an expression statement
      * @return A node representing the expression statement
      */
     std::unique_ptr<ASTNode> parseExpressionStatement() {
         auto expr = parseExpression();
         consume(TokenType::SEMICOLON, "Expected ';' after expression");
         
         return std::make_unique<ExpressionStmtNode>(std::move(expr), previous().location);
     }
     
     /**
      * @brief Parse an expression
      * @return A node representing the expression
      */
     std::unique_ptr<ASTNode> parseExpression() {
         return parseAssignment();
     }
     
     /**
      * @brief Parse an assignment expression
      * @return A node representing the assignment expression
      */
     std::unique_ptr<ASTNode> parseAssignment() {
         auto expr = parseConditional();
         
         if (match(TokenType::EQUAL) || match(TokenType::PLUS_EQUAL) || 
             match(TokenType::MINUS_EQUAL) || match(TokenType::ASTERISK_EQUAL) ||
             match(TokenType::SLASH_EQUAL) || match(TokenType::PERCENT_EQUAL) ||
             match(TokenType::AMPERSAND_EQUAL) || match(TokenType::PIPE_EQUAL) ||
             match(TokenType::CARET_EQUAL) || match(TokenType::LESS_LESS_EQUAL) ||
             match(TokenType::GREATER_GREATER_EQUAL)) {
             
             TokenType operatorType = previous().type;
             auto value = parseAssignment();
             
             AssignmentExprNode::Operator op;
             switch (operatorType) {
                 case TokenType::EQUAL: op = AssignmentExprNode::Operator::ASSIGN; break;
                 case TokenType::PLUS_EQUAL: op = AssignmentExprNode::Operator::ADD_ASSIGN; break;
                 case TokenType::MINUS_EQUAL: op = AssignmentExprNode::Operator::SUBTRACT_ASSIGN; break;
                 case TokenType::ASTERISK_EQUAL: op = AssignmentExprNode::Operator::MULTIPLY_ASSIGN; break;
                 case TokenType::SLASH_EQUAL: op = AssignmentExprNode::Operator::DIVIDE_ASSIGN; break;
                 case TokenType::PERCENT_EQUAL: op = AssignmentExprNode::Operator::MODULO_ASSIGN; break;
                 case TokenType::AMPERSAND_EQUAL: op = AssignmentExprNode::Operator::AND_ASSIGN; break;
                 case TokenType::PIPE_EQUAL: op = AssignmentExprNode::Operator::OR_ASSIGN; break;
                 case TokenType::CARET_EQUAL: op = AssignmentExprNode::Operator::XOR_ASSIGN; break;
                 case TokenType::LESS_LESS_EQUAL: op = AssignmentExprNode::Operator::LEFT_SHIFT_ASSIGN; break;
                 case TokenType::GREATER_GREATER_EQUAL: op = AssignmentExprNode::Operator::RIGHT_SHIFT_ASSIGN; break;
                 default:
                     errorReporter.reportError("Invalid assignment operator", previous().location);
                     throw std::runtime_error("Invalid assignment operator");
             }
             
             return std::make_unique<AssignmentExprNode>(op, std::move(expr), std::move(value),
                                                      previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a conditional expression (ternary operator)
      * @return A node representing the conditional expression
      */
     std::unique_ptr<ASTNode> parseConditional() {
         auto expr = parseLogicalOr();
         
         if (match(TokenType::QUESTION)) {
             auto thenBranch = parseExpression();
             consume(TokenType::COLON, "Expected ':' in conditional expression");
             auto elseBranch = parseConditional();
             
             // Simplified as a binary expression for now
             return std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::OR, 
                                                   std::move(expr), std::move(thenBranch),
                                                   previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a logical OR expression
      * @return A node representing the logical OR expression
      */
     std::unique_ptr<ASTNode> parseLogicalOr() {
         auto expr = parseLogicalAnd();
         
         while (match(TokenType::PIPE_PIPE)) {
             auto right = parseLogicalAnd();
             expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::OR, 
                                                  std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a logical AND expression
      * @return A node representing the logical AND expression
      */
     std::unique_ptr<ASTNode> parseLogicalAnd() {
         auto expr = parseBitwiseOr();
         
         while (match(TokenType::AMPERSAND_AMPERSAND)) {
             auto right = parseBitwiseOr();
             expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::AND, 
                                                  std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a bitwise OR expression
      * @return A node representing the bitwise OR expression
      */
     std::unique_ptr<ASTNode> parseBitwiseOr() {
         auto expr = parseBitwiseXor();
         
         while (match(TokenType::PIPE)) {
             auto right = parseBitwiseXor();
             expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_OR, 
                                                  std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a bitwise XOR expression
      * @return A node representing the bitwise XOR expression
      */
     std::unique_ptr<ASTNode> parseBitwiseXor() {
         auto expr = parseBitwiseAnd();
         
         while (match(TokenType::CARET)) {
             auto right = parseBitwiseAnd();
             expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_XOR, 
                                                  std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a bitwise AND expression
      * @return A node representing the bitwise AND expression
      */
     std::unique_ptr<ASTNode> parseBitwiseAnd() {
         auto expr = parseEquality();
         
         while (match(TokenType::AMPERSAND)) {
             auto right = parseEquality();
             expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_AND, 
                                                  std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse an equality expression
      * @return A node representing the equality expression
      */
     std::unique_ptr<ASTNode> parseEquality() {
         auto expr = parseComparison();
         
         while (match(TokenType::EQUAL_EQUAL) || match(TokenType::EXCLAMATION_EQUAL)) {
             TokenType operatorType = previous().type;
             auto right = parseComparison();
             
             BinaryExprNode::Operator op;
             if (operatorType == TokenType::EQUAL_EQUAL) {
                 op = BinaryExprNode::Operator::EQUAL;
             } else {
                 op = BinaryExprNode::Operator::NOT_EQUAL;
             }
             
             expr = std::make_unique<BinaryExprNode>(op, std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a comparison expression
      * @return A node representing the comparison expression
      */
     std::unique_ptr<ASTNode> parseComparison() {
         auto expr = parseShift();
         
         while (match(TokenType::LESS) || match(TokenType::LESS_EQUAL) ||
               match(TokenType::GREATER) || match(TokenType::GREATER_EQUAL)) {
             
             TokenType operatorType = previous().type;
             auto right = parseShift();
             
             BinaryExprNode::Operator op;
             switch (operatorType) {
                 case TokenType::LESS: op = BinaryExprNode::Operator::LESS; break;
                 case TokenType::LESS_EQUAL: op = BinaryExprNode::Operator::LESS_EQUAL; break;
                 case TokenType::GREATER: op = BinaryExprNode::Operator::GREATER; break;
                 case TokenType::GREATER_EQUAL: op = BinaryExprNode::Operator::GREATER_EQUAL; break;
                 default:
                     errorReporter.reportError("Invalid comparison operator", previous().location);
                     throw std::runtime_error("Invalid comparison operator");
             }
             
             expr = std::make_unique<BinaryExprNode>(op, std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a shift expression
      * @return A node representing the shift expression
      */
     std::unique_ptr<ASTNode> parseShift() {
         auto expr = parseAdditive();
         
         while (match(TokenType::LESS_LESS) || match(TokenType::GREATER_GREATER)) {
             TokenType operatorType = previous().type;
             auto right = parseAdditive();
             
             BinaryExprNode::Operator op;
             if (operatorType == TokenType::LESS_LESS) {
                 op = BinaryExprNode::Operator::LEFT_SHIFT;
             } else {
                 op = BinaryExprNode::Operator::RIGHT_SHIFT;
             }
             
             expr = std::make_unique<BinaryExprNode>(op, std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse an additive expression
      * @return A node representing the additive expression
      */
     std::unique_ptr<ASTNode> parseAdditive() {
         auto expr = parseMultiplicative();
         
         while (match(TokenType::PLUS) || match(TokenType::MINUS)) {
             TokenType operatorType = previous().type;
             auto right = parseMultiplicative();
             
             BinaryExprNode::Operator op;
             if (operatorType == TokenType::PLUS) {
                 op = BinaryExprNode::Operator::ADD;
             } else {
                 op = BinaryExprNode::Operator::SUBTRACT;
             }
             
             expr = std::make_unique<BinaryExprNode>(op, std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a multiplicative expression
      * @return A node representing the multiplicative expression
      */
     std::unique_ptr<ASTNode> parseMultiplicative() {
         auto expr = parseUnary();
         
         while (match(TokenType::ASTERISK) || match(TokenType::SLASH) || match(TokenType::PERCENT)) {
             TokenType operatorType = previous().type;
             auto right = parseUnary();
             
             BinaryExprNode::Operator op;
             switch (operatorType) {
                 case TokenType::ASTERISK: op = BinaryExprNode::Operator::MULTIPLY; break;
                 case TokenType::SLASH: op = BinaryExprNode::Operator::DIVIDE; break;
                 case TokenType::PERCENT: op = BinaryExprNode::Operator::MODULO; break;
                 default:
                     errorReporter.reportError("Invalid multiplicative operator", previous().location);
                     throw std::runtime_error("Invalid multiplicative operator");
             }
             
             expr = std::make_unique<BinaryExprNode>(op, std::move(expr), std::move(right),
                                                  previous().location);
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a unary expression
      * @return A node representing the unary expression
      */
     std::unique_ptr<ASTNode> parseUnary() {
         if (match(TokenType::EXCLAMATION) || match(TokenType::MINUS) || match(TokenType::TILDE) ||
             match(TokenType::AMPERSAND) || match(TokenType::ASTERISK) ||
             match(TokenType::PLUS_PLUS) || match(TokenType::MINUS_MINUS)) {
             
             TokenType operatorType = previous().type;
             auto right = parseUnary();
             
             UnaryExprNode::Operator op;
             switch (operatorType) {
                 case TokenType::EXCLAMATION: op = UnaryExprNode::Operator::NOT; break;
                 case TokenType::MINUS: op = UnaryExprNode::Operator::NEGATE; break;
                 case TokenType::TILDE: op = UnaryExprNode::Operator::BITWISE_NOT; break;
                 case TokenType::AMPERSAND: op = UnaryExprNode::Operator::ADDRESS_OF; break;
                 case TokenType::ASTERISK: op = UnaryExprNode::Operator::DEREFERENCE; break;
                 case TokenType::PLUS_PLUS: op = UnaryExprNode::Operator::PRE_INCREMENT; break;
                 case TokenType::MINUS_MINUS: op = UnaryExprNode::Operator::PRE_DECREMENT; break;
                 default:
                     errorReporter.reportError("Invalid unary operator", previous().location);
                     throw std::runtime_error("Invalid unary operator");
             }
             
             return std::make_unique<UnaryExprNode>(op, std::move(right), previous().location);
         }
         
         return parsePostfix();
     }
     
     /**
      * @brief Parse a postfix expression
      * @return A node representing the postfix expression
      */
     std::unique_ptr<ASTNode> parsePostfix() {
         auto expr = parsePrimary();
         
         while (match(TokenType::PLUS_PLUS) || match(TokenType::MINUS_MINUS) ||
               match(TokenType::LEFT_PAREN) || match(TokenType::LEFT_BRACKET) ||
               match(TokenType::DOT) || match(TokenType::ARROW)) {
             
             if (previous().type == TokenType::PLUS_PLUS) {
                 expr = std::make_unique<UnaryExprNode>(
                     UnaryExprNode::Operator::POST_INCREMENT,
                     std::move(expr), previous().location);
             } else if (previous().type == TokenType::MINUS_MINUS) {
                 expr = std::make_unique<UnaryExprNode>(
                     UnaryExprNode::Operator::POST_DECREMENT,
                     std::move(expr), previous().location);
             } else if (previous().type == TokenType::LEFT_PAREN) {
                 // Function call
                 std::vector<std::unique_ptr<ASTNode>> arguments;
                 
                 if (!check(TokenType::RIGHT_PAREN)) {
                     do {
                         arguments.push_back(parseExpression());
                     } while (match(TokenType::COMMA));
                 }
                 
                 consume(TokenType::RIGHT_PAREN, "Expected ')' after function call arguments");
                 
                 expr = std::make_unique<CallExprNode>(
                     std::move(expr), std::move(arguments), previous().location);
             } else if (previous().type == TokenType::LEFT_BRACKET) {
                 // Array access
                 auto index = parseExpression();
                 consume(TokenType::RIGHT_BRACKET, "Expected ']' after array index");
                 
                 // Simplified as a call expression for now
                 std::vector<std::unique_ptr<ASTNode>> arguments;
                 arguments.push_back(std::move(index));
                 
                 expr = std::make_unique<CallExprNode>(
                     std::move(expr), std::move(arguments), previous().location);
             } else if (previous().type == TokenType::DOT || previous().type == TokenType::ARROW) {
                 // Member access
                 if (!match(TokenType::IDENTIFIER)) {
                     errorReporter.reportError("Expected member name after '.' or '->'", peek().location);
                     throw std::runtime_error("Expected member name after '.' or '->'");
                 }
                 
                 std::string memberName = previous().lexeme;
                 
                 // Simplified as a variable expression for now
                 expr = std::make_unique<VariableExprNode>(memberName, previous().location);
             }
         }
         
         return expr;
     }
     
     /**
      * @brief Parse a primary expression
      * @return A node representing the primary expression
      */
     std::unique_ptr<ASTNode> parsePrimary() {
         if (match(TokenType::INTEGER_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::INTEGER,
                 previous().lexeme, previous().location);
         } else if (match(TokenType::FLOAT_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::FLOAT,
                 previous().lexeme, previous().location);
         } else if (match(TokenType::CHAR_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::CHARACTER,
                 previous().lexeme, previous().location);
         } else if (match(TokenType::STRING_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::STRING,
                 previous().lexeme, previous().location);
         } else if (match(TokenType::BOOL_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::BOOLEAN,
                 previous().lexeme, previous().location);
         } else if (match(TokenType::IDENTIFIER)) {
             return std::make_unique<VariableExprNode>(previous().lexeme, previous().location);
         } else if (match(TokenType::LEFT_PAREN)) {
             auto expr = parseExpression();
             consume(TokenType::RIGHT_PAREN, "Expected ')' after expression");
             return expr;
         }
         
         errorReporter.reportError("Expected expression", peek().location);
         throw std::runtime_error("Expected expression");
     }
     
     /**
      * @brief Utility method to check if the current token matches the expected type
      * @param type The token type to check against
      * @return True if the current token matches the expected type, false otherwise
      */
     bool match(TokenType type) {
         if (check(type)) {
             advance();
             return true;
         }
         return false;
     }
     
     /**
      * @brief Utility method to check if the current token is of the expected type
      * @param type The token type to check against
      * @return True if the current token is of the expected type, false otherwise
      */
     bool check(TokenType type) const {
         if (isAtEnd()) return false;
         return peek().type == type;
     }
     
     /**
      * @brief Utility method to consume the current token and return it
      * @return The consumed token
      */
     Token advance() {
         if (!isAtEnd()) current++;
         return previous();
     }
     
     /**
      * @brief Utility method to check if we're at the end of the token stream
      * @return True if at the end, false otherwise
      */
     bool isAtEnd() const {
         return peek().type == TokenType::END_OF_FILE;
     }
     
     /**
      * @brief Utility method to get the current token without consuming it
      * @return The current token
      */
     Token peek() const {
         return tokens[current];
     }
     
     /**
      * @brief Utility method to get the previous token
      * @return The previous token
      */
     Token previous() const {
         return tokens[current - 1];
     }
     
     /**
      * @brief Utility method to consume the current token if it matches the expected type
      * @param type The token type to check against
      * @param message The error message to display if the token doesn't match
      * @return The consumed token
      */
     Token consume(TokenType type, const std::string& message) {
         if (check(type)) return advance();
         
         errorReporter.reportError(message, peek().location);
         throw std::runtime_error(message);
     }
     
     /**
      * @brief Utility method to synchronize after an error
      * 
      * This method skips tokens until it finds a token that can be the start of a new statement.
      */
     void synchronize() {
         advance();
         
         while (!isAtEnd()) {
             if (previous().type == TokenType::SEMICOLON) return;
             
             switch (peek().type) {
                 case TokenType::CLASS:
                 case TokenType::STRUCT:
                 case TokenType::ENUM:
                 case TokenType::FUNCTION:
                 case TokenType::IF:
                 case TokenType::WHILE:
                 case TokenType::FOR:
                 case TokenType::RETURN:
                     return;
                 default:
                     // Do nothing
                     break;
             }
             
             advance();
         }
     }