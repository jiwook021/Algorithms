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
     ErrorReporter ErrorReporter;
     bool Verbose = false;
     
 public:
     Compiler(bool Verbose = false) : Verbose(Verbose) {}
     
     /**
      * @brief Compile a C++ source file to x86-64 assembly
      * @param inputFile The input C++ source file
      * @param outputFile The output assembly file
      * @return True if compilation was successful, false otherwise
      */
     bool Compile(const std::string& InputFile, const std::string& OutputFile) {
         try {
             // Read the input file
             std::string Source = ReadFile(InputFile);
             
             if (Verbose) {
                 std::cout << "Source code loaded (" << Source.length() << " bytes)" << std::endl;
             }
             
             // Lexical analysis
             Lexer Lexer(Source, InputFile, ErrorReporter);
             std::vector<Token> Tokens = Lexer.ScanTokens();
             
             if (Verbose) {
                 std::cout << "Lexical analysis completed (" << Tokens.size() << " tokens)" << std::endl;
             }
             
             if (ErrorReporter.HadError()) {
                 std::cerr << "Compilation failed during lexical analysis" << std::endl;
                 return false;
             }
             
             // Parsing
             Parser Parser(Tokens, ErrorReporter);
             auto Ast = Parser.Parse();
             
             if (Verbose) {
                 std::cout << "Parsing completed" << std::endl;
             }
             
             if (ErrorReporter.HadError()) {
                 std::cerr << "Compilation failed during parsing" << std::endl;
                 return false;
             }
             
             // Semantic analysis
             SemanticAnalyzer SemanticAnalyzer(ErrorReporter);
             bool SemanticSuccess = SemanticAnalyzer.Analyze(Ast);
             
             if (Verbose) {
                 std::cout << "Semantic analysis completed" << std::endl;
             }
             
             if (!SemanticSuccess) {
                 std::cerr << "Compilation failed during semantic analysis" << std::endl;
                 return false;
             }
             
             // IR generation (simplified for now)
             std::shared_ptr<Module> module = GenerateIR(Ast);
             
             if (Verbose) {
                 std::cout << "IR generation completed" << std::endl;
             }
             
             // Optimization
             Optimizer Optimizer(ErrorReporter);
             Optimizer.Optimize(module);
             
             if (Verbose) {
                 std::cout << "Optimization completed" << std::endl;
             }
             
             // Code generation
             CodeGenerator CodeGenerator(ErrorReporter);
             std::string Assembly = CodeGenerator.GenerateCode(module);
             
             if (Verbose) {
                 std::cout << "Code generation completed" << std::endl;
             }
             
             // Write the output file
             WriteFile(OutputFile, Assembly);
             
             if (Verbose) {
                 std::cout << "Assembly code written to " << OutputFile << std::endl;
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
     std::string ReadFile(const std::string& Filename) {
         std::ifstream File(Filename, std::ios::binary | std::ios::ate);
         
         if (!File) {
             throw std::runtime_error("Could not open file: " + Filename);
         }
         
         std::streamsize size = File.tellg();
         File.seekg(0, std::ios::beg);
         
         std::string Buffer(size, ' ');
         if (!File.read(Buffer.data(), size)) {
             throw std::runtime_error("Could not read file: " + Filename);
         }
         
         return Buffer;
     }
     
     /**
      * @brief Write a string to a file
      * @param filename The file to write to
      * @param content The content to write
      */
     void WriteFile(const std::string& Filename, const std::string& Content) {
         std::ofstream File(Filename);
         
         if (!File) {
             throw std::runtime_error("Could not open file for writing: " + Filename);
         }
         
         File << Content;
         
         if (!File) {
             throw std::runtime_error("Could not write to file: " + Filename);
         }
     }
     
     /**
      * @brief Generate IR from the AST
      * @param ast The AST
      * @return The generated IR module
      */
     std::shared_ptr<Module> GenerateIR(const std::unique_ptr<ProgramNode>& Ast) {
         auto module = std::make_shared<Module>("main_module");
         
         // Process each declaration in the AST
         for (const auto& Decl : Ast->Declarations) {
             if (Decl->Type == ASTNodeType::FUNCTION_DECL) {
                 auto FuncDecl = static_cast<const FunctionDeclNode*>(Decl.get());
                 GenerateFunctionIR(module, FuncDecl);
             }
             // Other declarations like global variables, classes, etc.
             // would be handled here in a full compiler
         }
         
         // If no main function was found, create a minimal one
         bool HasMain = false;
         for (const auto& Func : module->Functions) {
             if (Func->Name == "main") {
                 HasMain = true;
                 break;
             }
         }
         
         if (!HasMain) {
             // Create a main function
             auto MainType = std::make_shared<FunctionType>(
                 std::make_shared<Type>(Type::TypeKind::INT),
                 std::vector<std::shared_ptr<Type>>()
             );
             
             auto MainFunction = std::make_shared<Function>("main", MainType);
             
             // Create entry block
             auto EntryBlock = std::make_shared<BasicBlock>("entry");
             
             // Create a return 0 instruction
             auto RetInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
             auto ReturnValue = std::make_shared<Value>(Value::ValueType::INTEGER);
             RetInst->AddOperand(ReturnValue);
             
             // Add instruction to block
             EntryBlock->AddInstruction(RetInst);
             
             // Add block to function
             MainFunction->AddBlock(EntryBlock);
             
             // Add function to module
             module->AddFunction(MainFunction);
         }
         
         return module;
     }
     
     /**
      * @brief Generate IR for a function declaration
      * @param module The module to add the function to
      * @param funcDecl The function declaration AST node
      */
     void GenerateFunctionIR(std::shared_ptr<Module> module, const FunctionDeclNode* FuncDecl) {
         // Create function type and function in the IR
         auto function = std::make_shared<Function>(FuncDecl->Name, FuncDecl->Type);
         
         // Create entry block
         auto EntryBlock = std::make_shared<BasicBlock>("entry");
         function->AddBlock(EntryBlock);
         
         // Track the current block being generated
         std::shared_ptr<BasicBlock> CurrentBlock = EntryBlock;
         
         // Create a symbol table for this function's scope
         std::unordered_map<std::string, std::shared_ptr<Value>> SymbolTable;
         
         // Allocate space for parameters
         for (size_t i = 0; i < FuncDecl->Parameters.size(); i++) {
             // For simplicity, just create an alloca instruction for each parameter
             auto AllocaInst = std::make_shared<Instruction>(Instruction::OpCode::ALLOCA);
             auto ParamValue = std::make_shared<Value>(Value::ValueType::POINTER);
             AllocaInst->SetResult(ParamValue);
             
             CurrentBlock->AddInstruction(AllocaInst);
             
             // Store the parameter value
             auto StoreInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
             auto ArgValue = std::make_shared<Value>(Value::ValueType::INTEGER); // Simplified
             StoreInst->AddOperand(ArgValue);
             StoreInst->AddOperand(ParamValue);
             
             CurrentBlock->AddInstruction(StoreInst);
             
             // Add to symbol table
             std::string ParamName = "param" + std::to_string(i); // Simplified
             SymbolTable[ParamName] = ParamValue;
         }
         
         // Generate IR for the function body if it exists
         if (FuncDecl->Body) {
             GenerateStatementIR(function, CurrentBlock, FuncDecl->Body.get(), SymbolTable);
         }
         
         // If the function doesn't end with a return, add one
         if (CurrentBlock->Instructions.empty() || 
             CurrentBlock->Instructions.back()->Opcode != Instruction::OpCode::RET) {
             
             auto RetInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
             
             // If the function has a return type, provide a default return value
             if (FuncDecl->Type->ReturnType->Kind != Type::TypeKind::VOID) {
                 auto DefaultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 RetInst->AddOperand(DefaultValue);
             }
             
             CurrentBlock->AddInstruction(RetInst);
         }
         
         // Add function to module
         module->AddFunction(function);
     }
     
     /**
      * @brief Generate IR for a statement
      * @param function The current function
      * @param currentBlock The current basic block
      * @param stmt The statement AST node
      * @param symbolTable The symbol table for variable lookup
      * @return The next basic block to use
      */
     std::shared_ptr<BasicBlock> GenerateStatementIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> CurrentBlock,
         const ASTNode* Stmt,
         std::unordered_map<std::string, std::shared_ptr<Value>>& SymbolTable
     ) {
         switch (Stmt->Type) {
             case ASTNodeType::COMPOUND_STMT: {
                 const CompoundStmtNode* CompoundStmt = static_cast<const CompoundStmtNode*>(Stmt);
                 
                 // Create a new scope
                 std::unordered_map<std::string, std::shared_ptr<Value>> InnerSymbolTable = SymbolTable;
                 
                 // Generate IR for each statement in sequence
                 for (const auto& SubStmt : CompoundStmt->Statements) {
                     CurrentBlock = GenerateStatementIR(function, CurrentBlock, SubStmt.get(), InnerSymbolTable);
                     
                     // If we've ended the block (e.g., with a return), stop
                     if (CurrentBlock->Instructions.size() > 0 &&
                         (CurrentBlock->Instructions.back()->Opcode == Instruction::OpCode::RET ||
                          CurrentBlock->Instructions.back()->Opcode == Instruction::OpCode::BR)) {
                         break;
                     }
                 }
                 
                 return CurrentBlock;
             }
             
             case ASTNodeType::EXPRESSION_STMT: {
                 const ExpressionStmtNode* ExprStmt = static_cast<const ExpressionStmtNode*>(Stmt);
                 
                 // Generate IR for the expression
                 GenerateExpressionIR(function, CurrentBlock, ExprStmt->Expression.get(), SymbolTable);
                 
                 return CurrentBlock;
             }
             
             case ASTNodeType::IF_STMT: {
                 const IfStmtNode* IfStmt = static_cast<const IfStmtNode*>(Stmt);
                 
                 // Generate IR for the condition
                 auto CondValue = GenerateExpressionIR(function, CurrentBlock, IfStmt->Condition.get(), SymbolTable);
                 
                 // Create then and else blocks
                 auto ThenBlock = std::make_shared<BasicBlock>("then" + std::to_string(function->Blocks.size()));
                 auto ElseBlock = std::make_shared<BasicBlock>("else" + std::to_string(function->Blocks.size() + 1));
                 auto MergeBlock = std::make_shared<BasicBlock>("merge" + std::to_string(function->Blocks.size() + 2));
                 
                 // Add blocks to function
                 function->AddBlock(ThenBlock);
                 function->AddBlock(ElseBlock);
                 function->AddBlock(MergeBlock);
                 
                 // Update CFG
                 CurrentBlock->Successors.push_back(ThenBlock);
                 CurrentBlock->Successors.push_back(ElseBlock);
                 ThenBlock->Predecessors.push_back(CurrentBlock);
                 ElseBlock->Predecessors.push_back(CurrentBlock);
                 ThenBlock->Successors.push_back(MergeBlock);
                 ElseBlock->Successors.push_back(MergeBlock);
                 MergeBlock->Predecessors.push_back(ThenBlock);
                 MergeBlock->Predecessors.push_back(ElseBlock);
                 
                 // Create conditional branch
                 auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 BrInst->AddOperand(CondValue);
                 auto ThenValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto ElseValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 BrInst->AddOperand(ThenValue);
                 BrInst->AddOperand(ElseValue);
                 
                 // Add branch to current block
                 CurrentBlock->AddInstruction(BrInst);
                 
                 // Generate IR for then branch
                 auto ThenEnd = GenerateStatementIR(function, ThenBlock, IfStmt->ThenBranch.get(), SymbolTable);
                 
                 // Add branch to merge block if needed
                 if (ThenEnd->Instructions.empty() ||
                     ThenEnd->Instructions.back()->Opcode != Instruction::OpCode::BR) {
                     auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     auto MergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                     BrInst->AddOperand(MergeValue);
                     ThenEnd->AddInstruction(BrInst);
                 }
                 
                 // Generate IR for else branch if it exists
                 if (IfStmt->ElseBranch) {
                     auto ElseEnd = GenerateStatementIR(function, ElseBlock, IfStmt->ElseBranch.get(), SymbolTable);
                     
                     // Add branch to merge block if needed
                     if (ElseEnd->Instructions.empty() ||
                         ElseEnd->Instructions.back()->Opcode != Instruction::OpCode::BR) {
                         auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                         auto MergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         BrInst->AddOperand(MergeValue);
                         ElseEnd->AddInstruction(BrInst);
                     }
                 } else {
                     // Empty else branch, just branch to merge
                     auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     auto MergeValue = std::make_shared<Value>(Value::ValueType::POINTER);
                     BrInst->AddOperand(MergeValue);
                     ElseBlock->AddInstruction(BrInst);
                 }
                 
                 return MergeBlock;
             }
             
             case ASTNodeType::WHILE_STMT: {
                 const WhileStmtNode* WhileStmt = static_cast<const WhileStmtNode*>(Stmt);
                 
                 // Create loop header, body, and exit blocks
                 auto HeaderBlock = std::make_shared<BasicBlock>("loop_header" + std::to_string(function->Blocks.size()));
                 auto BodyBlock = std::make_shared<BasicBlock>("loop_body" + std::to_string(function->Blocks.size() + 1));
                 auto ExitBlock = std::make_shared<BasicBlock>("loop_exit" + std::to_string(function->Blocks.size() + 2));
                 
                 // Add blocks to function
                 function->AddBlock(HeaderBlock);
                 function->AddBlock(BodyBlock);
                 function->AddBlock(ExitBlock);
                 
                 // Update CFG
                 CurrentBlock->Successors.push_back(HeaderBlock);
                 HeaderBlock->Predecessors.push_back(CurrentBlock);
                 HeaderBlock->Successors.push_back(BodyBlock);
                 HeaderBlock->Successors.push_back(ExitBlock);
                 BodyBlock->Predecessors.push_back(HeaderBlock);
                 BodyBlock->Successors.push_back(HeaderBlock);
                 HeaderBlock->Predecessors.push_back(BodyBlock);
                 ExitBlock->Predecessors.push_back(HeaderBlock);
                 
                 // Branch to loop header
                 auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto HeaderValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 BrInst->AddOperand(HeaderValue);
                 CurrentBlock->AddInstruction(BrInst);
                 
                 // Generate IR for condition in header
                 auto CondValue = GenerateExpressionIR(function, HeaderBlock, WhileStmt->Condition.get(), SymbolTable);
                 
                 // Create conditional branch
                 auto CondBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 CondBrInst->AddOperand(CondValue);
                 auto BodyValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto ExitValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 CondBrInst->AddOperand(BodyValue);
                 CondBrInst->AddOperand(ExitValue);
                 
                 // Add branch to header block
                 HeaderBlock->AddInstruction(CondBrInst);
                 
                 // Generate IR for loop body
                 auto BodyEnd = GenerateStatementIR(function, BodyBlock, WhileStmt->Body.get(), SymbolTable);
                 
                 // Branch back to header
                 auto LoopBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 LoopBrInst->AddOperand(HeaderValue);
                 BodyEnd->AddInstruction(LoopBrInst);
                 
                 return ExitBlock;
             }
             
             case ASTNodeType::FOR_STMT: {
                 const ForStmtNode* ForStmt = static_cast<const ForStmtNode*>(Stmt);
                 
                 // Generate IR for initializer
                 if (ForStmt->Initializer) {
                     if (ForStmt->Initializer->Type == ASTNodeType::VARIABLE_DECL) {
                         // Handle variable declaration initializer
                         GenerateVariableDeclIR(function, CurrentBlock, 
                                              static_cast<const VariableDeclNode*>(ForStmt->Initializer.get()), 
                                              SymbolTable);
                     } else {
                         // Handle expression initializer
                         GenerateExpressionIR(function, CurrentBlock, ForStmt->Initializer.get(), SymbolTable);
                     }
                 }
                 
                 // Create loop header, body, increment, and exit blocks
                 auto HeaderBlock = std::make_shared<BasicBlock>("for_header" + std::to_string(function->Blocks.size()));
                 auto BodyBlock = std::make_shared<BasicBlock>("for_body" + std::to_string(function->Blocks.size() + 1));
                 auto IncBlock = std::make_shared<BasicBlock>("for_inc" + std::to_string(function->Blocks.size() + 2));
                 auto ExitBlock = std::make_shared<BasicBlock>("for_exit" + std::to_string(function->Blocks.size() + 3));
                 
                 // Add blocks to function
                 function->AddBlock(HeaderBlock);
                 function->AddBlock(BodyBlock);
                 function->AddBlock(IncBlock);
                 function->AddBlock(ExitBlock);
                 
                 // Update CFG
                 CurrentBlock->Successors.push_back(HeaderBlock);
                 HeaderBlock->Predecessors.push_back(CurrentBlock);
                 HeaderBlock->Predecessors.push_back(IncBlock);
                 HeaderBlock->Successors.push_back(BodyBlock);
                 HeaderBlock->Successors.push_back(ExitBlock);
                 BodyBlock->Predecessors.push_back(HeaderBlock);
                 BodyBlock->Successors.push_back(IncBlock);
                 IncBlock->Predecessors.push_back(BodyBlock);
                 IncBlock->Successors.push_back(HeaderBlock);
                 ExitBlock->Predecessors.push_back(HeaderBlock);
                 
                 // Branch to loop header
                 auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto HeaderValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 BrInst->AddOperand(HeaderValue);
                 CurrentBlock->AddInstruction(BrInst);
                 
                 // Generate IR for condition in header
                 std::shared_ptr<Value> CondValue;
                 if (ForStmt->Condition) {
                     CondValue = GenerateExpressionIR(function, HeaderBlock, ForStmt->Condition.get(), SymbolTable);
                 } else {
                     // If no condition, use true
                     CondValue = std::make_shared<Value>(Value::ValueType::BOOLEAN);
                 }
                 
                 // Create conditional branch
                 auto CondBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR_COND);
                 CondBrInst->AddOperand(CondValue);
                 auto BodyValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 auto ExitValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 CondBrInst->AddOperand(BodyValue);
                 CondBrInst->AddOperand(ExitValue);
                 
                 // Add branch to header block
                 HeaderBlock->AddInstruction(CondBrInst);
                 
                 // Generate IR for loop body
                 auto BodyEnd = GenerateStatementIR(function, BodyBlock, ForStmt->Body.get(), SymbolTable);
                 
                 // Branch to increment block
                 auto BodyBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 auto IncValue = std::make_shared<Value>(Value::ValueType::POINTER);
                 BodyBrInst->AddOperand(IncValue);
                 BodyEnd->AddInstruction(BodyBrInst);
                 
                 // Generate IR for increment
                 if (ForStmt->Increment) {
                     GenerateExpressionIR(function, IncBlock, ForStmt->Increment.get(), SymbolTable);
                 }
                 
                 // Branch back to header
                 auto IncBrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                 IncBrInst->AddOperand(HeaderValue);
                 IncBlock->AddInstruction(IncBrInst);
                 
                 return ExitBlock;
             }
             
             case ASTNodeType::RETURN_STMT: {
                 const ReturnStmtNode* ReturnStmt = static_cast<const ReturnStmtNode*>(Stmt);
                 
                 // Create return instruction
                 auto RetInst = std::make_shared<Instruction>(Instruction::OpCode::RET);
                 
                 // If there's a return value, generate IR for it
                 if (ReturnStmt->value) {
                     auto ReturnValue = GenerateExpressionIR(function, CurrentBlock, ReturnStmt->value.get(), SymbolTable);
                     RetInst->AddOperand(ReturnValue);
                 }
                 
                 // Add return instruction to current block
                 CurrentBlock->AddInstruction(RetInst);
                 
                 return CurrentBlock;
             }
             
             case ASTNodeType::VARIABLE_DECL: {
                 GenerateVariableDeclIR(function, CurrentBlock, 
                                      static_cast<const VariableDeclNode*>(Stmt), 
                                      SymbolTable);
                 return CurrentBlock;
             }
             
             default:
                 // For other statement types, we would add similar logic
                 return CurrentBlock;
         }
     }
     
     /**
      * @brief Generate IR for a variable declaration
      * @param function The current function
      * @param currentBlock The current basic block
      * @param varDecl The variable declaration AST node
      * @param symbolTable The symbol table for variable lookup
      */
     void GenerateVariableDeclIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> CurrentBlock,
         const VariableDeclNode* VarDecl,
         std::unordered_map<std::string, std::shared_ptr<Value>>& SymbolTable
     ) {
         // Create alloca instruction
         auto AllocaInst = std::make_shared<Instruction>(Instruction::OpCode::ALLOCA);
         auto VarValue = std::make_shared<Value>(Value::ValueType::POINTER);
         AllocaInst->SetResult(VarValue);
         
         // Add to symbol table
         SymbolTable[VarDecl->Name] = VarValue;
         
         // Add alloca instruction to current block
         CurrentBlock->AddInstruction(AllocaInst);
         
         // If there's an initializer, generate IR for it and store the result
         if (VarDecl->Initializer) {
             auto InitValue = GenerateExpressionIR(function, CurrentBlock, VarDecl->Initializer.get(), SymbolTable);
             
             // Create store instruction
             auto StoreInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
             StoreInst->AddOperand(InitValue);
             StoreInst->AddOperand(VarValue);
             
             // Add store instruction to current block
             CurrentBlock->AddInstruction(StoreInst);
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
     std::shared_ptr<Value> GenerateExpressionIR(
         std::shared_ptr<Function> function,
         std::shared_ptr<BasicBlock> CurrentBlock,
         const ASTNode* Expr,
         std::unordered_map<std::string, std::shared_ptr<Value>>& SymbolTable
     ) {
         switch (Expr->Type) {
             case ASTNodeType::BINARY_EXPR: {
                 const BinaryExprNode* BinExpr = static_cast<const BinaryExprNode*>(Expr);
                 
                 // Generate IR for left and right operands
                 auto LeftValue = GenerateExpressionIR(function, CurrentBlock, BinExpr->left.get(), SymbolTable);
                 auto RightValue = GenerateExpressionIR(function, CurrentBlock, BinExpr->right.get(), SymbolTable);
                 
                 // Create instruction based on operator
                 std::shared_ptr<Instruction> Inst;
                 
                 switch (BinExpr->Op) {
                     case BinaryExprNode::Operator::ADD:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                         break;
                     case BinaryExprNode::Operator::SUBTRACT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         break;
                     case BinaryExprNode::Operator::MULTIPLY:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::MUL);
                         break;
                     case BinaryExprNode::Operator::DIVIDE:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::DIV);
                         break;
                     case BinaryExprNode::Operator::MODULO:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::MOD);
                         break;
                     case BinaryExprNode::Operator::EQUAL:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::EQ);
                         break;
                     case BinaryExprNode::Operator::NOT_EQUAL:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::NE);
                         break;
                     case BinaryExprNode::Operator::LESS:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::LT);
                         break;
                     case BinaryExprNode::Operator::LESS_EQUAL:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::LE);
                         break;
                     case BinaryExprNode::Operator::GREATER:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::GT);
                         break;
                     case BinaryExprNode::Operator::GREATER_EQUAL:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::GE);
                         break;
                     case BinaryExprNode::Operator::AND:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                         break;
                     case BinaryExprNode::Operator::OR:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                         break;
                     case BinaryExprNode::Operator::BITWISE_AND:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                         break;
                     case BinaryExprNode::Operator::BITWISE_OR:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                         break;
                     case BinaryExprNode::Operator::BITWISE_XOR:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         break;
                     case BinaryExprNode::Operator::LEFT_SHIFT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::SHL);
                         break;
                     case BinaryExprNode::Operator::RIGHT_SHIFT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::SHR);
                         break;
                     default:
                         // Unsupported operator
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 // Add operands and set result
                 Inst->AddOperand(LeftValue);
                 Inst->AddOperand(RightValue);
                 auto ResultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 Inst->SetResult(ResultValue);
                 
                 // Add instruction to current block
                 CurrentBlock->AddInstruction(Inst);
                 
                 return ResultValue;
             }
             
             case ASTNodeType::UNARY_EXPR: {
                 const UnaryExprNode* UnaryExpr = static_cast<const UnaryExprNode*>(Expr);
                 
                 // Generate IR for operand
                 auto OperandValue = GenerateExpressionIR(function, CurrentBlock, UnaryExpr->Operand.get(), SymbolTable);
                 
                 // Create instruction based on operator
                 std::shared_ptr<Instruction> Inst;
                 
                 switch (UnaryExpr->Op) {
                     case UnaryExprNode::Operator::NEGATE:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         {
                             // Create a zero value for negation (0 - operand)
                             auto ZeroValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             Inst->AddOperand(ZeroValue);
                             Inst->AddOperand(OperandValue);
                         }
                         break;
                     case UnaryExprNode::Operator::NOT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         {
                             // Create a one value for logical NOT (operand XOR 1)
                             auto OneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             Inst->AddOperand(OperandValue);
                             Inst->AddOperand(OneValue);
                         }
                         break;
                     case UnaryExprNode::Operator::BITWISE_NOT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                         {
                             // Create a -1 value for bitwise NOT (operand XOR -1)
                             auto AllOnesValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             Inst->AddOperand(OperandValue);
                             Inst->AddOperand(AllOnesValue);
                         }
                         break;
                     case UnaryExprNode::Operator::ADDRESS_OF:
                         // No instruction needed, the operand itself is the address
                         return OperandValue;
                     case UnaryExprNode::Operator::DEREFERENCE:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::LOAD);
                         Inst->AddOperand(OperandValue);
                         break;
                     case UnaryExprNode::Operator::PRE_INCREMENT:
                     case UnaryExprNode::Operator::POST_INCREMENT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                         {
                             // Create a one value for increment (operand + 1)
                             auto OneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             Inst->AddOperand(OperandValue);
                             Inst->AddOperand(OneValue);
                         }
                         break;
                     case UnaryExprNode::Operator::PRE_DECREMENT:
                     case UnaryExprNode::Operator::POST_DECREMENT:
                         Inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                         {
                             // Create a one value for decrement (operand - 1)
                             auto OneValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                             Inst->AddOperand(OperandValue);
                             Inst->AddOperand(OneValue);
                         }
                         break;
                     default:
                         // Unsupported operator
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 // Set result
                 auto ResultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                 Inst->SetResult(ResultValue);
                 
                 // Add instruction to current block
                 CurrentBlock->AddInstruction(Inst);
                 
                 // For increment/decrement, store the result back to the variable
                 if (UnaryExpr->Op == UnaryExprNode::Operator::PRE_INCREMENT ||
                     UnaryExpr->Op == UnaryExprNode::Operator::PRE_DECREMENT ||
                     UnaryExpr->Op == UnaryExprNode::Operator::POST_INCREMENT ||
                     UnaryExpr->Op == UnaryExprNode::Operator::POST_DECREMENT) {
                     
                     // Create store instruction
                     auto StoreInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
                     StoreInst->AddOperand(ResultValue);
                     
                     // For simplicity, assume the operand is a variable expression
                     // In a real compiler, we would handle more complex lvalues
                     if (UnaryExpr->Operand->Type == ASTNodeType::VARIABLE_EXPR) {
                         const VariableExprNode* VarExpr = static_cast<const VariableExprNode*>(UnaryExpr->Operand.get());
                         auto VarValue = SymbolTable[VarExpr->Name];
                         StoreInst->AddOperand(VarValue);
                         
                         // Add store instruction to current block
                         CurrentBlock->AddInstruction(StoreInst);
                     }
                     
                     // For post-increment/decrement, return the original value
                     if (UnaryExpr->Op == UnaryExprNode::Operator::POST_INCREMENT ||
                         UnaryExpr->Op == UnaryExprNode::Operator::POST_DECREMENT) {
                         return OperandValue;
                     }
                 }
                 
                 return ResultValue;
             }
             
             case ASTNodeType::LITERAL_EXPR: {
                 const LiteralExprNode* LiteralExpr = static_cast<const LiteralExprNode*>(Expr);
                 
                 // Create a value for the literal
                 std::shared_ptr<Value> LiteralValue;
                 
                 switch (LiteralExpr->LiteralType) {
                     case LiteralExprNode::LiteralType::INTEGER:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                         break;
                     case LiteralExprNode::LiteralType::FLOAT:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::FLOAT);
                         break;
                     case LiteralExprNode::LiteralType::CHARACTER:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                         break;
                     case LiteralExprNode::LiteralType::STRING:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         break;
                     case LiteralExprNode::LiteralType::BOOLEAN:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::BOOLEAN);
                         break;
                     case LiteralExprNode::LiteralType::NULL_LITERAL:
                         LiteralValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         break;
                     default:
                         // Unsupported literal type
                         return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
                 
                 return LiteralValue;
             }
             
             case ASTNodeType::VARIABLE_EXPR: {
                 const VariableExprNode* VarExpr = static_cast<const VariableExprNode*>(Expr);
                 
                 // Look up variable in symbol table
                 auto It = SymbolTable.find(VarExpr->Name);
                 if (It != SymbolTable.end()) {
                     // Create load instruction to get the value
                     auto LoadInst = std::make_shared<Instruction>(Instruction::OpCode::LOAD);
                     LoadInst->AddOperand(It->second);
                     auto ResultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     LoadInst->SetResult(ResultValue);
                     
                     // Add load instruction to current block
                     CurrentBlock->AddInstruction(LoadInst);
                     
                     return ResultValue;
                 } else {
                     // Variable not found
                     return std::make_shared<Value>(Value::ValueType::INTEGER);
                 }
             }
             
             case ASTNodeType::ASSIGNMENT_EXPR: {
                 const AssignmentExprNode* AssignExpr = static_cast<const AssignmentExprNode*>(Expr);
                 
                 // Generate IR for right operand
                 auto RightValue = GenerateExpressionIR(function, CurrentBlock, AssignExpr->right.get(), SymbolTable);
                 
                 // Handle special assignment operators (+=, -=, etc.)
                 if (AssignExpr->Op != AssignmentExprNode::Operator::ASSIGN) {
                     // Generate IR for left operand
                     auto LeftValue = GenerateExpressionIR(function, CurrentBlock, AssignExpr->left.get(), SymbolTable);
                     
                     // Create instruction based on operator
                     std::shared_ptr<Instruction> Inst;
                     
                     switch (AssignExpr->Op) {
                         case AssignmentExprNode::Operator::ADD_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                             break;
                         case AssignmentExprNode::Operator::SUBTRACT_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::SUB);
                             break;
                         case AssignmentExprNode::Operator::MULTIPLY_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::MUL);
                             break;
                         case AssignmentExprNode::Operator::DIVIDE_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::DIV);
                             break;
                         case AssignmentExprNode::Operator::MODULO_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::MOD);
                             break;
                         case AssignmentExprNode::Operator::AND_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::AND);
                             break;
                         case AssignmentExprNode::Operator::OR_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::OR);
                             break;
                         case AssignmentExprNode::Operator::XOR_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::XOR);
                             break;
                         case AssignmentExprNode::Operator::LEFT_SHIFT_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::SHL);
                             break;
                         case AssignmentExprNode::Operator::RIGHT_SHIFT_ASSIGN:
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::SHR);
                             break;
                         default:
                             // Should not happen
                             Inst = std::make_shared<Instruction>(Instruction::OpCode::ADD);
                             break;
                     }
                     
                     // Add operands and set result
                     Inst->AddOperand(LeftValue);
                     Inst->AddOperand(RightValue);
                     auto ResultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     Inst->SetResult(ResultValue);
                     
                     // Add instruction to current block
                     CurrentBlock->AddInstruction(Inst);
                     
                     // Update rightValue to be the result of the operation
                     RightValue = ResultValue;
                 }
                 
                 // Create store instruction
                 auto StoreInst = std::make_shared<Instruction>(Instruction::OpCode::STORE);
                 StoreInst->AddOperand(RightValue);
                 
                 // Handling different types of left operands (variable, dereference, array access, etc.)
                 if (AssignExpr->left->Type == ASTNodeType::VARIABLE_EXPR) {
                     const VariableExprNode* VarExpr = static_cast<const VariableExprNode*>(AssignExpr->left.get());
                     
                     // Look up variable in symbol table
                     auto It = SymbolTable.find(VarExpr->Name);
                     if (It != SymbolTable.end()) {
                         StoreInst->AddOperand(It->second);
                     } else {
                         // Variable not found, create a new one
                         auto VarValue = std::make_shared<Value>(Value::ValueType::POINTER);
                         SymbolTable[VarExpr->Name] = VarValue;
                         StoreInst->AddOperand(VarValue);
                     }
                 } else {
                     // For more complex left operands (dereferenced pointers, array accesses, etc.)
                     // we would need to generate the appropriate IR
                     // For simplicity, just generate a dummy location
                     auto DummyLoc = std::make_shared<Value>(Value::ValueType::POINTER);
                     StoreInst->AddOperand(DummyLoc);
                 }
                 
                 // Add store instruction to current block
                 CurrentBlock->AddInstruction(StoreInst);
                 
                 return RightValue;
             }
             
             case ASTNodeType::CALL_EXPR: {
                 const CallExprNode* CallExpr = static_cast<const CallExprNode*>(Expr);
                 
                 // Generate IR for arguments
                 std::vector<std::shared_ptr<Value>> ArgValues;
                 for (const auto& Arg : CallExpr->Arguments) {
                     ArgValues.push_back(GenerateExpressionIR(function, CurrentBlock, Arg.get(), SymbolTable));
                 }
                 
                 // Create call instruction
                 auto CallInst = std::make_shared<Instruction>(Instruction::OpCode::CALL);
                 
                 // Add callee and arguments
                 // For simplicity, assume the callee is a variable expression
                 if (CallExpr->Callee->Type == ASTNodeType::VARIABLE_EXPR) {
                     const VariableExprNode* VarExpr = static_cast<const VariableExprNode*>(CallExpr->Callee.get());
                     
                     // Create a dummy value for the function
                     auto FuncValue = std::make_shared<Value>(Value::ValueType::FUNCTION);
                     CallInst->AddOperand(FuncValue);
                     
                     // Add arguments
                     for (auto& ArgValue : ArgValues) {
                         CallInst->AddOperand(ArgValue);
                     }
                     
                     // Set result
                     auto ResultValue = std::make_shared<Value>(Value::ValueType::INTEGER);
                     CallInst->SetResult(ResultValue);
                     
                     // Add call instruction to current block
                     CurrentBlock->AddInstruction(CallInst);
                     
                     return ResultValue;
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
         std::string InputFile;
         std::string OutputFile;
         bool Verbose = false;
         bool Optimize = true;
         int OptimizationLevel = 1;
         bool ShowHelp = false;
         bool ShowVersion = false;
         
         // Parse command line arguments
         for (int i = 1; i < argc; i++) {
             std::string Arg = argv[i];
             
             if (Arg == "-h" || Arg == "--help") {
                 ShowHelp = true;
             } else if (Arg == "-v" || Arg == "--verbose") {
                 Verbose = true;
             } else if (Arg == "--version") {
                 ShowVersion = true;
             } else if (Arg == "-O0") {
                 Optimize = false;
                 OptimizationLevel = 0;
             } else if (Arg == "-O1") {
                 Optimize = true;
                 OptimizationLevel = 1;
             } else if (Arg == "-O2") {
                 Optimize = true;
                 OptimizationLevel = 2;
             } else if (Arg == "-O3") {
                 Optimize = true;
                 OptimizationLevel = 3;
             } else if (Arg == "-o" && i + 1 < argc) {
                 OutputFile = argv[++i];
             } else if (Arg[0] == '-') {
                 std::cerr << "Unknown option: " << Arg << std::endl;
                 ShowHelp = true;
             } else {
                 if (InputFile.empty()) {
                     InputFile = Arg;
                 } else if (OutputFile.empty()) {
                     OutputFile = Arg;
                 } else {
                     std::cerr << "Too many arguments" << std::endl;
                     ShowHelp = true;
                 }
             }
         }
         
         // Show help message
         if (ShowHelp) {
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
         if (ShowVersion) {
             std::cout << "SimpleCppCompiler version 1.0.0" << std::endl;
             std::cout << "Built on " << __DATE__ << " " << __TIME__ << std::endl;
             return 0;
         }
         
         // Check for required arguments
         if (InputFile.empty()) {
             std::cerr << "Error: No input file specified" << std::endl;
             std::cerr << "Use --help for more information" << std::endl;
             return 1;
         }
         
         // Set default output file if not specified
         if (OutputFile.empty()) {
             size_t DotPos = InputFile.find_last_of('.');
             if (DotPos != std::string::npos) {
                 OutputFile = InputFile.substr(0, DotPos) + ".asm";
             } else {
                 OutputFile = InputFile + ".asm";
             }
         }
         
         // Create compiler with options
         Compiler Compiler(Verbose);
         
         // Additional compiler options based on command line arguments
         // In a real compiler, we would set more options here
         
         // Print compilation options if verbose
         if (Verbose) {
             std::cout << "Input file: " << InputFile << std::endl;
             std::cout << "Output file: " << OutputFile << std::endl;
             std::cout << "Optimization level: " << OptimizationLevel << std::endl;
         }
         
         // Run the compiler
         auto StartTime = std::chrono::high_resolution_clock::now();
         bool Success = Compiler.Compile(InputFile, OutputFile);
         auto EndTime = std::chrono::high_resolution_clock::now();
         
         // Calculate compilation time
         auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();
         
         if (Success) {
             std::cout << "Compilation successful: " << InputFile << " -> " << OutputFile << std::endl;
             
             if (Verbose) {
                 std::cout << "Compilation time: " << Duration << " ms" << std::endl;
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
     ErrorReporter& ErrorReporter;
     std::stringstream Output;
     std::unordered_map<std::string, size_t> LocalVars;
     size_t StackSize = 0;
     size_t LabelCounter = 0;
     
 public:
     CodeGenerator(ErrorReporter& ErrorReporter) : ErrorReporter(ErrorReporter) {}
     
     /**
      * @brief Generate assembly code from the intermediate representation
      * @param module The module to generate code for
      * @return The generated assembly code
      */
     std::string GenerateCode(std::shared_ptr<Module> module) {
         Output.str("");
         
         // Generate assembly header
         GenerateHeader(module->Name);
         
         // Generate code for each function
         for (auto& function : module->Functions) {
             GenerateFunction(function);
         }
         
         return Output.str();
     }
     
 private:
     /**
      * @brief Generate assembly header
      * @param moduleName The name of the module
      */
     void GenerateHeader(const std::string& ModuleName) {
         Output << "; Generated assembly for module: " << ModuleName << "\n";
         Output << "; Generated by SimpleCppCompiler\n\n";
         
         Output << "section .text\n";
         Output << "global main\n\n";
         
         // Import external functions
         Output << "extern printf\n";
         Output << "extern scanf\n";
         Output << "extern malloc\n";
         Output << "extern free\n\n";
         
         // String literals
         Output << "section .data\n";
         Output << "format_int db \"%d\", 0\n";
         Output << "format_float db \"%f\", 0\n";
         Output << "format_char db \"%c\", 0\n";
         Output << "format_string db \"%s\", 0\n";
         Output << "format_bool_true db \"true\", 0\n";
         Output << "format_bool_false db \"false\", 0\n\n";
         
         Output << "section .text\n\n";
     }
     
     /**
      * @brief Generate assembly code for a function
      * @param function The function to generate code for
      */
     void GenerateFunction(std::shared_ptr<Function> function) {
         // Reset local variables and stack size
         LocalVars.clear();
         StackSize = 0;
         
         // Function label
         Output << function->Name << ":\n";
         
         // Function prologue
         Output << "    push rbp\n";
         Output << "    mov rbp, rsp\n";
         
         // Allocate stack space for local variables
         // In a real compiler, we would calculate this based on the variables used
         Output << "    sub rsp, 64\n";  // Allocate 64 bytes for local variables
         
         // Generate code for each basic block
         for (auto& Block : function->Blocks) {
             GenerateBasicBlock(Block);
         }
         
         // Function epilogue
         // This is just a default epilogue; in a real compiler, the actual return point
         // would depend on the control flow
         Output << "    mov rsp, rbp\n";
         Output << "    pop rbp\n";
         Output << "    ret\n\n";
     }
     
     /**
      * @brief Generate assembly code for a basic block
      * @param block The basic block to generate code for
      */
     void GenerateBasicBlock(std::shared_ptr<BasicBlock> Block) {
         Output << Block->Label << ":\n";
         
         // Generate code for each instruction
         for (auto& Instruction : Block->Instructions) {
             GenerateInstruction(Instruction);
         }
     }
     
     /**
      * @brief Generate assembly code for an instruction
      * @param instruction The instruction to generate code for
      */
     void GenerateInstruction(std::shared_ptr<Instruction> Instruction) {
         switch (Instruction->Opcode) {
             case Instruction::OpCode::ADD:
                 GenerateAdd(Instruction);
                 break;
             case Instruction::OpCode::SUB:
                 GenerateSub(Instruction);
                 break;
             case Instruction::OpCode::MUL:
                 GenerateMul(Instruction);
                 break;
             case Instruction::OpCode::DIV:
                 GenerateDiv(Instruction);
                 break;
             case Instruction::OpCode::MOD:
                 GenerateMod(Instruction);
                 break;
             case Instruction::OpCode::AND:
                 GenerateAnd(Instruction);
                 break;
             case Instruction::OpCode::OR:
                 GenerateOr(Instruction);
                 break;
             case Instruction::OpCode::XOR:
                 GenerateXor(Instruction);
                 break;
             case Instruction::OpCode::SHL:
                 GenerateShl(Instruction);
                 break;
             case Instruction::OpCode::SHR:
                 GenerateShr(Instruction);
                 break;
             case Instruction::OpCode::EQ:
                 GenerateEq(Instruction);
                 break;
             case Instruction::OpCode::NE:
                 GenerateNe(Instruction);
                 break;
             case Instruction::OpCode::LT:
                 GenerateLt(Instruction);
                 break;
             case Instruction::OpCode::LE:
                 GenerateLe(Instruction);
                 break;
             case Instruction::OpCode::GT:
                 GenerateGt(Instruction);
                 break;
             case Instruction::OpCode::GE:
                 GenerateGe(Instruction);
                 break;
             case Instruction::OpCode::ALLOCA:
                 GenerateAlloca(Instruction);
                 break;
             case Instruction::OpCode::LOAD:
                 GenerateLoad(Instruction);
                 break;
             case Instruction::OpCode::STORE:
                 GenerateStore(Instruction);
                 break;
             case Instruction::OpCode::BR:
                 GenerateBr(Instruction);
                 break;
             case Instruction::OpCode::BR_COND:
                 GenerateBrCond(Instruction);
                 break;
             case Instruction::OpCode::CALL:
                 GenerateCall(Instruction);
                 break;
             case Instruction::OpCode::RET:
                 GenerateRet(Instruction);
                 break;
             case Instruction::OpCode::PHI:
                 GeneratePhi(Instruction);
                 break;
             case Instruction::OpCode::CAST:
                 GenerateCast(Instruction);
                 break;
             default:
                 ErrorReporter.ReportError("Unknown instruction opcode", SourceLocation());
                 break;
         }
     }
     
     /**
      * @brief Generate assembly code for an ADD instruction
      * @param instruction The ADD instruction
      */
     void GenerateAdd(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("ADD instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Add second operand
         Output << "    add rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a SUB instruction
      * @param instruction The SUB instruction
      */
     void GenerateSub(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("SUB instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Subtract second operand
         Output << "    sub rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a MUL instruction
      * @param instruction The MUL instruction
      */
     void GenerateMul(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("MUL instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Multiply by second operand
         Output << "    imul rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a DIV instruction
      * @param instruction The DIV instruction
      */
     void GenerateDiv(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("DIV instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Clear RDX for division
         Output << "    xor rdx, rdx\n";
         
         // Load second operand into RCX
         Output << "    mov rcx, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Divide
         Output << "    div rcx\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a MOD instruction
      * @param instruction The MOD instruction
      */
     void GenerateMod(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("MOD instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Clear RDX for division
         Output << "    xor rdx, rdx\n";
         
         // Load second operand into RCX
         Output << "    mov rcx, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Divide
         Output << "    div rcx\n";
         
         // Store remainder (modulo)
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rdx\n";
     }
     
     /**
      * @brief Generate assembly code for an AND instruction
      * @param instruction The AND instruction
      */
     void GenerateAnd(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("AND instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // AND with second operand
         Output << "    and rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an OR instruction
      * @param instruction The OR instruction
      */
     void GenerateOr(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("OR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // OR with second operand
         Output << "    or rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an XOR instruction
      * @param instruction The XOR instruction
      */
     void GenerateXor(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("XOR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // XOR with second operand
         Output << "    xor rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an SHL instruction
      * @param instruction The SHL instruction
      */
     void GenerateShl(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("SHL instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Load second operand into RCX (shift count)
         Output << "    mov rcx, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Shift left
         Output << "    shl rax, cl\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an SHR instruction
      * @param instruction The SHR instruction
      */
     void GenerateShr(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("SHR instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Load second operand into RCX (shift count)
         Output << "    mov rcx, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Shift right
         Output << "    shr rax, cl\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an EQ instruction
      * @param instruction The EQ instruction
      */
     void GenerateEq(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("EQ instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    sete al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an NE instruction
      * @param instruction The NE instruction
      */
     void GenerateNe(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("NE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    setne al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an LT instruction
      * @param instruction The LT instruction
      */
     void GenerateLt(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("LT instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    setl al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an LE instruction
      * @param instruction The LE instruction
      */
     void GenerateLe(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("LE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    setle al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a GT instruction
      * @param instruction The GT instruction
      */
     void GenerateGt(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("GT instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    setg al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a GE instruction
      * @param instruction The GE instruction
      */
     void GenerateGe(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("GE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load first operand into RAX
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with second operand
         Output << "    cmp rax, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Set result based on comparison
         Output << "    setge al\n";
         Output << "    movzx rax, al\n";
         
         // Store result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for an ALLOCA instruction
      * @param instruction The ALLOCA instruction
      */
     void GenerateAlloca(std::shared_ptr<Instruction> Instruction) {
         // Allocate space on the stack for a variable
         // In a real compiler, we would use the size operand to determine how much space to allocate
         
         // For now, just reserve 8 bytes
         StackSize += 8;
         
         // Store the offset for the result
         if (Instruction->Result) {
             LocalVars[std::to_string(reinterpret_cast<uintptr_t>(Instruction->Result.get()))] = StackSize;
         }
     }
     
     /**
      * @brief Generate assembly code for a LOAD instruction
      * @param instruction The LOAD instruction
      */
     void GenerateLoad(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have one operand
         if (Instruction->Operands.size() != 1) {
             ErrorReporter.ReportError("LOAD instruction requires one operand", SourceLocation());
             return;
         }
         
         // Load the value from the address in the operand
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         Output << "    mov rax, [rax]\n";
         
         // Store the loaded value
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a STORE instruction
      * @param instruction The STORE instruction
      */
     void GenerateStore(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have two operands
         if (Instruction->Operands.size() != 2) {
             ErrorReporter.ReportError("STORE instruction requires two operands", SourceLocation());
             return;
         }
         
         // Load the value to be stored
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Load the address to store to
         Output << "    mov rcx, [rbp - " << GetOperandOffset(Instruction->Operands[1]) << "]\n";
         
         // Store the value
         Output << "    mov [rcx], rax\n";
     }
     
     /**
      * @brief Generate assembly code for a BR instruction
      * @param instruction The BR instruction
      */
     void GenerateBr(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have one operand
         if (Instruction->Operands.size() != 1) {
             ErrorReporter.ReportError("BR instruction requires one operand", SourceLocation());
             return;
         }
         
         // Branch to the specified label
         // In a real compiler, we would extract the label from the operand
         std::string Label = "label" + std::to_string(LabelCounter++);
         
         Output << "    jmp " << Label << "\n";
     }
     
     /**
      * @brief Generate assembly code for a BR_COND instruction
      * @param instruction The BR_COND instruction
      */
     void GenerateBrCond(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have three operands
         if (Instruction->Operands.size() != 3) {
             ErrorReporter.ReportError("BR_COND instruction requires three operands", SourceLocation());
             return;
         }
         
         // Load the condition
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // Compare with zero
         Output << "    cmp rax, 0\n";
         
         // Branch based on the condition
         // In a real compiler, we would extract the labels from the operands
         std::string TrueLabel = "label" + std::to_string(LabelCounter++);
         std::string FalseLabel = "label" + std::to_string(LabelCounter++);
         
         Output << "    jne " << TrueLabel << "\n";
         Output << "    jmp " << FalseLabel << "\n";
     }
     
     /**
      * @brief Generate assembly code for a CALL instruction
      * @param instruction The CALL instruction
      */
     void GenerateCall(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have at least one operand
         if (Instruction->Operands.size() < 1) {
             ErrorReporter.ReportError("CALL instruction requires at least one operand", SourceLocation());
             return;
         }
         
         // The first operand is the function to call
         // In a real compiler, we would extract the function name from the operand
         std::string FunctionName = "function" + std::to_string(LabelCounter++);
         
         // Save caller-saved registers
         Output << "    push rcx\n";
         Output << "    push rdx\n";
         Output << "    push rsi\n";
         Output << "    push rdi\n";
         Output << "    push r8\n";
         Output << "    push r9\n";
         Output << "    push r10\n";
         Output << "    push r11\n";
         
         // Load arguments into registers according to the x86-64 calling convention
         // In a real compiler, we would extract the arguments from the operands
         
         // First 6 arguments go in registers: RDI, RSI, RDX, RCX, R8, R9
         // Additional arguments are pushed on the stack in reverse order
         
         for (size_t i = 1; i < Instruction->Operands.size() && i <= 6; i++) {
             std::string Reg;
             
             switch (i) {
                 case 1: Reg = "rdi"; break;
                 case 2: Reg = "rsi"; break;
                 case 3: Reg = "rdx"; break;
                 case 4: Reg = "rcx"; break;
                 case 5: Reg = "r8"; break;
                 case 6: Reg = "r9"; break;
                 default: Reg = ""; break; // Shouldn't happen
             }
             
             if (!Reg.empty()) {
                 Output << "    mov " << Reg << ", [rbp - " << 
                     GetOperandOffset(Instruction->Operands[i]) << "]\n";
             }
         }
         
         // Call the function
         Output << "    call " << FunctionName << "\n";
         
         // Store the return value
         if (Instruction->Result) {
             Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
         }
         
         // Restore caller-saved registers
         Output << "    pop r11\n";
         Output << "    pop r10\n";
         Output << "    pop r9\n";
         Output << "    pop r8\n";
         Output << "    pop rdi\n";
         Output << "    pop rsi\n";
         Output << "    pop rdx\n";
         Output << "    pop rcx\n";
     }
     
     /**
      * @brief Generate assembly code for a RET instruction
      * @param instruction The RET instruction
      */
     void GenerateRet(std::shared_ptr<Instruction> Instruction) {
         // If we have an operand, it's the return value
         if (Instruction->Operands.size() >= 1) {
             Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         }
         
         // Function epilogue and return
         Output << "    mov rsp, rbp\n";
         Output << "    pop rbp\n";
         Output << "    ret\n";
     }
     
     /**
      * @brief Generate assembly code for a PHI instruction
      * @param instruction The PHI instruction
      */
     void GeneratePhi(std::shared_ptr<Instruction> Instruction) {
         // PHI instructions are handled during basic block generation
         // For simplicity, we'll just set the result to the first operand's value
         
         if (Instruction->Operands.size() >= 1) {
             Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
             Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
         }
     }
     
     /**
      * @brief Generate assembly code for a CAST instruction
      * @param instruction The CAST instruction
      */
     void GenerateCast(std::shared_ptr<Instruction> Instruction) {
         // Ensure we have at least one operand
         if (Instruction->Operands.size() < 1) {
             ErrorReporter.ReportError("CAST instruction requires at least one operand", SourceLocation());
             return;
         }
         
         // Load the value to cast
         Output << "    mov rax, [rbp - " << GetOperandOffset(Instruction->Operands[0]) << "]\n";
         
         // For now, we'll just copy the value
         // In a real compiler, we would handle different types of casts
         
         // Store the result
         Output << "    mov [rbp - " << GetResultOffset(Instruction->Result) << "], rax\n";
     }
     
     /**
      * @brief Get the stack offset for an operand
      * @param operand The operand
      * @return The stack offset
      */
     size_t GetOperandOffset(std::shared_ptr<Value> Operand) {
         // In a real compiler, we would track variable locations
         // For now, just use a map from operand addresses to stack offsets
         
         std::string key = std::to_string(reinterpret_cast<uintptr_t>(Operand.get()));
         
         auto It = LocalVars.find(key);
         if (It != LocalVars.end()) {
             return It->second;
         }
         
         // If we don't have an offset for this operand, allocate one
         StackSize += 8;
         LocalVars[key] = StackSize;
         
         return StackSize;
     }
     
     /**
      * @brief Get the stack offset for a result
      * @param result The result
      * @return The stack offset
      */
     size_t GetResultOffset(std::shared_ptr<Value> Result) {
         return GetOperandOffset(Result); // Same as operand offset
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
     
     ValueType Type;
     
     Value(ValueType Type) : Type(Type) {}
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
     
     OpCode Opcode;
     std::vector<std::shared_ptr<Value>> Operands;
     std::shared_ptr<Value> Result;
     
     Instruction(OpCode Opcode) : Opcode(Opcode) {}
     
     void AddOperand(std::shared_ptr<Value> Operand) {
         Operands.push_back(Operand);
     }
     
     void SetResult(std::shared_ptr<Value> Result) {
         this->Result = Result;
     }
 };
 
 /**
  * @brief Basic block in the intermediate representation
  */
 class BasicBlock {
 public:
     std::string Label;
     std::vector<std::shared_ptr<Instruction>> Instructions;
     std::vector<std::shared_ptr<BasicBlock>> Predecessors;
     std::vector<std::shared_ptr<BasicBlock>> Successors;
     
     BasicBlock(const std::string& Label) : Label(Label) {}
     
     void AddInstruction(std::shared_ptr<Instruction> Instruction) {
         Instructions.push_back(Instruction);
     }
     
     void AddPredecessor(std::shared_ptr<BasicBlock> Predecessor) {
         Predecessors.push_back(Predecessor);
     }
     
     void AddSuccessor(std::shared_ptr<BasicBlock> Successor) {
         Successors.push_back(Successor);
     }
 };
 
 /**
  * @brief Function in the intermediate representation
  */
 class Function {
 public:
     std::string Name;
     std::shared_ptr<FunctionType> Type;
     std::vector<std::shared_ptr<BasicBlock>> Blocks;
     
     Function(const std::string& Name, std::shared_ptr<FunctionType> Type)
         : Name(Name), Type(Type) {}
     
     void AddBlock(std::shared_ptr<BasicBlock> Block) {
         Blocks.push_back(Block);
     }
 };
 
 /**
  * @brief Module representing a compilation unit
  */
 class Module {
 public:
     std::string Name;
     std::vector<std::shared_ptr<Function>> Functions;
     
     Module(const std::string& Name) : Name(Name) {}
     
     void AddFunction(std::shared_ptr<Function> function) {
         Functions.push_back(function);
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
     ErrorReporter& ErrorReporter;
     
 public:
     Optimizer(ErrorReporter& ErrorReporter) : ErrorReporter(ErrorReporter) {}
     
     /**
      * @brief Optimize a module
      * @param module The module to optimize
      */
     void Optimize(std::shared_ptr<Module> module) {
         for (auto& function : module->Functions) {
             OptimizeFunction(function);
         }
     }
     
 private:
     /**
      * @brief Optimize a function
      * @param function The function to optimize
      */
     void OptimizeFunction(std::shared_ptr<Function> function) {
         // Apply various optimization passes
         EliminateDeadCode(function);
         ConstantFolding(function);
         ConstantPropagation(function);
         SimplifyControlFlow(function);
     }
     
     /**
      * @brief Eliminate dead code in a function
      * @param function The function to optimize
      */
     void EliminateDeadCode(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Mark all instructions as potentially dead
             std::unordered_map<std::shared_ptr<Instruction>, bool> IsLive;
             
             // First pass: mark instructions with side effects as live
             for (auto& Block : function->Blocks) {
                 for (auto& Instruction : Block->Instructions) {
                     // Instructions with side effects
                     if (Instruction->Opcode == Instruction::OpCode::STORE ||
                         Instruction->Opcode == Instruction::OpCode::CALL ||
                         Instruction->Opcode == Instruction::OpCode::RET) {
                         IsLive[Instruction] = true;
                     } else {
                         IsLive[Instruction] = false;
                     }
                 }
             }
             
             // Second pass: propagate liveness backward
             bool LocalChanged = true;
             while (LocalChanged) {
                 LocalChanged = false;
                 
                 for (auto& Block : function->Blocks) {
                     for (auto& Instruction : Block->Instructions) {
                         if (IsLive[Instruction]) continue;
                         
                         // Check if this instruction's result is used by a live instruction
                         bool Used = false;
                         
                         for (auto& OtherBlock : function->Blocks) {
                             for (auto& OtherInstruction : OtherBlock->Instructions) {
                                 if (!IsLive[OtherInstruction]) continue;
                                 
                                 for (auto& Operand : OtherInstruction->Operands) {
                                     if (Operand == Instruction->Result) {
                                         Used = true;
                                         break;
                                     }
                                 }
                                 
                                 if (Used) break;
                             }
                             
                             if (Used) break;
                         }
                         
                         if (Used) {
                             IsLive[Instruction] = true;
                             LocalChanged = true;
                             Changed = true;
                         }
                     }
                 }
             }
             
             // Third pass: remove dead instructions
             for (auto& Block : function->Blocks) {
                 auto It = Block->Instructions.begin();
                 while (It != Block->Instructions.end()) {
                     if (!IsLive[*It]) {
                         It = Block->Instructions.erase(It);
                         Changed = true;
                     } else {
                         ++It;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform constant folding in a function
      * @param function The function to optimize
      */
     void ConstantFolding(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             for (auto& Block : function->Blocks) {
                 auto It = Block->Instructions.begin();
                 while (It != Block->Instructions.end()) {
                     auto& Instruction = *It;
                     
                     // Check if all operands are constants
                     bool AllConstant = true;
                     
                     // This is a simplified version that doesn't actually check for constants
                     // In a real optimizer, we would have a way to identify constant values
                     
                     if (AllConstant) {
                         // Evaluate the instruction at compile time
                         // In a real optimizer, we would actually compute the result
                         
                         // Replace uses of this instruction with the constant result
                         // In a real optimizer, we would update all uses
                         
                         // Remove the instruction
                         It = Block->Instructions.erase(It);
                         Changed = true;
                     } else {
                         ++It;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform constant propagation in a function
      * @param function The function to optimize
      */
     void ConstantPropagation(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Map of values to their constant values (if known)
             std::unordered_map<std::shared_ptr<Value>, std::shared_ptr<Value>> Constants;
             
             // First pass: identify constant values
             for (auto& Block : function->Blocks) {
                 for (auto& Instruction : Block->Instructions) {
                     // Check for constant assignment instructions
                     if (Instruction->Opcode == Instruction::OpCode::ALLOCA && 
                         Instruction->Operands.size() >= 1) {
                         
                         // Check if the operand is a constant value
                         // In a real optimizer, we would have proper constant detection
                         if (Instruction->Operands[0]->Type == Value::ValueType::INTEGER ||
                             Instruction->Operands[0]->Type == Value::ValueType::FLOAT ||
                             Instruction->Operands[0]->Type == Value::ValueType::BOOLEAN) {
                             Constants[Instruction->Result] = Instruction->Operands[0];
                         }
                     }
                 }
             }
             
             // Second pass: propagate constants
             for (auto& Block : function->Blocks) {
                 for (auto& Instruction : Block->Instructions) {
                     // Replace operands with constants if known
                     for (size_t i = 0; i < Instruction->Operands.size(); i++) {
                         auto It = Constants.find(Instruction->Operands[i]);
                         if (It != Constants.end()) {
                             Instruction->Operands[i] = It->second;
                             Changed = true;
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
     void CommonSubexpressionElimination(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Map to track expressions that compute the same value
             std::unordered_map<std::string, std::shared_ptr<Value>> ExpressionMap;
             
             for (auto& Block : function->Blocks) {
                 // Reset the map for each basic block (local CSE)
                 ExpressionMap.clear();
                 
                 auto It = Block->Instructions.begin();
                 while (It != Block->Instructions.end()) {
                     auto& Instruction = *It;
                     
                     // Skip instructions with side effects
                     if (Instruction->Opcode == Instruction::OpCode::STORE ||
                         Instruction->Opcode == Instruction::OpCode::CALL ||
                         Instruction->Opcode == Instruction::OpCode::RET) {
                         ++It;
                         continue;
                     }
                     
                     // Create a key for the instruction
                     std::string key = std::to_string(static_cast<int>(Instruction->Opcode));
                     for (auto& Operand : Instruction->Operands) {
                         // In a real compiler, we would have a proper way to identify values
                         key += "_" + std::to_string(reinterpret_cast<uintptr_t>(Operand.get()));
                     }
                     
                     // Check if we've seen this expression before
                     auto MapIt = ExpressionMap.find(key);
                     if (MapIt != ExpressionMap.end()) {
                         // Replace all uses of this instruction's result with the previous result
                         for (auto& OtherBlock : function->Blocks) {
                             for (auto& OtherInstruction : OtherBlock->Instructions) {
                                 for (size_t i = 0; i < OtherInstruction->Operands.size(); i++) {
                                     if (OtherInstruction->Operands[i] == Instruction->Result) {
                                         OtherInstruction->Operands[i] = MapIt->second;
                                         Changed = true;
                                     }
                                 }
                             }
                         }
                         
                         // Remove the redundant instruction
                         It = Block->Instructions.erase(It);
                     } else {
                         // Add this expression to the map
                         ExpressionMap[key] = Instruction->Result;
                         ++It;
                     }
                 }
             }
         }
     }
     
     /**
      * @brief Perform loop invariant code motion in a function
      * @param function The function to optimize
      */
     void LoopInvariantCodeMotion(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Identify loops in the function
             std::vector<std::vector<std::shared_ptr<BasicBlock>>> Loops;
             IdentifyLoops(function, Loops);
             
             // For each loop
             for (auto& Loop : Loops) {
                 if (Loop.empty()) continue;
                 
                 // Find the loop header
                 auto Header = Loop[0];
                 
                 // Find loop pre-header (entry block to the loop)
                 std::shared_ptr<BasicBlock> PreHeader = nullptr;
                 for (auto& Block : function->Blocks) {
                     if (std::find(Block->Successors.begin(), Block->Successors.end(), Header) != Block->Successors.end() &&
                         std::find(Loop.begin(), Loop.end(), Block) == Loop.end()) {
                         PreHeader = Block;
                         break;
                     }
                 }
                 
                 if (!PreHeader) {
                     // Create a pre-header if it doesn't exist
                     PreHeader = std::make_shared<BasicBlock>("loop_preheader");
                     
                     // Update the function's blocks
                     auto It = std::find(function->Blocks.begin(), function->Blocks.end(), Header);
                     if (It != function->Blocks.end()) {
                         function->Blocks.insert(It, PreHeader);
                     } else {
                         function->Blocks.push_back(PreHeader);
                     }
                     
                     // Update the CFG
                     for (auto& Block : function->Blocks) {
                         auto It = std::find(Block->Successors.begin(), Block->Successors.end(), Header);
                         if (It != Block->Successors.end() && std::find(Loop.begin(), Loop.end(), Block) == Loop.end()) {
                             // Replace header with pre-header in the successor list
                             *It = PreHeader;
                             
                             // Add block to pre-header's predecessors
                             PreHeader->Predecessors.push_back(Block);
                         }
                     }
                     
                     // Add header to pre-header's successors
                     PreHeader->Successors.push_back(Header);
                     
                     // Add pre-header to header's predecessors
                     auto It = std::find(Header->Predecessors.begin(), Header->Predecessors.end(), PreHeader);
                     if (It == Header->Predecessors.end()) {
                         Header->Predecessors.push_back(PreHeader);
                     }
                     
                     // Generate a branch instruction to the header
                     auto BrInst = std::make_shared<Instruction>(Instruction::OpCode::BR);
                     BrInst->AddOperand(std::make_shared<Value>(Value::ValueType::POINTER)); // dummy operand
                     PreHeader->AddInstruction(BrInst);
                 }
                 
                 // Find loop-invariant instructions
                 std::vector<std::shared_ptr<Instruction>> InvariantInsts;
                 
                 for (auto& Block : Loop) {
                     auto It = Block->Instructions.begin();
                     while (It != Block->Instructions.end()) {
                         auto& Instruction = *It;
                         
                         // Skip instructions with side effects
                         if (Instruction->Opcode == Instruction::OpCode::STORE ||
                             Instruction->Opcode == Instruction::OpCode::CALL ||
                             Instruction->Opcode == Instruction::OpCode::RET) {
                             ++It;
                             continue;
                         }
                         
                         // Check if all operands are loop-invariant
                         bool IsInvariant = true;
                         for (auto& Operand : Instruction->Operands) {
                             // Check if the operand is defined outside the loop or is loop-invariant
                             bool OperandInvariant = false;
                             
                             // In a real compiler, we would have proper data flow analysis
                             // For now, just assume all operands are not invariant
                             OperandInvariant = false;
                             
                             if (!OperandInvariant) {
                                 IsInvariant = false;
                                 break;
                             }
                         }
                         
                         if (IsInvariant) {
                             // Add to the list of invariant instructions
                             InvariantInsts.push_back(Instruction);
                             
                             // Move the instruction to the pre-header
                             PreHeader->Instructions.insert(PreHeader->Instructions.end() - 1, Instruction);
                             
                             // Remove from the current block
                             It = Block->Instructions.erase(It);
                             Changed = true;
                         } else {
                             ++It;
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
     void InlineFunctions(std::shared_ptr<Module> module) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Find candidate functions for inlining
             std::unordered_set<std::shared_ptr<Function>> InlineCandidates;
             
             for (auto& function : module->Functions) {
                 // Skip the main function
                 if (function->Name == "main") continue;
                 
                 // Check if the function is small enough to inline
                 // This is a simplified heuristic; in a real compiler, we would consider
                 // function size, call frequency, etc.
                 if (function->Blocks.size() <= 3) {
                     InlineCandidates.insert(function);
                 }
             }
             
             // For each function in the module
             for (auto& function : module->Functions) {
                 // For each block in the function
                 for (auto& Block : function->Blocks) {
                     auto It = Block->Instructions.begin();
                     while (It != Block->Instructions.end()) {
                         auto& Instruction = *It;
                         
                         // Check if it's a call instruction
                         if (Instruction->Opcode == Instruction::OpCode::CALL && 
                             Instruction->Operands.size() >= 1) {
                             
                             // Find the called function
                             std::shared_ptr<Function> CalledFunction = nullptr;
                             
                             // In a real compiler, we would have a proper way to resolve function references
                             // For now, just assume we can't find the function
                             
                             // Check if the function is a candidate for inlining
                             if (CalledFunction && InlineCandidates.find(CalledFunction) != InlineCandidates.end()) {
                                 // Perform function inlining
                                 // In a real compiler, we would clone the callee's body, rewrite variable
                                 // references, handle returns, etc.
                                 
                                 // Mark as changed
                                 Changed = true;
                             } else {
                                 ++It;
                             }
                         } else {
                             ++It;
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
     void SimplifyControlFlow(std::shared_ptr<Function> function) {
         bool Changed = true;
         
         while (Changed) {
             Changed = false;
             
             // Remove empty blocks
             auto BlockIt = function->Blocks.begin();
             while (BlockIt != function->Blocks.end()) {
                 auto& Block = *BlockIt;
                 
                 // Skip blocks with instructions
                 if (!Block->Instructions.empty()) {
                     ++BlockIt;
                     continue;
                 }
                 
                 // Skip blocks with multiple predecessors or successors
                 if (Block->Predecessors.size() != 1 || Block->Successors.size() != 1) {
                     ++BlockIt;
                     continue;
                 }
                 
                 // Get the predecessor and successor
                 auto Pred = Block->Predecessors[0];
                 auto Succ = Block->Successors[0];
                 
                 // Update the CFG
                 auto It = std::find(Pred->Successors.begin(), Pred->Successors.end(), Block);
                 if (It != Pred->Successors.end()) {
                     *It = Succ;
                 }
                 
                 It = std::find(Succ->Predecessors.begin(), Succ->Predecessors.end(), Block);
                 if (It != Succ->Predecessors.end()) {
                     *It = Pred;
                 }
                 
                 // Remove the empty block
                 BlockIt = function->Blocks.erase(BlockIt);
                 Changed = true;
             }
             
             // Merge blocks with a single predecessor and successor
             BlockIt = function->Blocks.begin();
             while (BlockIt != function->Blocks.end()) {
                 auto& Block = *BlockIt;
                 
                 // Skip blocks with multiple predecessors or successors
                 if (Block->Predecessors.size() != 1 || Block->Successors.size() != 1) {
                     ++BlockIt;
                     continue;
                 }
                 
                 // Get the predecessor
                 auto Pred = Block->Predecessors[0];
                 
                 // Skip if the predecessor has multiple successors
                 if (Pred->Successors.size() != 1) {
                     ++BlockIt;
                     continue;
                 }
                 
                 // Merge the blocks
                 // Move instructions from block to the end of pred
                 Pred->Instructions.insert(Pred->Instructions.end(),
                                           Block->Instructions.begin(),
                                           Block->Instructions.end());
                 
                 // Update pred's successors
                 Pred->Successors = Block->Successors;
                 
                 // Update the successors' predecessors
                 for (auto& Succ : Block->Successors) {
                     auto It = std::find(Succ->Predecessors.begin(), Succ->Predecessors.end(), Block);
                     if (It != Succ->Predecessors.end()) {
                         *It = Pred;
                     }
                 }
                 
                 // Remove the merged block
                 BlockIt = function->Blocks.erase(BlockIt);
                 Changed = true;
             }
         }
     }
     
     /**
      * @brief Identify loops in a function
      * @param function The function
      * @param loops Output parameter to store the identified loops
      */
     void IdentifyLoops(std::shared_ptr<Function> function, 
                        std::vector<std::vector<std::shared_ptr<BasicBlock>>>& Loops) {
         // Build a map of dominators
         std::unordered_map<std::shared_ptr<BasicBlock>, std::unordered_set<std::shared_ptr<BasicBlock>>> Dominators;
         BuildDominators(function, Dominators);
         
         // For each block in the function
         for (auto& Block : function->Blocks) {
             // For each successor of the block
             for (auto& Succ : Block->Successors) {
                 // If the successor dominates the block, it's a back edge
                 if (Dominators[Block].find(Succ) != Dominators[Block].end()) {
                     // Identify the loop
                     std::vector<std::shared_ptr<BasicBlock>> Loop;
                     Loop.push_back(Succ); // Header
                     
                     // Add all blocks in the loop
                     std::function<void(std::shared_ptr<BasicBlock>)> AddToLoop =
                         [&](std::shared_ptr<BasicBlock> Current) {
                             if (Current != Succ && 
                                 std::find(Loop.begin(), Loop.end(), Current) == Loop.end()) {
                                 Loop.push_back(Current);
                                 
                                 for (auto& Pred : Current->Predecessors) {
                                     AddToLoop(Pred);
                                 }
                             }
                         };
                     
                     AddToLoop(Block);
                     
                     Loops.push_back(Loop);
                 }
             }
         }
     }
     
     /**
      * @brief Build the dominator sets for a function
      * @param function The function
      * @param dominators Output parameter to store the dominators
      */
     void BuildDominators(std::shared_ptr<Function> function,
                          std::unordered_map<std::shared_ptr<BasicBlock>, 
                                            std::unordered_set<std::shared_ptr<BasicBlock>>>& Dominators) {
         // Initialize all blocks to be dominated by all blocks
         for (auto& Block : function->Blocks) {
             Dominators[Block] = std::unordered_set<std::shared_ptr<BasicBlock>>();
             
             // Add all blocks as potential dominators, except for the entry block
             for (auto& OtherBlock : function->Blocks) {
                 if (Block != function->Blocks[0]) {
                     Dominators[Block].insert(OtherBlock);
                 }
             }
         }
         
         // The entry block is only dominated by itself
         Dominators[function->Blocks[0]].clear();
         Dominators[function->Blocks[0]].insert(function->Blocks[0]);
         
         bool Changed = true;
         while (Changed) {
             Changed = false;
             
             // For each block (except the entry block)
             for (size_t i = 1; i < function->Blocks.size(); i++) {
                 auto Block = function->Blocks[i];
                 std::unordered_set<std::shared_ptr<BasicBlock>> NewDominators = Dominators[Block];
                 
                 // Intersect the dominators of all predecessors
                 for (auto& Pred : Block->Predecessors) {
                     std::unordered_set<std::shared_ptr<BasicBlock>> Intersection;
                     
                     for (auto& Dom : Dominators[Pred]) {
                         if (Dominators[Block].find(Dom) != Dominators[Block].end()) {
                             Intersection.insert(Dom);
                         }
                     }
                     
                     NewDominators = Intersection;
                 }
                 
                 // Add the block itself to its dominators
                 NewDominators.insert(Block);
                 
                 // If the dominators changed, we need to continue iterating
                 if (NewDominators != Dominators[Block]) {
                     Dominators[Block] = NewDominators;
                     Changed = true;
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
     std::shared_ptr<SymbolTable> SymbolTable;
     ErrorReporter& ErrorReporter;
     
 public:
     SemanticAnalyzer(ErrorReporter& ErrorReporter)
         : ErrorReporter(ErrorReporter) {
         SymbolTable = std::make_shared<SymbolTable>();
     }
     
     /**
      * @brief Analyze the AST for semantic correctness
      * @param root The root node of the AST
      * @return True if semantic analysis was successful, false otherwise
      */
     bool Analyze(const std::unique_ptr<ASTNode>& Root) {
         try {
             // Define built-in types
             DefineBuiltInTypes();
             
             // Visit the AST
             VisitNode(Root);
             
             return !ErrorReporter.HadError();
         } catch (const std::exception& e) {
             ErrorReporter.ReportError(e.what(), SourceLocation());
             return false;
         }
     }
 
 private:
     /**
      * @brief Define built-in types in the symbol table
      */
     void DefineBuiltInTypes() {
         // Define primitive types
         SymbolTable->DefineType("void", std::make_shared<Type>(Type::TypeKind::VOID), SourceLocation());
         SymbolTable->DefineType("bool", std::make_shared<Type>(Type::TypeKind::BOOL), SourceLocation());
         SymbolTable->DefineType("char", std::make_shared<Type>(Type::TypeKind::CHAR), SourceLocation());
         SymbolTable->DefineType("int", std::make_shared<Type>(Type::TypeKind::INT), SourceLocation());
         SymbolTable->DefineType("float", std::make_shared<Type>(Type::TypeKind::FLOAT), SourceLocation());
         SymbolTable->DefineType("double", std::make_shared<Type>(Type::TypeKind::DOUBLE), SourceLocation());
         SymbolTable->DefineType("auto", std::make_shared<Type>(Type::TypeKind::AUTO), SourceLocation());
     }
     
     /**
      * @brief Visit a node in the AST
      * @param node The node to visit
      * @return The type of the node
      */
     std::shared_ptr<Type> VisitNode(const std::unique_ptr<ASTNode>& Node) {
         if (!Node) return nullptr;
         
         switch (Node->Type) {
             case ASTNodeType::PROGRAM:
                 return VisitProgram(static_cast<const ProgramNode&>(*Node));
             case ASTNodeType::VARIABLE_DECL:
                 return VisitVariableDecl(static_cast<const VariableDeclNode&>(*Node));
             case ASTNodeType::FUNCTION_DECL:
                 return VisitFunctionDecl(static_cast<const FunctionDeclNode&>(*Node));
             case ASTNodeType::COMPOUND_STMT:
                 return VisitCompoundStmt(static_cast<const CompoundStmtNode&>(*Node));
             case ASTNodeType::EXPRESSION_STMT:
                 return VisitExpressionStmt(static_cast<const ExpressionStmtNode&>(*Node));
             case ASTNodeType::IF_STMT:
                 return VisitIfStmt(static_cast<const IfStmtNode&>(*Node));
             case ASTNodeType::WHILE_STMT:
                 return VisitWhileStmt(static_cast<const WhileStmtNode&>(*Node));
             case ASTNodeType::FOR_STMT:
                 return VisitForStmt(static_cast<const ForStmtNode&>(*Node));
             case ASTNodeType::RETURN_STMT:
                 return VisitReturnStmt(static_cast<const ReturnStmtNode&>(*Node));
             case ASTNodeType::BINARY_EXPR:
                 return VisitBinaryExpr(static_cast<const BinaryExprNode&>(*Node));
             case ASTNodeType::UNARY_EXPR:
                 return VisitUnaryExpr(static_cast<const UnaryExprNode&>(*Node));
             case ASTNodeType::LITERAL_EXPR:
                 return VisitLiteralExpr(static_cast<const LiteralExprNode&>(*Node));
             case ASTNodeType::VARIABLE_EXPR:
                 return VisitVariableExpr(static_cast<const VariableExprNode&>(*Node));
             case ASTNodeType::ASSIGNMENT_EXPR:
                 return VisitAssignmentExpr(static_cast<const AssignmentExprNode&>(*Node));
             case ASTNodeType::CALL_EXPR:
                 return VisitCallExpr(static_cast<const CallExprNode&>(*Node));
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
     std::shared_ptr<Type> VisitProgram(const ProgramNode& Node) {
         for (const auto& Declaration : Node.Declarations) {
             VisitNode(Declaration);
         }
         return nullptr;
     }
     
     /**
      * @brief Visit a variable declaration node
      * @param node The variable declaration node
      * @return The type of the variable
      */
     std::shared_ptr<Type> VisitVariableDecl(const VariableDeclNode& Node) {
         std::shared_ptr<Type> InitializerType = nullptr;
         
         if (Node.Initializer) {
             InitializerType = VisitNode(Node.Initializer);
             
             // Check if the initializer type is compatible with the variable type
             if (InitializerType && !InitializerType->IsCompatibleWith(*Node.Type)) {
                 ErrorReporter.ReportError(
                     std::format("Cannot initialize variable of type '{}' with value of type '{}'",
                                Node.Type->ToString(), InitializerType->ToString()),
                     Node.Location
                 );
             }
         }
         
         // Add the variable to the symbol table
         if (!SymbolTable->DefineVariable(Node.Name, Node.Type, Node.IsConst, Node.Location)) {
             ErrorReporter.ReportError(
                 std::format("Variable '{}' already defined", Node.Name),
                 Node.Location
             );
         }
         
         return Node.Type;
     }
     
     /**
      * @brief Visit a function declaration node
      * @param node The function declaration node
      * @return The type of the function
      */
     std::shared_ptr<Type> VisitFunctionDecl(const FunctionDeclNode& Node) {
         // Add the function to the symbol table
         if (!SymbolTable->DefineFunction(Node.Name, Node.Type, Node.IsInline, Node.IsVirtual, Node.Location)) {
             ErrorReporter.ReportError(
                 std::format("Function '{}' already defined", Node.Name),
                 Node.Location
             );
         }
         
         // Create a new scope for the function body
         auto FunctionScope = std::make_shared<SymbolTable>(SymbolTable);
         auto OuterScope = SymbolTable;
         SymbolTable = FunctionScope;
         
         // Add parameters to the function's scope
         for (size_t i = 0; i < Node.Parameters.size(); i++) {
             // For simplicity, assume parameters are of the form TYPE NAME
             // In a real compiler, we would properly extract this information
             auto ParamType = Node.Type->ParameterTypes[i];
             std::string ParamName = "param" + std::to_string(i); // Simplified
             
             SymbolTable->DefineVariable(ParamName, ParamType, false, Node.Location);
         }
         
         // Visit the function body
         if (Node.Body) {
             VisitNode(Node.Body);
         }
         
         // Restore the outer scope
         SymbolTable = OuterScope;
         
         return Node.Type;
     }
     
     /**
      * @brief Visit a compound statement node
      * @param node The compound statement node
      * @return nullptr (compound statements don't have a type)
      */
     std::shared_ptr<Type> VisitCompoundStmt(const CompoundStmtNode& Node) {
         // Create a new scope for the compound statement
         auto BlockScope = std::make_shared<SymbolTable>(SymbolTable);
         auto OuterScope = SymbolTable;
         SymbolTable = BlockScope;
         
         for (const auto& Statement : Node.Statements) {
             VisitNode(Statement);
         }
         
         // Restore the outer scope
         SymbolTable = OuterScope;
         
         return nullptr;
     }
     
     /**
      * @brief Visit an expression statement node
      * @param node The expression statement node
      * @return The type of the expression
      */
     std::shared_ptr<Type> VisitExpressionStmt(const ExpressionStmtNode& Node) {
         return VisitNode(Node.Expression);
     }
     
     /**
      * @brief Visit an if statement node
      * @param node The if statement node
      * @return nullptr (if statements don't have a type)
      */
     std::shared_ptr<Type> VisitIfStmt(const IfStmtNode& Node) {
         auto ConditionType = VisitNode(Node.Condition);
         
         // Check if the condition is a boolean expression
         if (ConditionType && ConditionType->Kind != Type::TypeKind::BOOL) {
             ErrorReporter.ReportError(
                 std::format("Condition must be a boolean expression, got '{}'", 
                            ConditionType->ToString()),
                 Node.Location
             );
         }
         
         VisitNode(Node.ThenBranch);
         
         if (Node.ElseBranch) {
             VisitNode(Node.ElseBranch);
         }
         
         return nullptr;
     }
     
     /**
      * @brief Visit a while statement node
      * @param node The while statement node
      * @return nullptr (while statements don't have a type)
      */
     std::shared_ptr<Type> VisitWhileStmt(const WhileStmtNode& Node) {
         auto ConditionType = VisitNode(Node.Condition);
         
         // Check if the condition is a boolean expression
         if (ConditionType && ConditionType->Kind != Type::TypeKind::BOOL) {
             ErrorReporter.ReportError(
                 std::format("Condition must be a boolean expression, got '{}'", 
                            ConditionType->ToString()),
                 Node.Location
             );
         }
         
         VisitNode(Node.Body);
         
         return nullptr;
     }
     
     /**
      * @brief Visit a for statement node
      * @param node The for statement node
      * @return nullptr (for statements don't have a type)
      */
     std::shared_ptr<Type> VisitForStmt(const ForStmtNode& Node) {
         // Create a new scope for the for statement
         auto ForScope = std::make_shared<SymbolTable>(SymbolTable);
         auto OuterScope = SymbolTable;
         SymbolTable = ForScope;
         
         if (Node.Initializer) {
             VisitNode(Node.Initializer);
         }
         
         if (Node.Condition) {
             auto ConditionType = VisitNode(Node.Condition);
             
             // Check if the condition is a boolean expression
             if (ConditionType && ConditionType->Kind != Type::TypeKind::BOOL) {
                 ErrorReporter.ReportError(
                     std::format("Condition must be a boolean expression, got '{}'", 
                                ConditionType->ToString()),
                     Node.Location
                 );
             }
         }
         
         if (Node.Increment) {
             VisitNode(Node.Increment);
         }
         
         VisitNode(Node.Body);
         
         // Restore the outer scope
         SymbolTable = OuterScope;
         
         return nullptr;
     }
     
     /**
      * @brief Visit a return statement node
      * @param node The return statement node
      * @return nullptr (return statements don't have a type)
      */
     std::shared_ptr<Type> VisitReturnStmt(const ReturnStmtNode& Node) {
         // In a full implementation, we would check if the return type matches the function's return type
         if (Node.value) {
             VisitNode(Node.value);
         }
         
         return nullptr;
     }
     
     /**
      * @brief Visit a binary expression node
      * @param node The binary expression node
      * @return The type of the binary expression
      */
     std::shared_ptr<Type> VisitBinaryExpr(const BinaryExprNode& Node) {
         auto LeftType = VisitNode(Node.left);
         auto RightType = VisitNode(Node.right);
         
         if (!LeftType || !RightType) {
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Type checking for binary operators
         switch (Node.Op) {
             case BinaryExprNode::Operator::ADD:
             case BinaryExprNode::Operator::SUBTRACT:
             case BinaryExprNode::Operator::MULTIPLY:
             case BinaryExprNode::Operator::DIVIDE:
             case BinaryExprNode::Operator::MODULO:
                 // Arithmetic operators require numeric operands
                 if (!LeftType->IsNumeric() || !RightType->IsNumeric()) {
                     ErrorReporter.ReportError(
                         std::format("Arithmetic operator requires numeric operands, got '{}' and '{}'",
                                    LeftType->ToString(), RightType->ToString()),
                         Node.Location
                     );
                 }
                 
                 // If either operand is floating-point, the result is floating-point
                 if (LeftType->IsFloatingPoint() || RightType->IsFloatingPoint()) {
                     return std::make_shared<Type>(Type::TypeKind::DOUBLE);
                 } else {
                     return std::make_shared<Type>(Type::TypeKind::INT);
                 }
                 
             case BinaryExprNode::Operator::EQUAL:
             case BinaryExprNode::Operator::NOT_EQUAL:
                 // Equality operators can compare any types, but they must be compatible
                 if (!LeftType->IsCompatibleWith(*RightType)) {
                     ErrorReporter.ReportError(
                         std::format("Cannot compare '{}' and '{}'",
                                    LeftType->ToString(), RightType->ToString()),
                         Node.Location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::LESS:
             case BinaryExprNode::Operator::LESS_EQUAL:
             case BinaryExprNode::Operator::GREATER:
             case BinaryExprNode::Operator::GREATER_EQUAL:
                 // Comparison operators require numeric operands
                 if (!LeftType->IsNumeric() || !RightType->IsNumeric()) {
                     ErrorReporter.ReportError(
                         std::format("Comparison operator requires numeric operands, got '{}' and '{}'",
                                    LeftType->ToString(), RightType->ToString()),
                         Node.Location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::AND:
             case BinaryExprNode::Operator::OR:
                 // Logical operators require boolean operands
                 if (LeftType->Kind != Type::TypeKind::BOOL || RightType->Kind != Type::TypeKind::BOOL) {
                     ErrorReporter.ReportError(
                         std::format("Logical operator requires boolean operands, got '{}' and '{}'",
                                    LeftType->ToString(), RightType->ToString()),
                         Node.Location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case BinaryExprNode::Operator::BITWISE_AND:
             case BinaryExprNode::Operator::BITWISE_OR:
             case BinaryExprNode::Operator::BITWISE_XOR:
             case BinaryExprNode::Operator::LEFT_SHIFT:
             case BinaryExprNode::Operator::RIGHT_SHIFT:
                 // Bitwise operators require integer operands
                 if (!LeftType->IsInteger() || !RightType->IsInteger()) {
                     ErrorReporter.ReportError(
                         std::format("Bitwise operator requires integer operands, got '{}' and '{}'",
                                    LeftType->ToString(), RightType->ToString()),
                         Node.Location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::INT);
                 
             default:
                 ErrorReporter.ReportError("Unknown binary operator", Node.Location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a unary expression node
      * @param node The unary expression node
      * @return The type of the unary expression
      */
     std::shared_ptr<Type> VisitUnaryExpr(const UnaryExprNode& Node) {
         auto OperandType = VisitNode(Node.Operand);
         
         if (!OperandType) {
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Type checking for unary operators
         switch (Node.Op) {
             case UnaryExprNode::Operator::NEGATE:
                 // Negation requires a numeric operand
                 if (!OperandType->IsNumeric()) {
                     ErrorReporter.ReportError(
                         std::format("Unary negation requires a numeric operand, got '{}'",
                                    OperandType->ToString()),
                         Node.Location
                     );
                 }
                 return OperandType;
                 
             case UnaryExprNode::Operator::NOT:
                 // Logical NOT requires a boolean operand
                 if (OperandType->Kind != Type::TypeKind::BOOL) {
                     ErrorReporter.ReportError(
                         std::format("Logical NOT requires a boolean operand, got '{}'",
                                    OperandType->ToString()),
                         Node.Location
                     );
                 }
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
                 
             case UnaryExprNode::Operator::BITWISE_NOT:
                 // Bitwise NOT requires an integer operand
                 if (!OperandType->IsInteger()) {
                     ErrorReporter.ReportError(
                         std::format("Bitwise NOT requires an integer operand, got '{}'",
                                    OperandType->ToString()),
                         Node.Location
                     );
                 }
                 return OperandType;
                 
             case UnaryExprNode::Operator::ADDRESS_OF:
                 // Address-of operator returns a pointer to the operand's type
                 return std::make_shared<PointerType>(OperandType);
                 
             case UnaryExprNode::Operator::DEREFERENCE:
                 // Dereference operator requires a pointer operand
                 if (OperandType->Kind != Type::TypeKind::POINTER) {
                     ErrorReporter.ReportError(
                         std::format("Dereference operator requires a pointer operand, got '{}'",
                                    OperandType->ToString()),
                         Node.Location
                     );
                     return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
                 }
                 
                 return static_cast<PointerType*>(OperandType.get())->BaseType;
                 
             case UnaryExprNode::Operator::PRE_INCREMENT:
             case UnaryExprNode::Operator::PRE_DECREMENT:
             case UnaryExprNode::Operator::POST_INCREMENT:
             case UnaryExprNode::Operator::POST_DECREMENT:
                 // Increment and decrement operators require a numeric operand
                 if (!OperandType->IsNumeric()) {
                     ErrorReporter.ReportError(
                         std::format("Increment/decrement operator requires a numeric operand, got '{}'",
                                    OperandType->ToString()),
                         Node.Location
                     );
                 }
                 return OperandType;
                 
             default:
                 ErrorReporter.ReportError("Unknown unary operator", Node.Location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a literal expression node
      * @param node The literal expression node
      * @return The type of the literal
      */
     std::shared_ptr<Type> VisitLiteralExpr(const LiteralExprNode& Node) {
         switch (Node.LiteralType) {
             case LiteralExprNode::LiteralType::INTEGER:
                 return std::make_shared<Type>(Type::TypeKind::INT);
             case LiteralExprNode::LiteralType::FLOAT:
                 return std::make_shared<Type>(Type::TypeKind::DOUBLE);
             case LiteralExprNode::LiteralType::CHARACTER:
                 return std::make_shared<Type>(Type::TypeKind::CHAR);
             case LiteralExprNode::LiteralType::STRING:
                 return std::make_shared<ArrayType>(
                     std::make_shared<Type>(Type::TypeKind::CHAR),
                     Node.value.length() - 2 + 1 // -2 for quotes, +1 for null terminator
                 );
             case LiteralExprNode::LiteralType::BOOLEAN:
                 return std::make_shared<Type>(Type::TypeKind::BOOL);
             case LiteralExprNode::LiteralType::NULL_LITERAL:
                 return std::make_shared<PointerType>(std::make_shared<Type>(Type::TypeKind::VOID));
             default:
                 ErrorReporter.ReportError("Unknown literal type", Node.Location);
                 return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
     }
     
     /**
      * @brief Visit a variable expression node
      * @param node The variable expression node
      * @return The type of the variable
      */
     std::shared_ptr<Type> VisitVariableExpr(const VariableExprNode& Node) {
         auto Variable = SymbolTable->ResolveVariable(Node.Name);
         
         if (!Variable) {
             ErrorReporter.ReportError(
                 std::format("Undefined variable '{}'", Node.Name),
                 Node.Location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         return Variable->Type;
     }
     
     /**
      * @brief Visit an assignment expression node
      * @param node The assignment expression node
      * @return The type of the assigned value
      */
     std::shared_ptr<Type> VisitAssignmentExpr(const AssignmentExprNode& Node) {
         auto LeftType = VisitNode(Node.left);
         auto RightType = VisitNode(Node.right);
         
         if (!LeftType || !RightType) {
             return LeftType ? LeftType : RightType;
         }
         
         // Check if the left operand is an lvalue
         if (Node.left->Type != ASTNodeType::VARIABLE_EXPR && 
             Node.left->Type != ASTNodeType::MEMBER_ACCESS_EXPR &&
             Node.left->Type != ASTNodeType::ARRAY_ACCESS_EXPR) {
             ErrorReporter.ReportError(
                 "Left-hand side of assignment must be an lvalue",
                 Node.Location
             );
         }
         
         // Check if the right operand is compatible with the left operand
         if (!RightType->IsCompatibleWith(*LeftType)) {
             ErrorReporter.ReportError(
                 std::format("Cannot assign value of type '{}' to variable of type '{}'",
                            RightType->ToString(), LeftType->ToString()),
                 Node.Location
             );
         }
         
         return LeftType;
     }
     
     /**
      * @brief Visit a function call expression node
      * @param node The function call expression node
      * @return The return type of the function
      */
     std::shared_ptr<Type> VisitCallExpr(const CallExprNode& Node) {
         // For now, we'll assume the callee is a variable expression (function name)
         if (Node.Callee->Type != ASTNodeType::VARIABLE_EXPR) {
             ErrorReporter.ReportError(
                 "Function call on non-function expression",
                 Node.Location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         const VariableExprNode& Callee = static_cast<const VariableExprNode&>(*Node.Callee);
         auto function = SymbolTable->ResolveFunction(Callee.Name);
         
         if (!function) {
             ErrorReporter.ReportError(
                 std::format("Undefined function '{}'", Callee.Name),
                 Node.Location
             );
             return std::make_shared<Type>(Type::TypeKind::INT); // Default to int for error recovery
         }
         
         // Check argument count
         if (Node.Arguments.size() != function->Type->ParameterTypes.size()) {
             ErrorReporter.ReportError(
                 std::format("Function '{}' expects {} arguments, got {}",
                            Callee.Name, function->Type->ParameterTypes.size(), Node.Arguments.size()),
                 Node.Location
             );
         } else {
             // Check argument types
             for (size_t i = 0; i < Node.Arguments.size(); i++) {
                 auto ArgType = VisitNode(Node.Arguments[i]);
                 
                 if (ArgType && !ArgType->IsCompatibleWith(*function->Type->ParameterTypes[i])) {
                     ErrorReporter.ReportError(
                         std::format("Function '{}' expects argument {} of type '{}', got '{}'",
                                    Callee.Name, i + 1, function->Type->ParameterTypes[i]->ToString(),
                                    ArgType->ToString()),
                         Node.Location
                     );
                 }
             }
         }
         
         return function->Type->ReturnType;
     }
 };
 
 /**
  * @brief Converts a TokenType to a string for debugging and error reporting
  * @param type The token type to convert
  * @return A string representation of the token type
  */
 std::string TokenTypeToString(TokenType Type) {
     static const std::unordered_map<TokenType, std::string> TokenNames = {
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
     
     auto It = TokenNames.find(Type);
     if (It != TokenNames.end()) {
         return It->second;
     } else {
         return "UNKNOWN_TOKEN";
     }
 }
 
 /**
  * @brief Location information for error reporting
  */
 struct SourceLocation {
     std::string Filename;
     int Line;
     int Column;
     
     SourceLocation(const std::string& File = "", int l = 1, int c = 1)
         : Filename(File), Line(l), Column(c) {}
     
     std::string ToString() const {
         return std::format("{}:{}:{}", Filename, Line, Column);
     }
 };
 
 /**
  * @brief Token class representing lexical units from the source code
  */
 class Token {
 public:
     TokenType Type;
     std::string Lexeme;
     SourceLocation Location;
     
     Token(TokenType t, const std::string& Lex, const SourceLocation& Loc)
         : Type(t), Lexeme(Lex), Location(Loc) {}
     
     std::string ToString() const {
         return std::format("Token({}, '{}', {})", 
                           TokenTypeToString(Type), 
                           Lexeme, 
                           Location.ToString());
     }
 };
 
 /**
  * @brief Error handling class for reporting and tracking compilation errors
  */
 class ErrorReporter {
 private:
     std::vector<std::string> Errors;
     std::vector<std::string> Warnings;
     bool HasError = false;
     
     // Mutex for thread-safe error reporting
     // This is needed because error reporting might be called from different compilation stages
     // running in parallel, or from different threads processing different files
     mutable std::shared_mutex mutex;
 
 public:
     void ReportError(const std::string& Message, const SourceLocation& Location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         std::string ErrorMsg = std::format("Error at {}: {}", Location.ToString(), Message);
         Errors.push_back(ErrorMsg);
         HasError = true;
         std::cerr << ErrorMsg << std::endl;
     }
     
     void ReportWarning(const std::string& Message, const SourceLocation& Location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         std::string WarningMsg = std::format("Warning at {}: {}", Location.ToString(), Message);
         Warnings.push_back(WarningMsg);
         std::cerr << WarningMsg << std::endl;
     }
     
     bool HadError() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return HasError;
     }
     
     const std::vector<std::string>& GetErrors() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return Errors;
     }
     
     const std::vector<std::string>& GetWarnings() const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         return Warnings;
     }
     
     void Reset() {
         std::unique_lock lock(mutex); // Write lock for thread safety
         Errors.clear();
         Warnings.clear();
         HasError = false;
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
     std::string Source;
     std::string Filename;
     size_t Start = 0;
     size_t Current = 0;
     int Line = 1;
     int Column = 1;
     std::vector<Token> Tokens;
     ErrorReporter& ErrorReporter;
     
     // Map of keywords to their corresponding token types
     static const std::unordered_map<std::string, TokenType> Keywords;
 
 public:
     Lexer(const std::string& Source, const std::string& Filename, ErrorReporter& Reporter)
         : Source(Source), Filename(Filename), ErrorReporter(Reporter) {}
     
     /**
      * @brief Scan all tokens from the source code
      * @return Vector of tokens
      */
     std::vector<Token> ScanTokens() {
         while (!IsAtEnd()) {
             // Beginning of the next lexeme
             Start = Current;
             ScanToken();
         }
         
         // Add EOF token
         Tokens.emplace_back(TokenType::END_OF_FILE, "", SourceLocation(Filename, Line, Column));
         return Tokens;
     }
 
 private:
     /**
      * @brief Scan a single token from the source code
      */
     void ScanToken() {
         char c = advance();
         
         switch (c) {
             // Single-character tokens
             case '(': AddToken(TokenType::LEFT_PAREN); break;
             case ')': AddToken(TokenType::RIGHT_PAREN); break;
             case '[': AddToken(TokenType::LEFT_BRACKET); break;
             case ']': AddToken(TokenType::RIGHT_BRACKET); break;
             case '{': AddToken(TokenType::LEFT_BRACE); break;
             case '}': AddToken(TokenType::RIGHT_BRACE); break;
             case ',': AddToken(TokenType::COMMA); break;
             case '.': AddToken(TokenType::DOT); break;
             case ';': AddToken(TokenType::SEMICOLON); break;
             case '?': AddToken(TokenType::QUESTION); break;
             case '~': AddToken(TokenType::TILDE); break;
             
             // Operators that could be part of multi-character operators
             case '+': 
                 if (Match('+')) AddToken(TokenType::PLUS_PLUS);
                 else if (Match('=')) AddToken(TokenType::PLUS_EQUAL);
                 else AddToken(TokenType::PLUS);
                 break;
                 
             case '-': 
                 if (Match('>')) AddToken(TokenType::ARROW);
                 else if (Match('-')) AddToken(TokenType::MINUS_MINUS);
                 else if (Match('=')) AddToken(TokenType::MINUS_EQUAL);
                 else AddToken(TokenType::MINUS);
                 break;
                 
             case '*': 
                 if (Match('=')) AddToken(TokenType::ASTERISK_EQUAL);
                 else AddToken(TokenType::ASTERISK);
                 break;
                 
             case '/': 
                 if (Match('/')) {
                     // Single-line comment
                     while (peek() != '\n' && !IsAtEnd()) advance();
                     // Don't add comment tokens for now
                 } else if (Match('*')) {
                     // Multi-line comment
                     while (!(peek() == '*' && PeekNext() == '/') && !IsAtEnd()) {
                         if (peek() == '\n') {
                             Line++;
                             Column = 1;
                         }
                         advance();
                     }
                     
                     if (IsAtEnd()) {
                         ErrorReporter.ReportError("Unterminated comment", 
                                                  SourceLocation(Filename, Line, Column));
                     } else {
                         // Consume the closing */
                         advance();
                         advance();
                     }
                     // Don't add comment tokens for now
                 } else if (Match('=')) {
                     AddToken(TokenType::SLASH_EQUAL);
                 } else {
                     AddToken(TokenType::SLASH);
                 }
                 break;
                 
             case '%': 
                 if (Match('=')) AddToken(TokenType::PERCENT_EQUAL);
                 else AddToken(TokenType::PERCENT);
                 break;
                 
             case '&': 
                 if (Match('&')) AddToken(TokenType::AMPERSAND_AMPERSAND);
                 else if (Match('=')) AddToken(TokenType::AMPERSAND_EQUAL);
                 else AddToken(TokenType::AMPERSAND);
                 break;
                 
             case '|': 
                 if (Match('|')) AddToken(TokenType::PIPE_PIPE);
                 else if (Match('=')) AddToken(TokenType::PIPE_EQUAL);
                 else AddToken(TokenType::PIPE);
                 break;
                 
             case '^': 
                 if (Match('=')) AddToken(TokenType::CARET_EQUAL);
                 else AddToken(TokenType::CARET);
                 break;
                 
             case '!': 
                 if (Match('=')) AddToken(TokenType::EXCLAMATION_EQUAL);
                 else AddToken(TokenType::EXCLAMATION);
                 break;
                 
             case '=': 
                 if (Match('=')) AddToken(TokenType::EQUAL_EQUAL);
                 else AddToken(TokenType::EQUAL);
                 break;
                 
             case '<': 
                 if (Match('<')) {
                     if (Match('=')) AddToken(TokenType::LESS_LESS_EQUAL);
                     else AddToken(TokenType::LESS_LESS);
                 } else if (Match('=')) {
                     AddToken(TokenType::LESS_EQUAL);
                 } else {
                     AddToken(TokenType::LESS);
                 }
                 break;
                 
             case '>': 
                 if (Match('>')) {
                     if (Match('=')) AddToken(TokenType::GREATER_GREATER_EQUAL);
                     else AddToken(TokenType::GREATER_GREATER);
                 } else if (Match('=')) {
                     AddToken(TokenType::GREATER_EQUAL);
                 } else {
                     AddToken(TokenType::GREATER);
                 }
                 break;
                 
             case ':': 
                 if (Match(':')) AddToken(TokenType::COLON_COLON);
                 else AddToken(TokenType::COLON);
                 break;
                 
             // Whitespace handling
             case ' ':
             case '\r':
             case '\t':
                 // Ignore whitespace
                 break;
                 
             case '\n':
                 Line++;
                 Column = 1;
                 break;
                 
             // Literals
             case '"': StringLiteral(); break;
             case '\'': CharLiteral(); break;
                 
             // Preprocessor directive
             case '#': 
                 // Handle preprocessor directives
                 while (peek() != '\n' && !IsAtEnd()) advance();
                 // Currently just skipping preprocessor directives
                 break;
                 
             default:
                 if (IsDigit(c)) {
                     Number();
                 } else if (IsAlpha(c) || c == '_') {
                     Identifier();
                 } else {
                     ErrorReporter.ReportError(
                         std::format("Unexpected character: {}", c),
                         SourceLocation(Filename, Line, Column - 1)
                     );
                 }
                 break;
         }
     }
     
     /**
      * @brief Process an identifier or keyword
      */
     void Identifier() {
         while (IsAlphaNumeric(peek())) advance();
         
         // See if the identifier is a reserved word
         std::string Text = Source.substr(Start, Current - Start);
         
         auto It = Keywords.find(Text);
         TokenType Type = It != Keywords.end() ? It->second : TokenType::IDENTIFIER;
         
         // Handle boolean literals
         if (Text == "true" || Text == "false") {
             Type = TokenType::BOOL_LITERAL;
         }
         
         AddToken(Type);
     }
     
     /**
      * @brief Process a numeric literal
      */
     void Number() {
         bool IsFloat = false;
         
         // Consume integers
         while (IsDigit(peek())) advance();
         
         // Look for decimal point
         if (peek() == '.' && IsDigit(PeekNext())) {
             IsFloat = true;
             advance(); // Consume the '.'
             
             // Consume fractional part
             while (IsDigit(peek())) advance();
         }
         
         // Look for exponent
         if (peek() == 'e' || peek() == 'E') {
             IsFloat = true;
             advance(); // Consume the 'e' or 'E'
             
             // Optional sign
             if (peek() == '+' || peek() == '-') advance();
             
             // Exponent digits
             if (!IsDigit(peek())) {
                 ErrorReporter.ReportError(
                     "Expected digits after exponent",
                     SourceLocation(Filename, Line, Column)
                 );
             }
             
             while (IsDigit(peek())) advance();
         }
         
         // Look for suffixes
         if (peek() == 'f' || peek() == 'F' || peek() == 'l' || peek() == 'L') {
             IsFloat = true;
             advance();
         } else if ((peek() == 'u' || peek() == 'U') && !IsFloat) {
             advance();
             // Optional size suffix
             if (peek() == 'l' || peek() == 'L') {
                 advance();
                 if (peek() == 'l' || peek() == 'L') advance();
             }
         } else if ((peek() == 'l' || peek() == 'L') && !IsFloat) {
             advance();
             if (peek() == 'l' || peek() == 'L') advance();
             // Optional unsigned suffix
             if (peek() == 'u' || peek() == 'U') advance();
         }
         
         AddToken(IsFloat ? TokenType::FLOAT_LITERAL : TokenType::INTEGER_LITERAL);
     }
     
     /**
      * @brief Process a string literal
      */
     void StringLiteral() {
         while (peek() != '"' && !IsAtEnd()) {
             if (peek() == '\n') {
                 ErrorReporter.ReportError(
                     "Unterminated string literal",
                     SourceLocation(Filename, Line, Column)
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
         
         if (IsAtEnd()) {
             ErrorReporter.ReportError(
                 "Unterminated string literal",
                 SourceLocation(Filename, Line, Column)
             );
             return;
         }
         
         // Consume the closing "
         advance();
         
         // Extract the string content (without the quotes)
         AddToken(TokenType::STRING_LITERAL);
     }
     
     /**
      * @brief Process a character literal
      */
     void CharLiteral() {
         if (IsAtEnd() || peek() == '\'') {
             ErrorReporter.ReportError(
                 "Empty character literal",
                 SourceLocation(Filename, Line, Column)
             );
             if (!IsAtEnd()) advance(); // Consume the closing '
             AddToken(TokenType::CHAR_LITERAL);
             return;
         }
         
         if (peek() == '\\') {
             advance(); // Consume the backslash
             if (IsAtEnd()) {
                 ErrorReporter.ReportError(
                     "Unterminated character literal",
                     SourceLocation(Filename, Line, Column)
                 );
                 return;
             }
             advance(); // Consume the escaped character
         } else {
             advance(); // Consume the character
         }
         
         if (IsAtEnd() || peek() != '\'') {
             ErrorReporter.ReportError(
                 "Unterminated character literal",
                 SourceLocation(Filename, Line, Column)
             );
             return;
         }
         
         advance(); // Consume the closing '
         AddToken(TokenType::CHAR_LITERAL);
     }
     
     /**
      * @brief Check if we're at the end of the source code
      * @return True if at the end, false otherwise
      */
     bool IsAtEnd() const {
         return Current >= Source.length();
     }
     
     /**
      * @brief Consume the current character and return it
      * @return The current character
      */
     char advance() {
         char c = Source[Current++];
         Column++;
         return c;
     }
     
     /**
      * @brief Add a token to the token list
      * @param type The type of token to add
      */
     void AddToken(TokenType Type) {
         std::string Lexeme = Source.substr(Start, Current - Start);
         Tokens.emplace_back(Type, Lexeme, SourceLocation(Filename, Line, Column - Lexeme.length()));
     }
     
     /**
      * @brief Check if the current character matches the expected character
      * @param expected The character to check against
      * @return True if the characters match, false otherwise
      */
     bool Match(char Expected) {
         if (IsAtEnd()) return false;
         if (Source[Current] != Expected) return false;
         
         Current++;
         Column++;
         return true;
     }
     
     /**
      * @brief Look at the current character without consuming it
      * @return The current character, or '\0' if at the end
      */
     char peek() const {
         if (IsAtEnd()) return '\0';
         return Source[Current];
     }
     
     /**
      * @brief Look at the next character without consuming it
      * @return The next character, or '\0' if at the end
      */
     char PeekNext() const {
         if (Current + 1 >= Source.length()) return '\0';
         return Source[Current + 1];
     }
     
     /**
      * @brief Check if a character is a digit
      * @param c The character to check
      * @return True if the character is a digit, false otherwise
      */
     static bool IsDigit(char c) {
         return c >= '0' && c <= '9';
     }
     
     /**
      * @brief Check if a character is alphabetic
      * @param c The character to check
      * @return True if the character is alphabetic, false otherwise
      */
     static bool IsAlpha(char c) {
         return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
     }
     
     /**
      * @brief Check if a character is alphanumeric
      * @param c The character to check
      * @return True if the character is alphanumeric, false otherwise
      */
     static bool IsAlphaNumeric(char c) {
         return IsAlpha(c) || IsDigit(c);
     }
 };
 
 // Static initialization of keywords map
 const std::unordered_map<std::string, TokenType> Lexer::Keywords = {
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
     ASTNodeType Type;
     SourceLocation Location;
     
     ASTNode(ASTNodeType Type, const SourceLocation& Location)
         : Type(Type), Location(Location) {}
     
     virtual ~ASTNode() = default;
     
     virtual std::string ToString() const {
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
     
     TypeKind Kind;
     bool IsConst = false;
     bool IsVolatile = false;
     
     Type(TypeKind Kind) : Kind(Kind) {}
     
     virtual ~Type() = default;
     
     virtual std::string ToString() const {
         std::string Result;
         
         if (IsConst) Result += "const ";
         if (IsVolatile) Result += "volatile ";
         
         switch (Kind) {
             case TypeKind::VOID: Result += "void"; break;
             case TypeKind::BOOL: Result += "bool"; break;
             case TypeKind::CHAR: Result += "char"; break;
             case TypeKind::INT: Result += "int"; break;
             case TypeKind::FLOAT: Result += "float"; break;
             case TypeKind::DOUBLE: Result += "double"; break;
             case TypeKind::AUTO: Result += "auto"; break;
             default: Result += "unknown"; break;
         }
         
         return Result;
     }
     
     virtual bool IsCompatibleWith(const Type& Other) const {
         return Kind == Other.Kind;
     }
     
     virtual size_t GetSize() const {
         switch (Kind) {
             case TypeKind::VOID: return 0;
             case TypeKind::BOOL: return 1;
             case TypeKind::CHAR: return 1;
             case TypeKind::INT: return 4;
             case TypeKind::FLOAT: return 4;
             case TypeKind::DOUBLE: return 8;
             default: return 0;
         }
     }
     
     virtual bool IsNumeric() const {
         return Kind == TypeKind::INT || Kind == TypeKind::FLOAT || 
                Kind == TypeKind::DOUBLE || Kind == TypeKind::CHAR;
     }
     
     virtual bool IsInteger() const {
         return Kind == TypeKind::INT || Kind == TypeKind::CHAR;
     }
     
     virtual bool IsFloatingPoint() const {
         return Kind == TypeKind::FLOAT || Kind == TypeKind::DOUBLE;
     }
 };
 
 /**
  * @brief Pointer type in the type system
  */
 class PointerType : public Type {
 public:
     std::shared_ptr<Type> BaseType;
     
     PointerType(std::shared_ptr<Type> BaseType)
         : Type(TypeKind::POINTER), BaseType(BaseType) {}
     
     std::string ToString() const override {
         return BaseType->ToString() + "*";
     }
     
     bool IsCompatibleWith(const Type& Other) const override {
         if (Other.Kind != TypeKind::POINTER) return false;
         
         const PointerType& OtherPtr = static_cast<const PointerType&>(Other);
         return BaseType->IsCompatibleWith(*OtherPtr.BaseType);
     }
     
     size_t GetSize() const override {
         return 8; // Assume 64-bit pointers
     }
 };
 
 /**
  * @brief Reference type in the type system
  */
 class ReferenceType : public Type {
 public:
     std::shared_ptr<Type> BaseType;
     
     ReferenceType(std::shared_ptr<Type> BaseType)
         : Type(TypeKind::REFERENCE), BaseType(BaseType) {}
     
     std::string ToString() const override {
         return BaseType->ToString() + "&";
     }
     
     bool IsCompatibleWith(const Type& Other) const override {
         if (Other.Kind != TypeKind::REFERENCE) return false;
         
         const ReferenceType& OtherRef = static_cast<const ReferenceType&>(Other);
         return BaseType->IsCompatibleWith(*OtherRef.BaseType);
     }
     
     size_t GetSize() const override {
         return 8; // Assume 64-bit references
     }
 };
 
 /**
  * @brief Array type in the type system
  */
 class ArrayType : public Type {
 public:
     std::shared_ptr<Type> ElementType;
     int size; // -1 for unknown size
     
     ArrayType(std::shared_ptr<Type> ElementType, int size = -1)
         : Type(TypeKind::ARRAY), ElementType(ElementType), size(size) {}
     
     std::string ToString() const override {
         if (size >= 0) {
             return ElementType->ToString() + "[" + std::to_string(size) + "]";
         } else {
             return ElementType->ToString() + "[]";
         }
     }
     
     bool IsCompatibleWith(const Type& Other) const override {
         if (Other.Kind != TypeKind::ARRAY) return false;
         
         const ArrayType& OtherArray = static_cast<const ArrayType&>(Other);
         return ElementType->IsCompatibleWith(*OtherArray.ElementType);
     }
     
     size_t GetSize() const override {
         if (size < 0) return 0;
         return size * ElementType->GetSize();
     }
 };
 
 /**
  * @brief Function type in the type system
  */
 class FunctionType : public Type {
 public:
     std::shared_ptr<Type> ReturnType;
     std::vector<std::shared_ptr<Type>> ParameterTypes;
     
     FunctionType(std::shared_ptr<Type> ReturnType,
                  std::vector<std::shared_ptr<Type>> ParameterTypes)
         : Type(TypeKind::FUNCTION), ReturnType(ReturnType), ParameterTypes(ParameterTypes) {}
     
     std::string ToString() const override {
         std::string Result = ReturnType->ToString() + " (";
         
         for (size_t i = 0; i < ParameterTypes.size(); i++) {
             if (i > 0) Result += ", ";
             Result += ParameterTypes[i]->ToString();
         }
         
         Result += ")";
         return Result;
     }
     
     bool IsCompatibleWith(const Type& Other) const override {
         if (Other.Kind != TypeKind::FUNCTION) return false;
         
         const FunctionType& OtherFunc = static_cast<const FunctionType&>(Other);
         
         if (!ReturnType->IsCompatibleWith(*OtherFunc.ReturnType)) return false;
         if (ParameterTypes.size() != OtherFunc.ParameterTypes.size()) return false;
         
         for (size_t i = 0; i < ParameterTypes.size(); i++) {
             if (!ParameterTypes[i]->IsCompatibleWith(*OtherFunc.ParameterTypes[i])) {
                 return false;
             }
         }
         
         return true;
     }
     
     size_t GetSize() const override {
         return 8; // Function pointers are typically 8 bytes on 64-bit systems
     }
 };
 
 /**
  * @brief Class/Struct type in the type system
  */
 class CompositeType : public Type {
 public:
     std::string Name;
     std::unordered_map<std::string, std::shared_ptr<Type>> Members;
     
     CompositeType(TypeKind Kind, const std::string& Name)
         : Type(Kind), Name(Name) {
         assert(Kind == TypeKind::CLASS || Kind == TypeKind::STRUCT || 
                Kind == TypeKind::UNION);
     }
     
     std::string ToString() const override {
         std::string KindStr;
         switch (Kind) {
             case TypeKind::CLASS: KindStr = "class"; break;
             case TypeKind::STRUCT: KindStr = "struct"; break;
             case TypeKind::UNION: KindStr = "union"; break;
             default: KindStr = "unknown"; break;
         }
         
         return KindStr + " " + Name;
     }
     
     bool IsCompatibleWith(const Type& Other) const override {
         if (Other.Kind != Kind) return false;
         
         const CompositeType& OtherType = static_cast<const CompositeType&>(Other);
         return Name == OtherType.Name;
     }
     
     size_t GetSize() const override {
         // This is a simplified calculation that doesn't account for padding
         size_t TotalSize = 0;
         
         if (Kind == TypeKind::UNION) {
             // For unions, the size is the size of the largest member
             for (const auto& [MemberName, MemberType] : Members) {
                 TotalSize = std::max(TotalSize, MemberType->GetSize());
             }
         } else {
             // For classes and structs, the size is the sum of the members
             for (const auto& [MemberName, MemberType] : Members) {
                 TotalSize += MemberType->GetSize();
             }
         }
         
         return TotalSize;
     }
     
     void AddMember(const std::string& Name, std::shared_ptr<Type> Type) {
         Members[Name] = Type;
     }
     
     std::shared_ptr<Type> GetMember(const std::string& Name) const {
         auto It = Members.find(Name);
         if (It != Members.end()) {
             return It->second;
         }
         return nullptr;
     }
 };
 
 /**
  * @brief A symbol table entry for variable declarations
  */
 struct VariableSymbol {
     std::string Name;
     std::shared_ptr<Type> Type;
     bool IsConst;
     SourceLocation Location;
     
     VariableSymbol(const std::string& Name, std::shared_ptr<Type> Type, 
                    bool IsConst, const SourceLocation& Location)
         : Name(Name), Type(Type), IsConst(IsConst), Location(Location) {}
 };
 
 /**
  * @brief A symbol table entry for function declarations
  */
 struct FunctionSymbol {
     std::string Name;
     std::shared_ptr<FunctionType> Type;
     bool IsInline;
     bool IsVirtual;
     SourceLocation Location;
     
     FunctionSymbol(const std::string& Name, std::shared_ptr<FunctionType> Type,
                   bool IsInline, bool IsVirtual, const SourceLocation& Location)
         : Name(Name), Type(Type), IsInline(IsInline), IsVirtual(IsVirtual), Location(Location) {}
 };
 
 /**
  * @brief A symbol table entry for type declarations
  */
 struct TypeSymbol {
     std::string Name;
     std::shared_ptr<Type> Type;
     SourceLocation Location;
     
     TypeSymbol(const std::string& Name, std::shared_ptr<Type> Type, const SourceLocation& Location)
         : Name(Name), Type(Type), Location(Location) {}
 };
 
 /**
  * @brief A symbol table for managing variable, function, and type declarations
  * 
  * Thread-safe to allow parallel processing of different scopes.
  */
 class SymbolTable {
 private:
     std::unordered_map<std::string, VariableSymbol> Variables;
     std::unordered_map<std::string, FunctionSymbol> Functions;
     std::unordered_map<std::string, TypeSymbol> Types;
     std::shared_ptr<SymbolTable> Parent;
     mutable std::shared_mutex mutex; // For thread safety
 
 public:
     SymbolTable(std::shared_ptr<SymbolTable> Parent = nullptr) : Parent(Parent) {}
     
     /**
      * @brief Define a variable in the current scope
      * @param name The variable name
      * @param type The variable type
      * @param isConst Whether the variable is const
      * @param location Source location for error reporting
      * @return True if successfully defined, false if already defined
      */
     bool DefineVariable(const std::string& Name, std::shared_ptr<Type> Type, 
                         bool IsConst, const SourceLocation& Location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (Variables.find(Name) != Variables.end()) {
             return false; // Already defined in this scope
         }
         
         Variables.Emplace(Name, VariableSymbol(Name, Type, IsConst, Location));
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
     bool DefineFunction(const std::string& Name, std::shared_ptr<FunctionType> Type,
                        bool IsInline, bool IsVirtual, const SourceLocation& Location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (Functions.find(Name) != Functions.end()) {
             return false; // Already defined in this scope
         }
         
         Functions.Emplace(Name, FunctionSymbol(Name, Type, IsInline, IsVirtual, Location));
         return true;
     }
     
     /**
      * @brief Define a type in the current scope
      * @param name The type name
      * @param type The type definition
      * @param location Source location for error reporting
      * @return True if successfully defined, false if already defined
      */
     bool DefineType(const std::string& Name, std::shared_ptr<Type> Type, 
                     const SourceLocation& Location) {
         std::unique_lock lock(mutex); // Write lock for thread safety
         
         if (Types.find(Name) != Types.end()) {
             return false; // Already defined in this scope
         }
         
         Types.Emplace(Name, TypeSymbol(Name, Type, Location));
         return true;
     }
     
     /**
      * @brief Resolve a variable from the current or parent scopes
      * @param name The variable name to resolve
      * @return The variable symbol if found, nullptr otherwise
      */
     std::optional<VariableSymbol> ResolveVariable(const std::string& Name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto It = Variables.find(Name);
         if (It != Variables.end()) {
             return It->second;
         }
         
         if (Parent) {
             return Parent->ResolveVariable(Name);
         }
         
         return std::nullopt;
     }
     
     /**
      * @brief Resolve a function from the current or parent scopes
      * @param name The function name to resolve
      * @return The function symbol if found, nullptr otherwise
      */
     std::optional<FunctionSymbol> ResolveFunction(const std::string& Name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto It = Functions.find(Name);
         if (It != Functions.end()) {
             return It->second;
         }
         
         if (Parent) {
             return Parent->ResolveFunction(Name);
         }
         
         return std::nullopt;
     }
     
     /**
      * @brief Resolve a type from the current or parent scopes
      * @param name The type name to resolve
      * @return The type symbol if found, nullptr otherwise
      */
     std::optional<TypeSymbol> ResolveType(const std::string& Name) const {
         std::shared_lock lock(mutex); // Read lock for thread safety
         
         auto It = Types.find(Name);
         if (It != Types.end()) {
             return It->second;
         }
         
         if (Parent) {
             return Parent->ResolveType(Name);
         }
         
         return std::nullopt;
     }
 };
 
 /**
  * @brief Node representing a program in the AST
  */
 class ProgramNode : public ASTNode {
 public:
     std::vector<std::unique_ptr<ASTNode>> Declarations;
     
     ProgramNode(const SourceLocation& Location)
         : ASTNode(ASTNodeType::PROGRAM, Location) {}
     
     std::string ToString() const override {
         return "Program";
     }
 };
 
 /**
  * @brief Node representing a variable declaration in the AST
  */
 class VariableDeclNode : public ASTNode {
 public:
     std::string Name;
     std::shared_ptr<Type> Type;
     std::unique_ptr<ASTNode> Initializer;
     bool IsConst;
     
     VariableDeclNode(const std::string& Name, std::shared_ptr<Type> Type,
                     std::unique_ptr<ASTNode> Initializer, bool IsConst,
                     const SourceLocation& Location)
         : ASTNode(ASTNodeType::VARIABLE_DECL, Location), Name(Name), Type(Type),
           Initializer(std::move(Initializer)), IsConst(IsConst) {}
     
     std::string ToString() const override {
         std::string Result = "VariableDecl: " + Name + " : " + Type->ToString();
         if (IsConst) Result += " (const)";
         return Result;
     }
 };
 
 /**
  * @brief Node representing a function declaration in the AST
  */
 class FunctionDeclNode : public ASTNode {
 public:
     std::string Name;
     std::shared_ptr<FunctionType> Type;
     std::vector<std::unique_ptr<ASTNode>> Parameters;
     std::unique_ptr<ASTNode> Body;
     bool IsInline;
     bool IsVirtual;
     
     FunctionDeclNode(const std::string& Name, std::shared_ptr<FunctionType> Type,
                     std::vector<std::unique_ptr<ASTNode>> Parameters,
                     std::unique_ptr<ASTNode> Body, bool IsInline, bool IsVirtual,
                     const SourceLocation& Location)
         : ASTNode(ASTNodeType::FUNCTION_DECL, Location), Name(Name), Type(Type),
           Parameters(std::move(Parameters)), Body(std::move(Body)),
           IsInline(IsInline), IsVirtual(IsVirtual) {}
     
     std::string ToString() const override {
         std::string Result = "FunctionDecl: " + Name + " : " + Type->ToString();
         if (IsInline) Result += " (inline)";
         if (IsVirtual) Result += " (virtual)";
         return Result;
     }
 };
 
 /**
  * @brief Node representing a compound statement (block) in the AST
  */
 class CompoundStmtNode : public ASTNode {
 public:
     std::vector<std::unique_ptr<ASTNode>> Statements;
     
     CompoundStmtNode(const SourceLocation& Location)
         : ASTNode(ASTNodeType::COMPOUND_STMT, Location) {}
     
     std::string ToString() const override {
         return "CompoundStmt";
     }
 };
 
 /**
  * @brief Node representing an expression statement in the AST
  */
 class ExpressionStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> Expression;
     
     ExpressionStmtNode(std::unique_ptr<ASTNode> Expression, const SourceLocation& Location)
         : ASTNode(ASTNodeType::EXPRESSION_STMT, Location), Expression(std::move(Expression)) {}
     
     std::string ToString() const override {
         return "ExpressionStmt";
     }
 };
 
 /**
  * @brief Node representing an if statement in the AST
  */
 class IfStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> Condition;
     std::unique_ptr<ASTNode> ThenBranch;
     std::unique_ptr<ASTNode> ElseBranch;
     
     IfStmtNode(std::unique_ptr<ASTNode> Condition, std::unique_ptr<ASTNode> ThenBranch,
               std::unique_ptr<ASTNode> ElseBranch, const SourceLocation& Location)
         : ASTNode(ASTNodeType::IF_STMT, Location), Condition(std::move(Condition)),
           ThenBranch(std::move(ThenBranch)), ElseBranch(std::move(ElseBranch)) {}
     
     std::string ToString() const override {
         return "IfStmt";
     }
 };
 
 /**
  * @brief Node representing a while statement in the AST
  */
 class WhileStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> Condition;
     std::unique_ptr<ASTNode> Body;
     
     WhileStmtNode(std::unique_ptr<ASTNode> Condition, std::unique_ptr<ASTNode> Body,
                  const SourceLocation& Location)
         : ASTNode(ASTNodeType::WHILE_STMT, Location), Condition(std::move(Condition)),
           Body(std::move(Body)) {}
     
     std::string ToString() const override {
         return "WhileStmt";
     }
 };
 
 /**
  * @brief Node representing a for statement in the AST
  */
 class ForStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> Initializer;
     std::unique_ptr<ASTNode> Condition;
     std::unique_ptr<ASTNode> Increment;
     std::unique_ptr<ASTNode> Body;
     
     ForStmtNode(std::unique_ptr<ASTNode> Initializer, std::unique_ptr<ASTNode> Condition,
                std::unique_ptr<ASTNode> Increment, std::unique_ptr<ASTNode> Body,
                const SourceLocation& Location)
         : ASTNode(ASTNodeType::FOR_STMT, Location), Initializer(std::move(Initializer)),
           Condition(std::move(Condition)), Increment(std::move(Increment)),
           Body(std::move(Body)) {}
     
     std::string ToString() const override {
         return "ForStmt";
     }
 };
 
 /**
  * @brief Node representing a return statement in the AST
  */
 class ReturnStmtNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> value;
     
     ReturnStmtNode(std::unique_ptr<ASTNode> value, const SourceLocation& Location)
         : ASTNode(ASTNodeType::RETURN_STMT, Location), value(std::move(value)) {}
     
     std::string ToString() const override {
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
     
     Operator Op;
     std::unique_ptr<ASTNode> left;
     std::unique_ptr<ASTNode> right;
     
     BinaryExprNode(Operator Op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right,
                   const SourceLocation& Location)
         : ASTNode(ASTNodeType::BINARY_EXPR, Location), Op(Op), left(std::move(left)),
           right(std::move(right)) {}
     
     std::string ToString() const override {
         std::string OpStr;
         switch (Op) {
             case Operator::ADD: OpStr = "+"; break;
             case Operator::SUBTRACT: OpStr = "-"; break;
             case Operator::MULTIPLY: OpStr = "*"; break;
             case Operator::DIVIDE: OpStr = "/"; break;
             case Operator::MODULO: OpStr = "%"; break;
             case Operator::EQUAL: OpStr = "=="; break;
             case Operator::NOT_EQUAL: OpStr = "!="; break;
             case Operator::LESS: OpStr = "<"; break;
             case Operator::LESS_EQUAL: OpStr = "<="; break;
             case Operator::GREATER: OpStr = ">"; break;
             case Operator::GREATER_EQUAL: OpStr = ">="; break;
             case Operator::AND: OpStr = "&&"; break;
             case Operator::OR: OpStr = "||"; break;
             case Operator::BITWISE_AND: OpStr = "&"; break;
             case Operator::BITWISE_OR: OpStr = "|"; break;
             case Operator::BITWISE_XOR: OpStr = "^"; break;
             case Operator::LEFT_SHIFT: OpStr = "<<"; break;
             case Operator::RIGHT_SHIFT: OpStr = ">>"; break;
         }
         
         return "BinaryExpr: " + OpStr;
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
     
     Operator Op;
     std::unique_ptr<ASTNode> Operand;
     
     UnaryExprNode(Operator Op, std::unique_ptr<ASTNode> Operand, const SourceLocation& Location)
         : ASTNode(ASTNodeType::UNARY_EXPR, Location), Op(Op), Operand(std::move(Operand)) {}
     
     std::string ToString() const override {
         std::string OpStr;
         switch (Op) {
             case Operator::NEGATE: OpStr = "-"; break;
             case Operator::NOT: OpStr = "!"; break;
             case Operator::BITWISE_NOT: OpStr = "~"; break;
             case Operator::ADDRESS_OF: OpStr = "&"; break;
             case Operator::DEREFERENCE: OpStr = "*"; break;
             case Operator::PRE_INCREMENT: OpStr = "++"; break;
             case Operator::PRE_DECREMENT: OpStr = "--"; break;
             case Operator::POST_INCREMENT: OpStr = "++ (post)"; break;
             case Operator::POST_DECREMENT: OpStr = "-- (post)"; break;
         }
         
         return "UnaryExpr: " + OpStr;
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
     
     LiteralType LiteralType;
     std::string value;
     
     LiteralExprNode(LiteralType LiteralType, const std::string& value, 
                    const SourceLocation& Location)
         : ASTNode(ASTNodeType::LITERAL_EXPR, Location), LiteralType(LiteralType), value(value) {}
     
     std::string ToString() const override {
         std::string TypeStr;
         switch (LiteralType) {
             case LiteralType::INTEGER: TypeStr = "Integer"; break;
             case LiteralType::FLOAT: TypeStr = "Float"; break;
             case LiteralType::CHARACTER: TypeStr = "Character"; break;
             case LiteralType::STRING: TypeStr = "String"; break;
             case LiteralType::BOOLEAN: TypeStr = "Boolean"; break;
             case LiteralType::NULL_LITERAL: TypeStr = "Null"; break;
         }
         
         return "LiteralExpr: " + TypeStr + " " + value;
     }
 };
 
 /**
  * @brief Node representing a variable expression in the AST
  */
 class VariableExprNode : public ASTNode {
 public:
     std::string Name;
     
     VariableExprNode(const std::string& Name, const SourceLocation& Location)
         : ASTNode(ASTNodeType::VARIABLE_EXPR, Location), Name(Name) {}
     
     std::string ToString() const override {
         return "VariableExpr: " + Name;
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
     
     Operator Op;
     std::unique_ptr<ASTNode> left;
     std::unique_ptr<ASTNode> right;
     
     AssignmentExprNode(Operator Op, std::unique_ptr<ASTNode> left, std::unique_ptr<ASTNode> right,
                       const SourceLocation& Location)
         : ASTNode(ASTNodeType::ASSIGNMENT_EXPR, Location), Op(Op), left(std::move(left)),
           right(std::move(right)) {}
     
     std::string ToString() const override {
         std::string OpStr;
         switch (Op) {
             case Operator::ASSIGN: OpStr = "="; break;
             case Operator::ADD_ASSIGN: OpStr = "+="; break;
             case Operator::SUBTRACT_ASSIGN: OpStr = "-="; break;
             case Operator::MULTIPLY_ASSIGN: OpStr = "*="; break;
             case Operator::DIVIDE_ASSIGN: OpStr = "/="; break;
             case Operator::MODULO_ASSIGN: OpStr = "%="; break;
             case Operator::AND_ASSIGN: OpStr = "&="; break;
             case Operator::OR_ASSIGN: OpStr = "|="; break;
             case Operator::XOR_ASSIGN: OpStr = "^="; break;
             case Operator::LEFT_SHIFT_ASSIGN: OpStr = "<<="; break;
             case Operator::RIGHT_SHIFT_ASSIGN: OpStr = ">>="; break;
         }
         
         return "AssignmentExpr: " + OpStr;
     }
 };
 
 /**
  * @brief Node representing a function call expression in the AST
  */
 class CallExprNode : public ASTNode {
 public:
     std::unique_ptr<ASTNode> Callee;
     std::vector<std::unique_ptr<ASTNode>> Arguments;
     
     CallExprNode(std::unique_ptr<ASTNode> Callee, std::vector<std::unique_ptr<ASTNode>> Arguments,
                 const SourceLocation& Location)
         : ASTNode(ASTNodeType::CALL_EXPR, Location), Callee(std::move(Callee)),
           Arguments(std::move(Arguments)) {}
     
     std::string ToString() const override {
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
     const std::vector<Token>& Tokens;
     size_t Current = 0;
     ErrorReporter& ErrorReporter;
     std::shared_ptr<SymbolTable> GlobalSymbols;
 
 public:
     Parser(const std::vector<Token>& Tokens, ErrorReporter& ErrorReporter)
         : Tokens(Tokens), ErrorReporter(ErrorReporter) {
         GlobalSymbols = std::make_shared<SymbolTable>();
     }
     
     /**
      * @brief Parse the tokens into an AST
      * @return The root node of the AST
      */
     std::unique_ptr<ProgramNode> Parse() {
         auto Program = std::make_unique<ProgramNode>(SourceLocation());
         
         while (!IsAtEnd()) {
             try {
                 Program->Declarations.push_back(ParseDeclaration());
             } catch (const std::exception& e) {
                 ErrorReporter.ReportError(e.what(), peek().Location);
                 Synchronize();
             }
         }
         
         return Program;
     }
 
 private:
     /**
      * @brief Parse a declaration
      * @return A node representing the declaration
      */
     std::unique_ptr<ASTNode> ParseDeclaration() {
         if (Match(TokenType::CLASS)) {
             return ParseClassDeclaration();
         } else if (Match(TokenType::STRUCT)) {
             return ParseStructDeclaration();
         } else if (Match(TokenType::ENUM)) {
             return ParseEnumDeclaration();
         } else if (Check(TokenType::INT) || Check(TokenType::CHAR) || Check(TokenType::BOOL) ||
                   Check(TokenType::FLOAT) || Check(TokenType::DOUBLE) || Check(TokenType::VOID) ||
                   Check(TokenType::AUTO) || Check(TokenType::CONST)) {
             return ParseVariableOrFunctionDeclaration();
         } else if (Match(TokenType::NAMESPACE)) {
             return ParseNamespaceDeclaration();
         } else if (Match(TokenType::USING)) {
             return ParseUsingDirective();
         } else if (Match(TokenType::TEMPLATE)) {
             return ParseTemplateDeclaration();
         } else if (Match(TokenType::TYPEDEF)) {
             return ParseTypedefDeclaration();
         }
         
         ErrorReporter.ReportError("Expected declaration", peek().Location);
         throw std::runtime_error("Expected declaration");
     }
     
     /**
      * @brief Parse a class declaration
      * @return A node representing the class declaration
      */
     std::unique_ptr<ASTNode> ParseClassDeclaration() {
         // This is a simplified implementation
         // In a full compiler, this would handle inheritance, access specifiers, etc.
         
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected class name", peek().Location);
             throw std::runtime_error("Expected class name");
         }
         
         std::string ClassName = Previous().Lexeme;
         
         if (!Match(TokenType::LEFT_BRACE)) {
             ErrorReporter.ReportError("Expected '{' after class name", peek().Location);
             throw std::runtime_error("Expected '{' after class name");
         }
         
         // Skip to the end of the class declaration for now
         int BraceCount = 1;
         while (BraceCount > 0 && !IsAtEnd()) {
             if (peek().Type == TokenType::LEFT_BRACE) {
                 BraceCount++;
             } else if (peek().Type == TokenType::RIGHT_BRACE) {
                 BraceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::CLASS_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse a struct declaration
      * @return A node representing the struct declaration
      */
     std::unique_ptr<ASTNode> ParseStructDeclaration() {
         // Similar to class declaration for now
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected struct name", peek().Location);
             throw std::runtime_error("Expected struct name");
         }
         
         std::string StructName = Previous().Lexeme;
         
         if (!Match(TokenType::LEFT_BRACE)) {
             ErrorReporter.ReportError("Expected '{' after struct name", peek().Location);
             throw std::runtime_error("Expected '{' after struct name");
         }
         
         // Skip to the end of the struct declaration for now
         int BraceCount = 1;
         while (BraceCount > 0 && !IsAtEnd()) {
             if (peek().Type == TokenType::LEFT_BRACE) {
                 BraceCount++;
             } else if (peek().Type == TokenType::RIGHT_BRACE) {
                 BraceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::STRUCT_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse an enum declaration
      * @return A node representing the enum declaration
      */
     std::unique_ptr<ASTNode> ParseEnumDeclaration() {
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected enum name", peek().Location);
             throw std::runtime_error("Expected enum name");
         }
         
         std::string EnumName = Previous().Lexeme;
         
         if (!Match(TokenType::LEFT_BRACE)) {
             ErrorReporter.ReportError("Expected '{' after enum name", peek().Location);
             throw std::runtime_error("Expected '{' after enum name");
         }
         
         // Skip to the end of the enum declaration for now
         int BraceCount = 1;
         while (BraceCount > 0 && !IsAtEnd()) {
             if (peek().Type == TokenType::LEFT_BRACE) {
                 BraceCount++;
             } else if (peek().Type == TokenType::RIGHT_BRACE) {
                 BraceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::ENUM_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse a variable or function declaration
      * @return A node representing the variable or function declaration
      */
     std::unique_ptr<ASTNode> ParseVariableOrFunctionDeclaration() {
         // Check for const qualifier
         bool IsConst = Match(TokenType::CONST);
         
         // Parse type specifier
         auto TypeSpecifier = ParseTypeSpecifier();
         
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected identifier", peek().Location);
             throw std::runtime_error("Expected identifier");
         }
         
         std::string Name = Previous().Lexeme;
         
         // If next token is '(', it's a function declaration
         if (Match(TokenType::LEFT_PAREN)) {
             // Function parameters
             std::vector<std::unique_ptr<ASTNode>> Parameters;
             
             if (!Check(TokenType::RIGHT_PAREN)) {
                 do {
                     // Parse parameter
                     bool ParamConst = Match(TokenType::CONST);
                     auto ParamType = ParseTypeSpecifier();
                     
                     if (!Match(TokenType::IDENTIFIER)) {
                         ErrorReporter.ReportError("Expected parameter name", peek().Location);
                         throw std::runtime_error("Expected parameter name");
                     }
                     
                     std::string ParamName = Previous().Lexeme;
                     
                     // Create parameter node (simplified for now)
                     Parameters.push_back(std::make_unique<ASTNode>(ASTNodeType::PARAMETER, Previous().Location));
                     
                 } while (Match(TokenType::COMMA));
             }
             
             Consume(TokenType::RIGHT_PAREN, "Expected ')' after function parameters");
             
             // Function body
             std::unique_ptr<ASTNode> Body = nullptr;
             if (Match(TokenType::SEMICOLON)) {
                 // Function declaration without body
             } else {
                 Body = ParseCompoundStatement();
             }
             
             // Create function node (simplified for now)
             return std::make_unique<ASTNode>(ASTNodeType::FUNCTION_DECL, Previous().Location);
             
         } else {
             // It's a variable declaration
             std::unique_ptr<ASTNode> Initializer = nullptr;
             
             if (Match(TokenType::EQUAL)) {
                 Initializer = ParseExpression();
             }
             
             Consume(TokenType::SEMICOLON, "Expected ';' after variable declaration");
             
             // Create variable node (simplified for now)
             return std::make_unique<ASTNode>(ASTNodeType::VARIABLE_DECL, Previous().Location);
         }
     }
     
     /**
      * @brief Parse a namespace declaration
      * @return A node representing the namespace declaration
      */
     std::unique_ptr<ASTNode> ParseNamespaceDeclaration() {
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected namespace name", peek().Location);
             throw std::runtime_error("Expected namespace name");
         }
         
         std::string NamespaceName = Previous().Lexeme;
         
         if (!Match(TokenType::LEFT_BRACE)) {
             ErrorReporter.ReportError("Expected '{' after namespace name", peek().Location);
             throw std::runtime_error("Expected '{' after namespace name");
         }
         
         // Skip to the end of the namespace declaration for now
         int BraceCount = 1;
         while (BraceCount > 0 && !IsAtEnd()) {
             if (peek().Type == TokenType::LEFT_BRACE) {
                 BraceCount++;
             } else if (peek().Type == TokenType::RIGHT_BRACE) {
                 BraceCount--;
             }
             advance();
         }
         
         return std::make_unique<ASTNode>(ASTNodeType::NAMESPACE_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse a using directive
      * @return A node representing the using directive
      */
     std::unique_ptr<ASTNode> ParseUsingDirective() {
         if (!Match(TokenType::NAMESPACE)) {
             ErrorReporter.ReportError("Expected 'namespace' in using directive", peek().Location);
             throw std::runtime_error("Expected 'namespace' in using directive");
         }
         
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected namespace name", peek().Location);
             throw std::runtime_error("Expected namespace name");
         }
         
         std::string NamespaceName = Previous().Lexeme;
         
         Consume(TokenType::SEMICOLON, "Expected ';' after using directive");
         
         return std::make_unique<ASTNode>(ASTNodeType::USING_DIRECTIVE, Previous().Location);
     }
     
     /**
      * @brief Parse a template declaration
      * @return A node representing the template declaration
      */
     std::unique_ptr<ASTNode> ParseTemplateDeclaration() {
         Consume(TokenType::LESS, "Expected '<' after 'template'");
         
         // Parse template parameters
         do {
             if (Match(TokenType::CLASS) || Match(TokenType::TYPENAME)) {
                 if (!Match(TokenType::IDENTIFIER)) {
                     ErrorReporter.ReportError("Expected template parameter name", peek().Location);
                     throw std::runtime_error("Expected template parameter name");
                 }
             } else {
                 // Parse non-type template parameter
                 auto ParamType = ParseTypeSpecifier();
                 
                 if (!Match(TokenType::IDENTIFIER)) {
                     ErrorReporter.ReportError("Expected template parameter name", peek().Location);
                     throw std::runtime_error("Expected template parameter name");
                 }
                 
                 if (Match(TokenType::EQUAL)) {
                     // Parse default value
                     ParseExpression();
                 }
             }
         } while (Match(TokenType::COMMA));
         
         Consume(TokenType::GREATER, "Expected '>' after template parameters");
         
         // Parse the templated declaration
         auto Declaration = ParseDeclaration();
         
         return std::make_unique<ASTNode>(ASTNodeType::TEMPLATE_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse a typedef declaration
      * @return A node representing the typedef declaration
      */
     std::unique_ptr<ASTNode> ParseTypedefDeclaration() {
         auto Type = ParseTypeSpecifier();
         
         if (!Match(TokenType::IDENTIFIER)) {
             ErrorReporter.ReportError("Expected type alias name", peek().Location);
             throw std::runtime_error("Expected type alias name");
         }
         
         std::string AliasName = Previous().Lexeme;
         
         Consume(TokenType::SEMICOLON, "Expected ';' after typedef declaration");
         
         return std::make_unique<ASTNode>(ASTNodeType::TYPEDEF_DECL, Previous().Location);
     }
     
     /**
      * @brief Parse a type specifier
      * @return A shared pointer to the type
      */
     std::shared_ptr<Type> ParseTypeSpecifier() {
         TokenType TypeToken = peek().Type;
         advance();
         
         Type::TypeKind Kind;
         switch (TypeToken) {
             case TokenType::VOID: Kind = Type::TypeKind::VOID; break;
             case TokenType::BOOL: Kind = Type::TypeKind::BOOL; break;
             case TokenType::CHAR: Kind = Type::TypeKind::CHAR; break;
             case TokenType::INT: Kind = Type::TypeKind::INT; break;
             case TokenType::FLOAT: Kind = Type::TypeKind::FLOAT; break;
             case TokenType::DOUBLE: Kind = Type::TypeKind::DOUBLE; break;
             case TokenType::AUTO: Kind = Type::TypeKind::AUTO; break;
             default:
                 ErrorReporter.ReportError("Expected type specifier", Previous().Location);
                 throw std::runtime_error("Expected type specifier");
         }
         
         auto BaseType = std::make_shared<Type>(Kind);
         
         // Handle pointers, references, and arrays
         while (Match(TokenType::ASTERISK) || Match(TokenType::AMPERSAND) || 
               Check(TokenType::LEFT_BRACKET)) {
             if (Previous().Type == TokenType::ASTERISK) {
                 BaseType = std::make_shared<PointerType>(BaseType);
             } else if (Previous().Type == TokenType::AMPERSAND) {
                 BaseType = std::make_shared<ReferenceType>(BaseType);
             } else if (peek().Type == TokenType::LEFT_BRACKET) {
                 advance();
                 int size = -1;
                 
                 if (Match(TokenType::INTEGER_LITERAL)) {
                     size = std::stoi(Previous().Lexeme);
                 }
                 
                 Consume(TokenType::RIGHT_BRACKET, "Expected ']' after array size");
                 BaseType = std::make_shared<ArrayType>(BaseType, size);
             }
         }
         
         return BaseType;
     }
     
     /**
      * @brief Parse a compound statement (block)
      * @return A node representing the compound statement
      */
     std::unique_ptr<ASTNode> ParseCompoundStatement() {
         Consume(TokenType::LEFT_BRACE, "Expected '{' at the beginning of a block");
         
         auto CompoundStmt = std::make_unique<CompoundStmtNode>(Previous().Location);
         
         while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
             CompoundStmt->Statements.push_back(ParseStatement());
         }
         
         Consume(TokenType::RIGHT_BRACE, "Expected '}' at the end of a block");
         
         return CompoundStmt;
     }
     
     /**
      * @brief Parse a statement
      * @return A node representing the statement
      */
     std::unique_ptr<ASTNode> ParseStatement() {
         if (Match(TokenType::IF)) {
             return ParseIfStatement();
         } else if (Match(TokenType::WHILE)) {
             return ParseWhileStatement();
         } else if (Match(TokenType::FOR)) {
             return ParseForStatement();
         } else if (Match(TokenType::RETURN)) {
             return ParseReturnStatement();
         } else if (Match(TokenType::BREAK)) {
             auto Node = std::make_unique<ASTNode>(ASTNodeType::BREAK_STMT, Previous().Location);
             Consume(TokenType::SEMICOLON, "Expected ';' after break statement");
             return Node;
         } else if (Match(TokenType::CONTINUE)) {
             auto Node = std::make_unique<ASTNode>(ASTNodeType::CONTINUE_STMT, Previous().Location);
             Consume(TokenType::SEMICOLON, "Expected ';' after continue statement");
             return Node;
         } else if (Match(TokenType::LEFT_BRACE)) {
             // Parse a block
             Current--; // Backtrack to the '{'
             return ParseCompoundStatement();
         } else {
             // Expression statement or variable declaration
             if (Check(TokenType::INT) || Check(TokenType::CHAR) || Check(TokenType::BOOL) ||
                 Check(TokenType::FLOAT) || Check(TokenType::DOUBLE) || Check(TokenType::VOID) ||
                 Check(TokenType::AUTO) || Check(TokenType::CONST)) {
                 return ParseVariableOrFunctionDeclaration();
             } else {
                 return ParseExpressionStatement();
             }
         }
     }
     
     /**
      * @brief Parse an if statement
      * @return A node representing the if statement
      */
     std::unique_ptr<ASTNode> ParseIfStatement() {
         Consume(TokenType::LEFT_PAREN, "Expected '(' after 'if'");
         auto Condition = ParseExpression();
         Consume(TokenType::RIGHT_PAREN, "Expected ')' after if condition");
         
         auto ThenBranch = ParseStatement();
         std::unique_ptr<ASTNode> ElseBranch = nullptr;
         
         if (Match(TokenType::ELSE)) {
             ElseBranch = ParseStatement();
         }
         
         return std::make_unique<IfStmtNode>(std::move(Condition), std::move(ThenBranch),
                                           std::move(ElseBranch), Previous().Location);
     }
     
     /**
      * @brief Parse a while statement
      * @return A node representing the while statement
      */
     std::unique_ptr<ASTNode> ParseWhileStatement() {
         Consume(TokenType::LEFT_PAREN, "Expected '(' after 'while'");
         auto Condition = ParseExpression();
         Consume(TokenType::RIGHT_PAREN, "Expected ')' after while condition");
         
         auto Body = ParseStatement();
         
         return std::make_unique<WhileStmtNode>(std::move(Condition), std::move(Body),
                                              Previous().Location);
     }
     
     /**
      * @brief Parse a for statement
      * @return A node representing the for statement
      */
     std::unique_ptr<ASTNode> ParseForStatement() {
         Consume(TokenType::LEFT_PAREN, "Expected '(' after 'for'");
         
         std::unique_ptr<ASTNode> Initializer = nullptr;
         if (!Check(TokenType::SEMICOLON)) {
             if (Check(TokenType::INT) || Check(TokenType::CHAR) || Check(TokenType::BOOL) ||
                 Check(TokenType::FLOAT) || Check(TokenType::DOUBLE) || Check(TokenType::VOID) ||
                 Check(TokenType::AUTO) || Check(TokenType::CONST)) {
                 Initializer = ParseVariableOrFunctionDeclaration();
             } else {
                 Initializer = ParseExpressionStatement();
             }
         } else {
             Consume(TokenType::SEMICOLON, "Expected ';'");
         }
         
         std::unique_ptr<ASTNode> Condition = nullptr;
         if (!Check(TokenType::SEMICOLON)) {
             Condition = ParseExpression();
         }
         Consume(TokenType::SEMICOLON, "Expected ';' after for condition");
         
         std::unique_ptr<ASTNode> Increment = nullptr;
         if (!Check(TokenType::RIGHT_PAREN)) {
             Increment = ParseExpression();
         }
         Consume(TokenType::RIGHT_PAREN, "Expected ')' after for clauses");
         
         auto Body = ParseStatement();
         
         return std::make_unique<ForStmtNode>(std::move(Initializer), std::move(Condition),
                                            std::move(Increment), std::move(Body),
                                            Previous().Location);
     }
     
     /**
      * @brief Parse a return statement
      * @return A node representing the return statement
      */
     std::unique_ptr<ASTNode> ParseReturnStatement() {
         auto Location = Previous().Location;
         
         std::unique_ptr<ASTNode> value = nullptr;
         if (!Check(TokenType::SEMICOLON)) {
             value = ParseExpression();
         }
         
         Consume(TokenType::SEMICOLON, "Expected ';' after return value");
         
         return std::make_unique<ReturnStmtNode>(std::move(value), Location);
     }
     
     /**
      * @brief Parse an expression statement
      * @return A node representing the expression statement
      */
     std::unique_ptr<ASTNode> ParseExpressionStatement() {
         auto Expr = ParseExpression();
         Consume(TokenType::SEMICOLON, "Expected ';' after expression");
         
         return std::make_unique<ExpressionStmtNode>(std::move(Expr), Previous().Location);
     }
     
     /**
      * @brief Parse an expression
      * @return A node representing the expression
      */
     std::unique_ptr<ASTNode> ParseExpression() {
         return ParseAssignment();
     }
     
     /**
      * @brief Parse an assignment expression
      * @return A node representing the assignment expression
      */
     std::unique_ptr<ASTNode> ParseAssignment() {
         auto Expr = ParseConditional();
         
         if (Match(TokenType::EQUAL) || Match(TokenType::PLUS_EQUAL) || 
             Match(TokenType::MINUS_EQUAL) || Match(TokenType::ASTERISK_EQUAL) ||
             Match(TokenType::SLASH_EQUAL) || Match(TokenType::PERCENT_EQUAL) ||
             Match(TokenType::AMPERSAND_EQUAL) || Match(TokenType::PIPE_EQUAL) ||
             Match(TokenType::CARET_EQUAL) || Match(TokenType::LESS_LESS_EQUAL) ||
             Match(TokenType::GREATER_GREATER_EQUAL)) {
             
             TokenType OperatorType = Previous().Type;
             auto value = ParseAssignment();
             
             AssignmentExprNode::Operator Op;
             switch (OperatorType) {
                 case TokenType::EQUAL: Op = AssignmentExprNode::Operator::ASSIGN; break;
                 case TokenType::PLUS_EQUAL: Op = AssignmentExprNode::Operator::ADD_ASSIGN; break;
                 case TokenType::MINUS_EQUAL: Op = AssignmentExprNode::Operator::SUBTRACT_ASSIGN; break;
                 case TokenType::ASTERISK_EQUAL: Op = AssignmentExprNode::Operator::MULTIPLY_ASSIGN; break;
                 case TokenType::SLASH_EQUAL: Op = AssignmentExprNode::Operator::DIVIDE_ASSIGN; break;
                 case TokenType::PERCENT_EQUAL: Op = AssignmentExprNode::Operator::MODULO_ASSIGN; break;
                 case TokenType::AMPERSAND_EQUAL: Op = AssignmentExprNode::Operator::AND_ASSIGN; break;
                 case TokenType::PIPE_EQUAL: Op = AssignmentExprNode::Operator::OR_ASSIGN; break;
                 case TokenType::CARET_EQUAL: Op = AssignmentExprNode::Operator::XOR_ASSIGN; break;
                 case TokenType::LESS_LESS_EQUAL: Op = AssignmentExprNode::Operator::LEFT_SHIFT_ASSIGN; break;
                 case TokenType::GREATER_GREATER_EQUAL: Op = AssignmentExprNode::Operator::RIGHT_SHIFT_ASSIGN; break;
                 default:
                     ErrorReporter.ReportError("Invalid assignment operator", Previous().Location);
                     throw std::runtime_error("Invalid assignment operator");
             }
             
             return std::make_unique<AssignmentExprNode>(Op, std::move(Expr), std::move(value),
                                                      Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a conditional expression (ternary operator)
      * @return A node representing the conditional expression
      */
     std::unique_ptr<ASTNode> ParseConditional() {
         auto Expr = ParseLogicalOr();
         
         if (Match(TokenType::QUESTION)) {
             auto ThenBranch = ParseExpression();
             Consume(TokenType::COLON, "Expected ':' in conditional expression");
             auto ElseBranch = ParseConditional();
             
             // Simplified as a binary expression for now
             return std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::OR, 
                                                   std::move(Expr), std::move(ThenBranch),
                                                   Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a logical OR expression
      * @return A node representing the logical OR expression
      */
     std::unique_ptr<ASTNode> ParseLogicalOr() {
         auto Expr = ParseLogicalAnd();
         
         while (Match(TokenType::PIPE_PIPE)) {
             auto right = ParseLogicalAnd();
             Expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::OR, 
                                                  std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a logical AND expression
      * @return A node representing the logical AND expression
      */
     std::unique_ptr<ASTNode> ParseLogicalAnd() {
         auto Expr = ParseBitwiseOr();
         
         while (Match(TokenType::AMPERSAND_AMPERSAND)) {
             auto right = ParseBitwiseOr();
             Expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::AND, 
                                                  std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a bitwise OR expression
      * @return A node representing the bitwise OR expression
      */
     std::unique_ptr<ASTNode> ParseBitwiseOr() {
         auto Expr = ParseBitwiseXor();
         
         while (Match(TokenType::PIPE)) {
             auto right = ParseBitwiseXor();
             Expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_OR, 
                                                  std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a bitwise XOR expression
      * @return A node representing the bitwise XOR expression
      */
     std::unique_ptr<ASTNode> ParseBitwiseXor() {
         auto Expr = ParseBitwiseAnd();
         
         while (Match(TokenType::CARET)) {
             auto right = ParseBitwiseAnd();
             Expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_XOR, 
                                                  std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a bitwise AND expression
      * @return A node representing the bitwise AND expression
      */
     std::unique_ptr<ASTNode> ParseBitwiseAnd() {
         auto Expr = ParseEquality();
         
         while (Match(TokenType::AMPERSAND)) {
             auto right = ParseEquality();
             Expr = std::make_unique<BinaryExprNode>(BinaryExprNode::Operator::BITWISE_AND, 
                                                  std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse an equality expression
      * @return A node representing the equality expression
      */
     std::unique_ptr<ASTNode> ParseEquality() {
         auto Expr = ParseComparison();
         
         while (Match(TokenType::EQUAL_EQUAL) || Match(TokenType::EXCLAMATION_EQUAL)) {
             TokenType OperatorType = Previous().Type;
             auto right = ParseComparison();
             
             BinaryExprNode::Operator Op;
             if (OperatorType == TokenType::EQUAL_EQUAL) {
                 Op = BinaryExprNode::Operator::EQUAL;
             } else {
                 Op = BinaryExprNode::Operator::NOT_EQUAL;
             }
             
             Expr = std::make_unique<BinaryExprNode>(Op, std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a comparison expression
      * @return A node representing the comparison expression
      */
     std::unique_ptr<ASTNode> ParseComparison() {
         auto Expr = ParseShift();
         
         while (Match(TokenType::LESS) || Match(TokenType::LESS_EQUAL) ||
               Match(TokenType::GREATER) || Match(TokenType::GREATER_EQUAL)) {
             
             TokenType OperatorType = Previous().Type;
             auto right = ParseShift();
             
             BinaryExprNode::Operator Op;
             switch (OperatorType) {
                 case TokenType::LESS: Op = BinaryExprNode::Operator::LESS; break;
                 case TokenType::LESS_EQUAL: Op = BinaryExprNode::Operator::LESS_EQUAL; break;
                 case TokenType::GREATER: Op = BinaryExprNode::Operator::GREATER; break;
                 case TokenType::GREATER_EQUAL: Op = BinaryExprNode::Operator::GREATER_EQUAL; break;
                 default:
                     ErrorReporter.ReportError("Invalid comparison operator", Previous().Location);
                     throw std::runtime_error("Invalid comparison operator");
             }
             
             Expr = std::make_unique<BinaryExprNode>(Op, std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a shift expression
      * @return A node representing the shift expression
      */
     std::unique_ptr<ASTNode> ParseShift() {
         auto Expr = ParseAdditive();
         
         while (Match(TokenType::LESS_LESS) || Match(TokenType::GREATER_GREATER)) {
             TokenType OperatorType = Previous().Type;
             auto right = ParseAdditive();
             
             BinaryExprNode::Operator Op;
             if (OperatorType == TokenType::LESS_LESS) {
                 Op = BinaryExprNode::Operator::LEFT_SHIFT;
             } else {
                 Op = BinaryExprNode::Operator::RIGHT_SHIFT;
             }
             
             Expr = std::make_unique<BinaryExprNode>(Op, std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse an additive expression
      * @return A node representing the additive expression
      */
     std::unique_ptr<ASTNode> ParseAdditive() {
         auto Expr = ParseMultiplicative();
         
         while (Match(TokenType::PLUS) || Match(TokenType::MINUS)) {
             TokenType OperatorType = Previous().Type;
             auto right = ParseMultiplicative();
             
             BinaryExprNode::Operator Op;
             if (OperatorType == TokenType::PLUS) {
                 Op = BinaryExprNode::Operator::ADD;
             } else {
                 Op = BinaryExprNode::Operator::SUBTRACT;
             }
             
             Expr = std::make_unique<BinaryExprNode>(Op, std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a multiplicative expression
      * @return A node representing the multiplicative expression
      */
     std::unique_ptr<ASTNode> ParseMultiplicative() {
         auto Expr = ParseUnary();
         
         while (Match(TokenType::ASTERISK) || Match(TokenType::SLASH) || Match(TokenType::PERCENT)) {
             TokenType OperatorType = Previous().Type;
             auto right = ParseUnary();
             
             BinaryExprNode::Operator Op;
             switch (OperatorType) {
                 case TokenType::ASTERISK: Op = BinaryExprNode::Operator::MULTIPLY; break;
                 case TokenType::SLASH: Op = BinaryExprNode::Operator::DIVIDE; break;
                 case TokenType::PERCENT: Op = BinaryExprNode::Operator::MODULO; break;
                 default:
                     ErrorReporter.ReportError("Invalid multiplicative operator", Previous().Location);
                     throw std::runtime_error("Invalid multiplicative operator");
             }
             
             Expr = std::make_unique<BinaryExprNode>(Op, std::move(Expr), std::move(right),
                                                  Previous().Location);
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a unary expression
      * @return A node representing the unary expression
      */
     std::unique_ptr<ASTNode> ParseUnary() {
         if (Match(TokenType::EXCLAMATION) || Match(TokenType::MINUS) || Match(TokenType::TILDE) ||
             Match(TokenType::AMPERSAND) || Match(TokenType::ASTERISK) ||
             Match(TokenType::PLUS_PLUS) || Match(TokenType::MINUS_MINUS)) {
             
             TokenType OperatorType = Previous().Type;
             auto right = ParseUnary();
             
             UnaryExprNode::Operator Op;
             switch (OperatorType) {
                 case TokenType::EXCLAMATION: Op = UnaryExprNode::Operator::NOT; break;
                 case TokenType::MINUS: Op = UnaryExprNode::Operator::NEGATE; break;
                 case TokenType::TILDE: Op = UnaryExprNode::Operator::BITWISE_NOT; break;
                 case TokenType::AMPERSAND: Op = UnaryExprNode::Operator::ADDRESS_OF; break;
                 case TokenType::ASTERISK: Op = UnaryExprNode::Operator::DEREFERENCE; break;
                 case TokenType::PLUS_PLUS: Op = UnaryExprNode::Operator::PRE_INCREMENT; break;
                 case TokenType::MINUS_MINUS: Op = UnaryExprNode::Operator::PRE_DECREMENT; break;
                 default:
                     ErrorReporter.ReportError("Invalid unary operator", Previous().Location);
                     throw std::runtime_error("Invalid unary operator");
             }
             
             return std::make_unique<UnaryExprNode>(Op, std::move(right), Previous().Location);
         }
         
         return ParsePostfix();
     }
     
     /**
      * @brief Parse a postfix expression
      * @return A node representing the postfix expression
      */
     std::unique_ptr<ASTNode> ParsePostfix() {
         auto Expr = ParsePrimary();
         
         while (Match(TokenType::PLUS_PLUS) || Match(TokenType::MINUS_MINUS) ||
               Match(TokenType::LEFT_PAREN) || Match(TokenType::LEFT_BRACKET) ||
               Match(TokenType::DOT) || Match(TokenType::ARROW)) {
             
             if (Previous().Type == TokenType::PLUS_PLUS) {
                 Expr = std::make_unique<UnaryExprNode>(
                     UnaryExprNode::Operator::POST_INCREMENT,
                     std::move(Expr), Previous().Location);
             } else if (Previous().Type == TokenType::MINUS_MINUS) {
                 Expr = std::make_unique<UnaryExprNode>(
                     UnaryExprNode::Operator::POST_DECREMENT,
                     std::move(Expr), Previous().Location);
             } else if (Previous().Type == TokenType::LEFT_PAREN) {
                 // Function call
                 std::vector<std::unique_ptr<ASTNode>> Arguments;
                 
                 if (!Check(TokenType::RIGHT_PAREN)) {
                     do {
                         Arguments.push_back(ParseExpression());
                     } while (Match(TokenType::COMMA));
                 }
                 
                 Consume(TokenType::RIGHT_PAREN, "Expected ')' after function call arguments");
                 
                 Expr = std::make_unique<CallExprNode>(
                     std::move(Expr), std::move(Arguments), Previous().Location);
             } else if (Previous().Type == TokenType::LEFT_BRACKET) {
                 // Array access
                 auto Index = ParseExpression();
                 Consume(TokenType::RIGHT_BRACKET, "Expected ']' after array index");
                 
                 // Simplified as a call expression for now
                 std::vector<std::unique_ptr<ASTNode>> Arguments;
                 Arguments.push_back(std::move(Index));
                 
                 Expr = std::make_unique<CallExprNode>(
                     std::move(Expr), std::move(Arguments), Previous().Location);
             } else if (Previous().Type == TokenType::DOT || Previous().Type == TokenType::ARROW) {
                 // Member access
                 if (!Match(TokenType::IDENTIFIER)) {
                     ErrorReporter.ReportError("Expected member name after '.' or '->'", peek().Location);
                     throw std::runtime_error("Expected member name after '.' or '->'");
                 }
                 
                 std::string MemberName = Previous().Lexeme;
                 
                 // Simplified as a variable expression for now
                 Expr = std::make_unique<VariableExprNode>(MemberName, Previous().Location);
             }
         }
         
         return Expr;
     }
     
     /**
      * @brief Parse a primary expression
      * @return A node representing the primary expression
      */
     std::unique_ptr<ASTNode> ParsePrimary() {
         if (Match(TokenType::INTEGER_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::INTEGER,
                 Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::FLOAT_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::FLOAT,
                 Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::CHAR_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::CHARACTER,
                 Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::STRING_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::STRING,
                 Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::BOOL_LITERAL)) {
             return std::make_unique<LiteralExprNode>(
                 LiteralExprNode::LiteralType::BOOLEAN,
                 Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::IDENTIFIER)) {
             return std::make_unique<VariableExprNode>(Previous().Lexeme, Previous().Location);
         } else if (Match(TokenType::LEFT_PAREN)) {
             auto Expr = ParseExpression();
             Consume(TokenType::RIGHT_PAREN, "Expected ')' after expression");
             return Expr;
         }
         
         ErrorReporter.ReportError("Expected expression", peek().Location);
         throw std::runtime_error("Expected expression");
     }
     
     /**
      * @brief Utility method to check if the current token matches the expected type
      * @param type The token type to check against
      * @return True if the current token matches the expected type, false otherwise
      */
     bool Match(TokenType Type) {
         if (Check(Type)) {
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
     bool Check(TokenType Type) const {
         if (IsAtEnd()) return false;
         return peek().Type == Type;
     }
     
     /**
      * @brief Utility method to consume the current token and return it
      * @return The consumed token
      */
     Token advance() {
         if (!IsAtEnd()) Current++;
         return Previous();
     }
     
     /**
      * @brief Utility method to check if we're at the end of the token stream
      * @return True if at the end, false otherwise
      */
     bool IsAtEnd() const {
         return peek().Type == TokenType::END_OF_FILE;
     }
     
     /**
      * @brief Utility method to get the current token without consuming it
      * @return The current token
      */
     Token peek() const {
         return Tokens[Current];
     }
     
     /**
      * @brief Utility method to get the previous token
      * @return The previous token
      */
     Token Previous() const {
         return Tokens[Current - 1];
     }
     
     /**
      * @brief Utility method to consume the current token if it matches the expected type
      * @param type The token type to check against
      * @param message The error message to display if the token doesn't match
      * @return The consumed token
      */
     Token Consume(TokenType Type, const std::string& Message) {
         if (Check(Type)) return advance();
         
         ErrorReporter.ReportError(Message, peek().Location);
         throw std::runtime_error(Message);
     }
     
     /**
      * @brief Utility method to synchronize after an error
      * 
      * This method skips tokens until it finds a token that can be the start of a new statement.
      */
     void Synchronize() {
         advance();
         
         while (!IsAtEnd()) {
             if (Previous().Type == TokenType::SEMICOLON) return;
             
             switch (peek().Type) {
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