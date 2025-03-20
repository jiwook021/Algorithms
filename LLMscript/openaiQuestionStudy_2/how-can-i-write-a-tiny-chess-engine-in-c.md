# How can I write **a tiny chess engine in C++**?

Creating a simple chess engine in C++ can be a fun and educational project. Chess engines usually consist of a board representation, a move generator, an evaluation function, and a search algorithm. Here, I'll outline steps and provide basic snippets to get you started on creating a tiny chess engine. This engine will be quite basic and is not designed to compete with advanced engines like Stockfish or AlphaZero.

### Step 1: Board Representation

The first step is to decide how to represent the chess board. One common method is using an 8x8 array where each piece is represented by a character or integer. For simplicity, we can use characters:

- 'P' for white pawns, 'p' for black pawns
- 'N' for white knights, 'n' for black knights
- 'B' for white bishops, 'b' for black bishops
- 'R' for white rooks, 'r' for black rooks
- 'Q' for white queens, 'q' for black queens
- 'K' for white kings, 'k' for black kings
- '.' for empty squares

Here is a simple way to initialize a chess board:

```cpp
#include <iostream>
#include <vector>

typedef std::vector<std::vector<char>> Board;

Board setupBoard() {
    Board board = {
        {'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'},
        {'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.'},
        {'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'},
        {'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'}
    };
    return board;
}

void printBoard(const Board& board) {
    for (const auto& row : board) {
        for (char piece : row) {
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
}
```

### Step 2: Move Generation

You need a function to generate legal moves for a given position. This is a complex part because you need to handle different move rules for each piece type, including special moves like castling, en passant, and pawn promotion.

Here's a very simplified version of a move generation function for pawns:

```cpp
#include <vector>
#include <utility> // for std::pair

typedef std::pair<int, int> Position;
typedef std::vector<Position> MoveList;

void generatePawnMoves(const Board& board, MoveList& moves, int row, int col, char pawn) {
    // Assuming pawns move up only one step and capture diagonally
    int direction = (pawn == 'P') ? -1 : 1;
    int nextRow = row + direction;
    
    // Move forward
    if (board[nextRow][col] == '.') {
        moves.push_back({nextRow, col});
    }
    // Capture diagonally
    if (col > 0 && board[nextRow][col - 1] != '.' && islower(board[row][col]) != islower(board[nextRow][col - 1])) {
        moves.push_back({nextRow, col - 1});
    }
    if (col < 7 && board[nextRow][col + 1] != '.' && islower(board[row][col]) != islower(board[nextRow][col + 1])) {
        moves.push_back({nextRow, col + 1});
    }
}
```

### Step 3: Evaluation Function

An evaluation function estimates the strength of a position. A simple approach could be counting material balance:

```cpp
int evaluate(const Board& board) {
    int score = 0;
    for (const auto& row : board) {
        for (char piece : row) {
            switch (piece) {
                case 'P': score += 1; break;
                case 'N': case 'B': score += 3; break;
                case 'R': score += 5; break;
                case 'Q': score += 9; break;
                case 'K': score += 100; break;
                case 'p': score -= 1; break;
                case 'n': case 'b': score -= 3; break;
                case 'r': score -= 5; break;
                case 'q': score -= 9; break;
                case 'k': score -= 100; break;
                default: break;
            }
        }
    }
    return score;
}
```

### Step 4: Search Algorithm

The minimax algorithm with alpha-beta pruning is a common choice. It tries to minimize the possible loss for a worst-case scenario:

```cpp
int minimax(Board& board, int depth, bool maximizingPlayer) {
    if (depth == 0) {
        return evaluate(board);
    }
    
    if (maximizingPlayer) {
        int maxEval = INT_MIN;
        // Generate and evaluate all moves
        return maxEval;
    } else {
        int minEval = INT_MAX;
        // Generate and evaluate all moves
        return minEval;
    }
}

```

### Step 5: Putting It All Together

Create a main function to initialize the board, generate moves, and start the search.

This will give you a very basic chess engine. Improving it involves handling all legal moves, implementing more sophisticated evaluation and better search technique, and potentially using opening books and endgame tablebases.