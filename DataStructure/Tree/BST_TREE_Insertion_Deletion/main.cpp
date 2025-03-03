#include <iostream>
#include <memory>

// Node class definition using std::unique_ptr for left and right children
class Node {
public:
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    int data;
    // Constructor to initialize Node with data
    Node(int data) : data(data), left(nullptr), right(nullptr) {}
};

// Tree class definition with smart pointers
class Tree {
public:
    std::unique_ptr<Node> root;

    // Constructor: root initialized as nullptr
    Tree() : root(nullptr) {}

    // Function to initialize a new Node using std::make_unique
    std::unique_ptr<Node> initNode(int data) {
        return std::make_unique<Node>(data);
    }

    // Insert function to add a node to the Binary Search Tree
    void pushBST(int data) {
        root = insertNode(std::move(root), data);
    }

    // Helper function for pushBST (inserts into the correct position)
    std::unique_ptr<Node> insertNode(std::unique_ptr<Node> node, int data) {
        if (!node) {
            return initNode(data);  // Create a new node if the spot is empty
        }
        if (data < node->data) {
            node->left = insertNode(std::move(node->left), data);  // Insert to the left subtree
        } else {
            node->right = insertNode(std::move(node->right), data);  // Insert to the right subtree
        }
        return node;  // Return the updated tree
    }

    // Inorder traversal to print the tree
    void inorderTraversal(const Node* node) const {
        if (node) {
            inorderTraversal(node->left.get());
            std::cout << node->data << " ";
            inorderTraversal(node->right.get());
        }
    }

    // Public interface to print the entire tree
    void printTree() const {
        inorderTraversal(root.get());
        std::cout << std::endl;
    }
};

int main() {
    auto tr = std::make_unique<Tree>();  // Create a new tree using unique_ptr
    tr->pushBST(10);
    tr->pushBST(5);
    tr->pushBST(20);
    tr->pushBST(3);
    tr->pushBST(7);
    // Print the tree using inorder traversal
    std::cout << "Inorder Traversal: ";
    tr->printTree();
    return 0;
}
