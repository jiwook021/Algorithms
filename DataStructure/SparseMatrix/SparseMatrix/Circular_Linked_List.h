#ifndef __CIRCULAR_LINKED_LIST_H__
#define __CIRCULAR_LINKED_LIST_H__



typedef struct Node {
    int Row_;
    int Column_;
    char Data_;
    struct Node* NextNode_;
    struct Node* BelowNode_;
} Node;

typedef struct LinkedList {
    int Size_;
    Node* Current_;
    Node* Tail_;
    Node* Head_;
} LinkedList;

#endif 