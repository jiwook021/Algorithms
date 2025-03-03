#ifndef __LINKED_LIST__
#define __LINKED_LIST__

typedef struct _node
{
    int Col_;
    int Row_;
    char Data_;
    struct _node* Next_;
    struct _node* Below_;
} Node;

typedef struct _linkedList
{
    Node* Head_;
    Node* Current_;
    int Size_;
} LinkedList;

void InitList(LinkedList* List);
void LNInsert(LinkedList* List, Node* NewNode, char DataVal);
void LBInsert(LinkedList* List, Node* NewNode, char DataVal);
void NodeDelete(LinkedList* List, char DataVal);
void NodeBelowDelete(LinkedList* List, char DataVal);

#endif