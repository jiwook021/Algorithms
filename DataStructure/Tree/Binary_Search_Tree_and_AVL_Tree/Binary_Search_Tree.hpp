#ifndef __BINARY_SEARCH_TREE_H__
#define __BINARY_SEARCH_TREE_H__
#include "Binary_tree.h"

void BSTMakeAndInit(BtreeNode ** PRoot); 
const int BSTGetNodeData(BtreeNode* Bst); 
BtreeNode* BSTInsert(BtreeNode** PRoot, int Data); 
BtreeNode* BSTSearch(BtreeNode* Bst, int Target);
BtreeNode* BSTRemove(BtreeNode** PRoot, int Target);
#endif