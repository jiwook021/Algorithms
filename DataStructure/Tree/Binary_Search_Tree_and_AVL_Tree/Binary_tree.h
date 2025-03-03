#ifndef __BINARY_TREE_H__
#define __BINARY_TREE_H__

// btreenode memory info
// [data] (btreeNode*)[left] (btreeNode*)[right]
typedef struct BtreeNode 
{
    int Data_;
    struct BtreeNode* Left_;
    struct BtreeNode* Right_;
}BtreeNode; 

void InitBtreeNode(int Data, BtreeNode* SelfNode);
const int GetData(BtreeNode* SelfNode); 
void SetData(int Data, BtreeNode* SelfNode);
BtreeNode* GetLeftTree(BtreeNode* SelfNode);
BtreeNode* GetRightTree(BtreeNode* SelfNode);
void MakeLeftTree(BtreeNode* Sub, BtreeNode* SelfNode);
void MakeRightTree(BtreeNode* Sub, BtreeNode* SelfNode);
BtreeNode* RemoveLeftTree(BtreeNode* SelfNode);
BtreeNode* RemoveRightTree(BtreeNode* SelfNode);
void ChangeLeftTree(BtreeNode* Sub, BtreeNode* SelfNode);
void ChangeRightTree(BtreeNode* Sub, BtreeNode* SelfNode);
void TravelInorder(BtreeNode* Root);
void TravelPreorder(BtreeNode* Root);
void TravelPostorder(BtreeNode* Root); 

#endif