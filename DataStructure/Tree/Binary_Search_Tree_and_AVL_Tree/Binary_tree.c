#include "Binary_tree.h"
#include <stdio.h>
#include <stdlib.h>

void InitBtreeNode(int Data, BtreeNode* SelfNode)
{
    SelfNode->Data_ = Data;
    SelfNode->Left_ = NULL;
    SelfNode->Right_ = NULL;
}
const int GetData(BtreeNode* SelfNode)
{
    return SelfNode->Data_;
}
void SetData(int Data, BtreeNode* SelfNode)
{
    SelfNode->Data_ = Data; 
}
BtreeNode* GetLeftTree(BtreeNode* SelfNode)
{
    return SelfNode->Left_;
}
BtreeNode* GetRightTree(BtreeNode* SelfNode)
{
    return SelfNode->Right_;
}
void MakeLeftTree(BtreeNode* Sub, BtreeNode* SelfNode)
{
    if (SelfNode->Left_ != NULL)
        free(SelfNode->Left_);

    SelfNode->Left_ = Sub;
}
void MakeRightTree(BtreeNode* Sub, BtreeNode* SelfNode)
{
    if (SelfNode->Right_ != NULL)
        free(SelfNode->Right_);

    SelfNode->Right_ = Sub;
}
BtreeNode* RemoveLeftTree(BtreeNode* SelfNode)
{
    BtreeNode* DelNode = NULL;

    if (SelfNode != NULL) {
        DelNode = SelfNode->Left_;
        SelfNode->Left_ = NULL; 
    }
    return DelNode;
}
BtreeNode* RemoveRightTree(BtreeNode* SelfNode)
{
    BtreeNode* DelNode = NULL;
    if (SelfNode != NULL) {
        DelNode = SelfNode->Right_;
        SelfNode->Right_ = NULL;
    }
    return DelNode;
}
void ChangeLeftTree(BtreeNode* Sub, BtreeNode* SelfNode)
{
    SelfNode->Left_ = Sub; 
}
void ChangeRightTree(BtreeNode* Sub, BtreeNode* SelfNode)
{
    SelfNode->Right_ = Sub; 
}
void TravelInorder(BtreeNode * Root)
{
    if (Root == NULL)
    {
        return;
    }
    TravelInorder(GetLeftTree(Root));
    printf("%d ", GetData(Root));
    TravelInorder(GetRightTree(Root));
}

void TravelPreorder(BtreeNode* Root)
{
    if (Root == NULL)
    {
        return;
    }
    printf("%d ", GetData(Root));
    TravelPreorder(GetLeftTree(Root));
    TravelPreorder(GetRightTree(Root));
}

void TravelPostorder(BtreeNode* Root)
{
    if (Root == NULL)
    {
        return;
    }
    TravelInorder(GetLeftTree(Root));
    TravelInorder(GetRightTree(Root));
    printf("%d ", GetData(Root));
}
