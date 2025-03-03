#include "AVLRebalance.hpp"
#include "Binary_Search_Tree.hpp"
#include "Binary_tree.h"
#include <iostream>

int GetHeight(BtreeNode* BstNode)
{
    int LeftH, RightH;
    if (BstNode == NULL)
        return 0;
    LeftH = GetHeight(GetLeftTree(BstNode));
    RightH = GetHeight(GetRightTree(BstNode));
    if (LeftH > RightH)
        return LeftH + 1;
    else
        return RightH + 1;
}
int GetHeightDiff(BtreeNode* BstNode)
{
    int LeftSearchHeight, RightSearchHeight; 

    if (BstNode == NULL)
        return 0; 

    LeftSearchHeight = GetHeight(GetLeftTree(BstNode));
    RightSearchHeight = GetHeight(GetRightTree(BstNode));
    return LeftSearchHeight - RightSearchHeight;
}
BtreeNode* RotateLL(BtreeNode* BstNode)
{
    BtreeNode* ParentNode; 
    BtreeNode* ChildNode;

    ParentNode = BstNode; 
    ChildNode = GetLeftTree(ParentNode); 
    ChangeLeftTree(GetRightTree(ChildNode), ParentNode); 
    ChangeRightTree(ParentNode, ChildNode);
    return ChildNode; 
}
BtreeNode* RotateRR(BtreeNode* BstNode)
{
    BtreeNode* ParentNode;
    BtreeNode* ChildNode;  
    ParentNode = BstNode;
    ChildNode = GetRightTree(ParentNode);
    ChangeRightTree(GetLeftTree(ChildNode), ParentNode);
    ChangeLeftTree(ParentNode, ChildNode);
    return ChildNode;
}
BtreeNode* RotateLR(BtreeNode* BstNode)
{
    BtreeNode* ParentNode;
    BtreeNode* ChildNode;
    ParentNode = BstNode;
    ChildNode = GetLeftTree(ParentNode);
    ChangeLeftTree(RotateRR(ChildNode), ParentNode);
    return RotateLL(ParentNode);
}
BtreeNode* RotateRL(BtreeNode* BstNode)
{
    BtreeNode* ParentNode;
    BtreeNode* ChildNode;
    ParentNode = BstNode;
    ChildNode = GetRightTree(ParentNode);
    ChangeRightTree(RotateLL(ChildNode), ParentNode);
    return RotateRR(ParentNode);
}

BtreeNode* Rebalance(BtreeNode** ParentRoot)
{
    int HeightDifference = GetHeightDiff(*ParentRoot);

    if (HeightDifference > 1)
    {
        if (GetHeightDiff(GetLeftTree(*ParentRoot)) > 0)
        {
            *ParentRoot = RotateLL(*ParentRoot);
        }
        else
        {
            *ParentRoot = RotateLR(*ParentRoot);
        }
    }

    if (HeightDifference < -1)
    {
        if (GetHeightDiff(GetRightTree(*ParentRoot)) < 0)
        {
            *ParentRoot = RotateRR(*ParentRoot);
        }
        else
        {
            *ParentRoot = RotateRL(*ParentRoot);
        }
    }
    return *ParentRoot; 
}