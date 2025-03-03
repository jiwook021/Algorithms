#include "Binary_Search_Tree.hpp"
#include "AVLRebalance.hpp"
#include <cstdlib>

void BSTMakeAndInit(BtreeNode** ParentRoot)
{
    *ParentRoot = nullptr; 
}

const int BSTGetNodeData(BtreeNode* BstNode)
{
    return GetData(BstNode);
}

BtreeNode* BSTInsert(BtreeNode** ParentRoot, int Data)
{
    if (*ParentRoot == NULL)
    {
        *ParentRoot = (BtreeNode*) malloc (sizeof(BtreeNode));
        InitBtreeNode(Data, *ParentRoot);
    }
    else if (Data < GetData(*ParentRoot))
    {
        BSTInsert(&((*ParentRoot)->Left_), Data);
        *ParentRoot = Rebalance(ParentRoot);
    }
    else if (Data > GetData(*ParentRoot))
    {
        BSTInsert(&((*ParentRoot)->Right_), Data);
        *ParentRoot = Rebalance(ParentRoot);
    }
    else
    {
        return NULL;
    }
    return *ParentRoot;
}

BtreeNode* BSTSearch(BtreeNode* BstNode, int Target)
{
    BtreeNode* ChildNode = BstNode; 
    int Cd;
    while (ChildNode != nullptr)
    {
        Cd = GetData(ChildNode);
        if (Target == Cd)
            return ChildNode;
        else if (Target < Cd)
            ChildNode = GetLeftTree(ChildNode);
        else
            ChildNode = GetRightTree(ChildNode);
    }
    return nullptr; 
}

BtreeNode* BSTRemove(BtreeNode** ParentRoot, int Target)
{
    BtreeNode *PvRoot = (BtreeNode*) malloc (sizeof(BtreeNode));
    InitBtreeNode(0, PvRoot);
    BtreeNode* PNode = PvRoot;
    BtreeNode* ChildNode = *ParentRoot; 
    BtreeNode* DNode;
    ChangeRightTree(*ParentRoot, PvRoot);
    while (ChildNode != nullptr && GetData(ChildNode) != Target)
    {
        PNode = ChildNode; 

        if (Target < GetData(ChildNode))
            ChildNode = GetLeftTree(ChildNode);
        else
            ChildNode = GetRightTree(ChildNode); 
    }
    if (ChildNode == nullptr)
        return nullptr;
    DNode = ChildNode; 
    if (GetLeftTree(DNode) == nullptr && GetRightTree(DNode) == nullptr)
    {
        if (GetLeftTree(PNode) == DNode)
            RemoveLeftTree(PNode);
        else
            RemoveRightTree(PNode); 
    }
    else if (GetLeftTree(DNode) == nullptr || GetRightTree(DNode) == nullptr)
    {
        BtreeNode* DcNode; 

        if (GetLeftTree(DNode) != nullptr)
            DcNode = GetLeftTree(DNode);
        else 
            DcNode = GetRightTree(DNode);

        if (GetLeftTree(PNode) == DNode)
            ChangeLeftTree(DcNode, PNode);
        else
            ChangeRightTree(DcNode, PNode);
    }
    else
    {
        BtreeNode* MNode = GetRightTree(DNode); 
        BtreeNode* MpNode = DNode; 
        int DelData; 

        while (GetLeftTree(MNode) != nullptr) {
            MpNode = MNode;
            MNode = GetLeftTree(MNode); 
        }
        DelData = GetData(DNode); 
        SetData(GetData(MNode), DNode);

        if (GetLeftTree(MpNode) == MNode)
            ChangeLeftTree(GetRightTree(MNode), MpNode);
        else
            ChangeRightTree(GetRightTree(MNode), MpNode);

        DNode = MNode; 
        SetData(DelData, DNode);
    }

    if (GetRightTree(PvRoot) != *ParentRoot)
        *ParentRoot = GetRightTree(PvRoot);
    free(PvRoot);
    Rebalance(ParentRoot);
    return DNode; 
}