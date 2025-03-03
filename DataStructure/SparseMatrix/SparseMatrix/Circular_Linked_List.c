#define _CRT_SECURE_NO_WARNINGS
#include "Circular_Linked_List.h"
#include <stdio.h>
#include <stdlib.h>

void ListInit(LinkedList* List)
{
    List->Current_ = NULL;
    List->Tail_ = NULL;
    List->Head_ = NULL;
    List->Size_ = 0; 
}

void VInsertion(int DataVal, LinkedList* List)
{
    Node* NewNode = (Node*)malloc(sizeof(Node));
    NewNode->Data_ = DataVal;
    NewNode->NextNode_ = NULL;

    if (List->Head_ == NULL)
    {
        List->Head_ = NewNode;
        List->Head_->NextNode_ = List->Head_;
    }
    else
    {
        List->Current_ = List->Head_;
        while (List->Current_->NextNode_ != List->Head_)
        {
            List->Current_ = List->Current_->NextNode_;
        }
        List->Current_->NextNode_ = NewNode;
        List->Tail_ = NewNode;
        NewNode->NextNode_ = List->Head_;
    }
    List->Size_++;
}


void VInsertion2(int DataVal, int Seq, LinkedList* List)
{
    Node* NewNode = (Node*)malloc(sizeof(Node));
    NewNode->Data_ = DataVal;
    NewNode->NextNode_ = NULL;

    List->Current_ = List->Head_;
    for (int I = 0; I < (Seq - 2); I++)
    {
        List->Current_ = List->Current_->NextNode_;
    }

    NewNode->NextNode_ = List->Current_->NextNode_;
    List->Current_->NextNode_ = NewNode;

    List->Tail_ = NewNode;
    List->Size_++;
}

void VRemove(int DataVal, LinkedList* List)
{
    if (List->Head_ == NULL) return;
    if (List->Size_ == 0) return;

    List->Current_ = List->Head_;
    Node* NPrev = List->Head_;

    if (!(DataVal == (List->Head_->Data_)))
    {
        int I;
        for (I = 0; I < List->Size_; I++)
        {
            NPrev = List->Current_;
        
            List->Current_ = List->Current_->NextNode_;
            
            if (List->Current_ == NULL)
            {
                printf("Cannot found %d\n", DataVal);
                return;
            }
            if (DataVal == (List->Current_->Data_))
            {
                NPrev->NextNode_ = List->Current_->NextNode_;
                free(List->Current_);
                List->Size_--;
                return;
            }
        }
    }
    else
    {
        List->Head_ = List->Head_->NextNode_;
        List->Tail_->NextNode_ = List->Head_; 
        free(NPrev);
        List->Size_--;
        return;
    }

}

void VSearch(int DataVal, LinkedList* List)
{
    List->Current_ = List->Head_;

    for (int I = 0; I < List->Size_; I++)
    {
        if (DataVal == List->Current_->Data_)
            printf("Found %d\n", DataVal);
        List->Current_ = List->Current_->NextNode_;
    }
}


void VPrint(LinkedList* List)
{
    List->Current_ = List->Head_;

    for (int I = 0; I < List->Size_; I++)
    {
        printf(("%d\n"), List->Current_->Data_);
        List->Current_ = List->Current_->NextNode_;
    }
}
