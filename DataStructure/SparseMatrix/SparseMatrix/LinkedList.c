#include "LinkedList.h"
#include <stdio.h>
#include <stdlib.h>

void InitList(LinkedList* List)
{
    List->Size_ = 0; 
    List->Head_ = NULL;
}

void LNInsert(LinkedList* List, Node* NewNode, char DataVal)
{
    NewNode->Data_ = DataVal;
    NewNode->Next_ = NULL;
    
    if (List->Head_ == NULL)
    {
        List->Head_ = NewNode;
        List->Head_->Next_ = List->Head_;
    }
    else
    {
        List->Current_ = List->Head_;
        while (List->Current_->Next_ != List->Head_)
        {
            List->Current_ = List->Current_->Next_;
        }
        List->Current_->Next_ = NewNode;
        NewNode->Next_ = List->Head_;
    }
    List->Size_++;
}

void LBInsert(LinkedList* List, Node* NewNode, char DataVal)
{
    NewNode->Below_ = NULL;

    if (List->Head_ == NULL)
    {
        List->Head_ = NewNode;
        List->Head_->Below_ = List->Head_;
    }
    else
    {
        List->Current_ = List->Head_;
        while (List->Current_->Below_ != List->Head_)
        {
            List->Current_ = List->Current_->Below_;
        }
        List->Current_->Below_ = NewNode;
        NewNode->Below_ = List->Head_;
    }
    List->Size_++;
}


void NodeDelete(LinkedList* List, char DataVal)
{
    if (List->Head_ == NULL) return;
    List->Current_ = List->Head_;
    Node* NPrev = List->Head_;

    if (!(DataVal == (List->Head_->Data_)))
    {
        for (int I = 0; I < List->Size_; I++)
        {
            NPrev = List->Current_;
            List->Current_ = List->Current_->Next_;

            if (List->Current_ == NULL)
            {
                printf("\nCannot found %d\n", DataVal);
                return;
            }
            if (DataVal == List->Current_->Data_)
            {
                NPrev->Next_ = List->Current_->Next_;
                free(List->Current_);
                List->Size_--;
                return;
            }
        }
    }
    else
    {
        List->Head_ = List->Head_->Next_;
        free(NPrev);
        List->Size_--;
        return;
    }
}

