#ifndef __Circular_doublelinkedlist_H__
#define __Circular_doublelinkedlist_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct node
{
	int data;
	struct node* next;
	struct node* prev;
}Node;

typedef struct linked_list
{
	Node* head;
	Node* tail;
	int size;
}Linked_list;

void vInsert(int data, Linked_list* l);
void vRemove(int data, Linked_list* l);
void vSearch(int data, Linked_list* l);
void vPrint();
#endif 