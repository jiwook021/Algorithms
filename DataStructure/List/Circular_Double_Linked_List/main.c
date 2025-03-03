#include "Circular_doublelinkedlist.h"


Linked_list *initlinkedlist()
{
	Linked_list* l = (Linked_list*)malloc(sizeof(Linked_list));
	l -> head = NULL;
	l -> tail = NULL;
	l -> size = 0;
	return l;
}

void vInsert(int data, Linked_list* l) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    
    if (l->size == 0) {
        newNode->next = newNode; // 새 노드의 next와 prev를 자기 자신으로 설정
        newNode->prev = newNode;
        l->head = newNode;
        l->tail = newNode;
    } else {
        newNode->prev = l->tail; // 새 노드의 prev를 현재 tail로 설정
        newNode->next = l->head; // 새 노드의 next를 head로 설정
        l->tail->next = newNode; // 현재 tail의 next를 새 노드로 설정
        l->head->prev = newNode; // head의 prev를 새 노드로 설정
        l->tail = newNode; // 새 노드를 새 tail로 업데이트
    }
    l->size++;
    printf("Inserted %d at tail\n", newNode->data); // 출력 메시지 수정
}

void vRemove(int data, Linked_list* l)
{
	if (l->head == NULL) return;
	Node* current = l->head;
	Node* nPrev = l->head;
	if (!(data == (l->head->data)))
	{
		for (int i = 0; i < l->size; i++)
		{
			current = current->next;
			if (data == current->data)
			{
				current->prev->next = current->next;
				current->next->prev = current->prev;
				free(current);
				l->size--;
				printf("Deleted %d ",data);
				return;
			}			
		}
		printf("Cannot found %d\n", data);
		return;			
	}
	else 
	{
		l->head = l->head->next;
		free(nPrev);
		l->size--;
		printf("Deleted %d ",data);
		return;
	}
}

void vSearch(int data, Linked_list* l)
{
	Node* current = l->head;
	if (l->size == 0)
	{
		printf("\nEmpty List\n");
	}
	for (int i = 0; i < l->size; i++)
	{
		current = current->next;
		if (data == current->data) 
		{
			printf("found :%d\n", current->data);
			return;
		}
	}
	printf("Did not Found :%d\n", data);
	return;
}

void vPrint(Linked_list* l)
{
	printf("\n");
	Node* current = l->head;
	for (int i = 0; i < l->size; i++)
	{
		printf("%d ", current->data);
		current = current->next;
	}
	printf("\n");
}

int random_number()
{
	return rand() % 20;
}

int idelete_random_number()
{
	return rand() % 10;
}


void example1(Linked_list *l)
{
	for(int i=0;i<=15;i++)
	{
		vInsert(i,l);
	}
	vPrint(l);
	for(int i=0; i<15;i++)
	{
		vSearch((random_number()),l);
	}
	for(int i=0;i<=10;i++)
	{
		vRemove(idelete_random_number(),l);
	}
	vPrint(l);
}

void example2(Linked_list *l)
{
	int iInput, iSelection;
	while (1)
	{
		printf("\n\nInsert Node with 0 or Delete Node with 1 and Number\n");
		scanf("%d %d", &iSelection, &iInput);
		if (iSelection == 0)
		{
			vInsert(iInput,l);
		}
		if (iSelection == 1)
		{
			vRemove(iInput,l);
		}
		if ((iSelection == -1) && (iInput == -1))
		{
			return;
		}
		vPrint(l);
	}
}

int main()
{
	Linked_list* l = initlinkedlist(); 
	example1(l);
	example2(l);
	return 0;
}