#include <stdio.h>
#include <stdbool.h>
#define MAXSIZE 20 

typedef struct circular_dequeue 
{
    int data[MAXSIZE];
    int head;
    int rear;
    int size;
}Deque;

static void init_circular_dequeue(struct circular_dequeue *cd)
{
    cd->head = -1;
    cd->rear = -1;
    cd->size= 0;
}
static bool isDequeEmpty(Deque* deque) {
    return deque->size==0;
}
// Check if deque is full
static bool isDequeFull(Deque* deque) {
    return (deque->size == MAXSIZE);
}

// Insert at front
static void insertFront(Deque* deque, int key) 
{
    if(isDequeFull(deque))
    {
        return;
    }
    int index = ((deque -> head -1) + MAXSIZE) % MAXSIZE;
    deque->head = index; 
    if(isDequeEmpty(deque))
    {
        deque->rear = index; 
    }
    deque->size++;
    deque->data[index] = key;
}
// Insert at rear
static void insertRear(Deque* deque, int key) {
    if(isDequeFull(deque))
    {
        return;
    }
    int index = (deque -> rear + 1) % MAXSIZE;
    deque->rear = index; 
    if(isDequeEmpty(deque))
    {
        deque->head = index; 
    }
    deque->size++;
    deque->data[index] = key;
}

// Delete from front
static int deleteFront(Deque* deque) {
    if(isDequeEmpty(deque))
    {
        return -1;
    }
    int result = deque->data[deque->head];
    deque -> head = (deque -> head + 1) % MAXSIZE;
    deque->size--;
    return result;
}
// Delete from rear
static int deleteRear(Deque* deque) {
    if(isDequeEmpty(deque))
    {
        return -1;
    }
    int result = deque->data[deque->rear];
    deque->rear = ((deque -> rear -1) + MAXSIZE) % MAXSIZE;
    deque->size--;
    return result;
}

int main()
{
    Deque deque;
    init_circular_dequeue(&deque);    
    for(int i = 0;i<30;i++)
    {
        insertFront(&deque, i);
        //insertRear(&deque, i);
    }
    for(int i = 0;i<5;i++)
    {
        printf("Front element: %d\n", deleteFront(&deque));
        printf("Rear element: %d\n", deleteRear(&deque));
    }
    for(int i = 0;i<10;i++)
    {
        insertFront(&deque, i);
       // insertRear(&deque, i);
    }
    for(int i = 0;i<20;i++)
    {
        printf("Front element: %d\n", deleteFront(&deque));
        printf("Rear element: %d\n", deleteRear(&deque));
    }    
}