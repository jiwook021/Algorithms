#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define QUEUE_SIZE 20

typedef struct {
    int items[QUEUE_SIZE];
    int front, rear;
} Circular_Queue;

Circular_Queue* initCircular_Queue() {
    Circular_Queue* CQ = (Circular_Queue*)malloc(sizeof(Circular_Queue));
    CQ->front = -1;
    CQ->rear = -1;
    return CQ;
}

bool QisEmpty(Circular_Queue* CQ) {
    return CQ->front == -1;
}

bool QisFull(Circular_Queue* CQ) {
    return (CQ->rear + 1) % QUEUE_SIZE == CQ->front;
}

void enqueue(int data, Circular_Queue* CQ) {
    if (QisFull(CQ)) {
        printf("Queue is full\n");
        return;
    }
    if (QisEmpty(CQ)) {
        CQ->front = 0;
    }
    CQ->rear = (CQ->rear + 1) % QUEUE_SIZE;
    CQ->items[CQ->rear] = data;
    printf("Enqueued: %d\n", data);
}

int dequeue(Circular_Queue* CQ) {
    if (QisEmpty(CQ)) {
        printf("Queue is empty\n");
        return -1;
    }
    int data = CQ->items[CQ->front];
    if (CQ->front == CQ->rear) {
        CQ->front = -1;
        CQ->rear = -1;
    } else {
        CQ->front = (CQ->front + 1) % QUEUE_SIZE;
    }
    return data;
}

int peek(Circular_Queue* CQ) {
    if (QisEmpty(CQ)) {
        printf("Queue is empty\n");
        return -1;
    }
    return CQ->items[CQ->front];
}

int main() {
    Circular_Queue* CQ = initCircular_Queue();
    for (int i = 0; i < QUEUE_SIZE; i++) {
        enqueue(rand() % 10 + 1, CQ);
    }
    printf("Starting to dequeue...\n");
    while (!QisEmpty(CQ)) {
        printf("Dequeued: %d\n", dequeue(CQ));
    }
    return 0;
}
