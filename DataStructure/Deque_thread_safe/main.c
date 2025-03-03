#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct Node
{
    int data;
    struct Node* next;
    struct Node* prev; 
}node; 

typedef struct Deque
{
    int data;
    int sz;
    node* front;
    node* back;
    pthread_mutex_t mutex;
    pthread_cond_t cond; // 조건 변수 추가
}deque;

typedef struct {
    deque* param1;
    int param2;
} ThreadParams;

// ThreadParams 구조체에 동적 메모리 할당 및 해제 함수 추가
ThreadParams* createThreadParams(deque* d, int data) {
    ThreadParams* params = (ThreadParams*)malloc(sizeof(ThreadParams));
    if (params == NULL) {
        perror("Failed to allocate ThreadParams");
        exit(EXIT_FAILURE);
    }
    params->param1 = d;
    params->param2 = data;
    return params;
}

deque* initdeque()
{
    deque* d = (deque*) malloc(sizeof(deque));
    d->front = NULL;
    d->back = NULL;
    d->sz = 0;
    pthread_mutex_init(&d->mutex, NULL);
    pthread_cond_init(&d->cond, NULL); // 조건 변수 초기화
    return d;
}

void destroydeque(deque* d) {
    pthread_cond_destroy(&d->cond); // 조건 변수 해제
    pthread_mutex_destroy(&d->mutex); // 뮤텍스 해제
    // ... 기타 필요한 자원 해제 ...
    free(d);
}

void push_front(void* arg)
{
    ThreadParams* params = (ThreadParams*)arg;
    deque* d= params->param1; 
    int data = params->param2;
    pthread_mutex_lock(&d->mutex);
    node * newNode = (node*) malloc(sizeof(node)); 
    newNode->data = data;
    newNode->prev = NULL;
    d->sz++;
    if(d->sz == 1)
    {
        newNode->next = NULL;
        d->front = d->back = newNode;
        pthread_mutex_unlock(&d->mutex);
        return;
    }
    d->front->prev = newNode; 
    newNode->next = d->front;
    d->front = newNode;
    pthread_mutex_unlock(&d->mutex);
    return; 
}
void push_back(void* arg)
{
    ThreadParams* params = (ThreadParams*)arg; 
    printf("current: %d  \n",params->param2);
    pthread_mutex_lock(&params->param1->mutex);
    deque* d= params->param1; 
    int data = params->param2;
    node * newNode = (node*) malloc(sizeof(node)); 
    newNode->data = data;
    newNode->next = NULL;
    d->sz++;
    if(d->sz == 1)
    {
        newNode->prev = NULL;
        d->front = d->back = newNode;
        printf("%d  \n",newNode->data);
        pthread_mutex_unlock(&d->mutex);
        pthread_cond_signal(&d->cond);
        return;
    }
    d->back->next = newNode; 
    newNode->prev = d->back;
    d->back = newNode;
    printf("%d  \n",newNode->data);
    pthread_mutex_unlock(&d->mutex);
    pthread_cond_signal(&d->cond);
    return; 
}

int pop_front(deque *d)
{
    pthread_mutex_lock(&d->mutex);
    if (d->sz == 0)
        return -1;
    d->sz--;
    int result = d->front->data;
    node* dnode = d->front;
    if (d->front == d->back) 
    {
        d->front = NULL;
        d->back = NULL;
    }
    else
    {
        d->front = d->front->next; 
        d->front->prev = NULL;
    }
    free(dnode);
    printf("%d  ", result);
    pthread_mutex_unlock(&d->mutex);
    return result;
}
int pop_back(deque *d)
{ 
    pthread_mutex_lock(&d->mutex);
    if (d->sz == 0)
        return -1;
    d->sz--;
    int result = d->back->data;
    node* dnode = d->back;
    if (d->front == d->back) 
    {
        d->front = NULL;
        d->back = NULL;
    }
    else
    {
        d->back = d->back->prev; 
        d->back->next = NULL;
    }
    free(dnode);
    printf("%d  ", result);
    pthread_mutex_unlock(&d->mutex);
    return result;
}


void print_deque(deque *d)
{
    pthread_mutex_lock(&d->mutex);
    node* current = d->front;
    while(current != NULL)
    {
        printf("%d ",current->data);
        current = current ->next;
    }
    printf("\n");
    pthread_mutex_unlock(&d->mutex);
}


#define NUM_THREADS 20
int main()
{
    deque* d = initdeque();
    pthread_t threads[NUM_THREADS];
    pthread_t push_threads[NUM_THREADS];
    ThreadParams params[NUM_THREADS]; // 각 스레드별 별도의 ThreadParams
    int i =0; 
    for(i;i<19;i++)
    {
        ThreadParams* params = createThreadParams(d, i);
        pthread_create(&push_threads[i], NULL, &push_back, params);
        pthread_cond_wait(&d->cond, &d->mutex);
    }
    pthread_mutex_unlock(&d->mutex);
    for (int i = 0; i < 19; i++) {
        pthread_join(push_threads[i], NULL); // push_threads에 대한 pthread_join
    }   
    printf("\n");
    for (int i = 0; i < 19; i++) {
        pthread_create(&threads[i], NULL, (void*(*)(void*))&pop_back, (void*)d);
    }
    for (int i = 0; i < 19; i++) {
        pthread_join(threads[i], NULL);
    }
    destroydeque(d);
    return 0;
}