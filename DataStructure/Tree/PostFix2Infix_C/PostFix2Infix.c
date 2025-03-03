#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void push(char* element, char* stack[], int* top)
{
    stack[(*top)] = element;
    (*top)++;
}



char* pop(char* stack[], int* top)
{
    int item;
    if (*top < -1)
    {
        return NULL;
    }
    else {
        (*top)--;
        item = (*top);
        return stack[item];
    }
}

int isOperand(char x)
{
    return (x >= 'a' && x <= 'z') ||
        (x >= 'A' && x <= 'Z');
}

void Posfix2Infix(char* szPostfixInput, char* szInfixOutput)
{
    char* stack[10] = { 0 };
    int top = 0;
    int i = 0;
    char* tempOp1;
    char* tempOp2;
 
    for (i = 0; szPostfixInput[i] != '\0'; i++)
    {

        if (isOperand(szPostfixInput[i]))
        {
            char* tempExp = (char*)malloc(sizeof(char) * 2);
            if (!tempExp) return;
            tempExp[0] = szPostfixInput[i];
            tempExp[1] = '\0';
            push(tempExp, stack, &top);
        }
        else
        {
            tempOp1 = pop(stack, &top);
            tempOp2 = pop(stack, &top);

            // Calculate required size: "(" + op2 + operator + op1 + ")" + '\0'
            size_t needed = 1 + strlen(tempOp2) + 1 + strlen(tempOp1) + 1 + 1;
            char* tempExp = (char*)calloc(needed, sizeof(char));
            if (!tempExp) { free(tempOp1); free(tempOp2); return; }

            strcat(tempExp, "(") ;
            strcat(tempExp, tempOp2);
            char op[2] = { szPostfixInput[i], '\0' };
            strcat(tempExp, op);
            strcat(tempExp, tempOp1);
            strcat(tempExp, ")");

            // Free intermediate strings that are no longer on the stack
            free(tempOp1);
            free(tempOp2);

            push(tempExp, stack, &top);
        }
    }
    char* result = pop(stack, &top);
    if (result) {
        strcpy(szInfixOutput, result);
        free(result);
    }
}


int main()
{

    char szPostfixInput1[20] = "ab*c+";
    char szPostfixInput2[20] = "abc/-ad/e-*";
    char szInfixOutput1[64] = { 0 };
    char szInfixOutput2[64] = { 0 };

    Posfix2Infix(szPostfixInput1, szInfixOutput1);
    Posfix2Infix(szPostfixInput2, szInfixOutput2);

    printf("Postfix input 1: %s \n", szPostfixInput1);
    printf("Postfix output 2: %s \n\n", szInfixOutput1);

    printf("Infix input 1: %s \n", szPostfixInput2);
    printf("Infix output 2: %s \n\n", szInfixOutput2);

    return 0;
}