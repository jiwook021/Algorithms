#include <iostream>
#include <memory>

void incrementInteger(std::unique_ptr<int[]>& arr, int& size) {
    for (int i = size - 1; i >= 0; --i) {
        if (arr[i] < 9) {
            arr[i]++;
            return;
        } else {
            arr[i] = 0;
        }
    }
    std::unique_ptr<int[]> newarr = std::make_unique<int[]>(size+1);
    std::copy(arr.get(), arr.get() + size, newarr.get());
    newarr[0] = 1;
    newarr[size] = 0;
    size = size+1;

    // Replace the old array with the new one
    arr = std::move(newarr);
}

void incrementArbInteger() {
    int size = 3;
    //int* number = new int[size];    
    std::unique_ptr<int[]> number = std::make_unique<int[]> (size); 
    number[0] = 1;
    number[1] = 2;
    number[2] = 9;
    incrementInteger(number, size);
    for (int i = 0; i < size; i++) {
        std::cout << number[i]; 
    }
    std::cout << std::endl;


    int sizes = 3;
    std::unique_ptr<int[]> number2 = std::make_unique<int[]> (sizes); 
    number2[0] = 9;
    number2[1] = 9;
    number2[2] = 9;
    
    incrementInteger(number2, sizes);
    for (int i = 0; i < sizes; i++) {
        std::cout << number2[i]; 
    }
    std::cout << std::endl;
    return;
}

int main()
{
    incrementArbInteger();
    return 0;
}