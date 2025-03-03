// #include <iostream>
// using namespace std;

// int main() {
//     int rows = 3;
//     int cols = 4;
    
//     // 데이터를 위한 단일 블록 할당
//     int* data = new int[rows * cols];
    
//     // 행 포인터 배열 할당
//     int** arr = new int*[rows];
    
//     // 각 행 포인터가 적절한 위치를 가리키도록 설정
//     for (int i = 0; i < rows; i++) {
//         arr[i] = &data[i * cols];
//     }
    
//     // 배열 초기화
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             arr[i][j] = i * cols + j;
//         }
//     }

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             data[i*cols + j] = i * cols + j;
//         }
//     }
    
//     // 배열 출력
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             cout << arr[i][j] << "\t";
//         }
//         cout << endl;
//     }

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             cout << data[i*cols + j]  << "\t";
//         }
//         cout << endl;
//     }
    
//     // 메모리 해제 (단일 블록과 포인터 배열)
//     delete[] data;
//     delete[] arr;
    
//     return 0;
// }

#include <iostream>
#include <vector>
using namespace std;

int main() {
    int rows = 3;
    int cols = 4;
    
    // 2차원 벡터 선언 및 크기 지정
    vector<vector<int>> arr(rows, vector<int>(cols));
    
    // 배열 초기화
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[i][j] = i * cols + j;
        }
    }
    
    // 배열 출력
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << arr[i][j] << "\t";
        }
        cout << endl;
    }
    
    // 벡터는 자동으로 메모리를 관리하므로 수동 해제 불필요
    
    return 0;
}