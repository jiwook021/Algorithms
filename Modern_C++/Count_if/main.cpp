#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 짝수 개수 세기
    int evenCount = std::count_if(numbers.begin(), numbers.end(), 
                                 [](int n) { return n % 2 == 0; });
    
    std::cout << "짝수의 개수: " << evenCount << std::endl;  // 출력: 5
    
    return 0;
}


// #include <iostream>
// #include <vector>
// #include <algorithm>

// int main() {
//     std::vector<int> scores = {75, 90, 65, 95, 82, 78, 88};
//     int passingScore = 80;
    
//     // 합격 점수(80점) 이상인 학생 수 세기
//     int passedStudents = std::count_if(scores.begin(), scores.end(),
//                                       [passingScore](int score) { return score >= passingScore; });
    
//     std::cout << "80점 이상 학생 수: " << passedStudents << std::endl;  // 출력: 4
    
//     return 0;
// }

    // #include <iostream>
    // #include <vector>
    // #include <algorithm>
    // #include <string>

    // // 함수 객체(functor) 정의
    // struct LengthGreaterThan {
    //     size_t threshold;
        
    //     LengthGreaterThan(size_t t) : threshold(t) {}
        
    //     bool operator()(const std::string& str) const {
    //         return str.length() > threshold;
    //     }
    // };

    // int main() {
    //     std::vector<std::string> words = {"apple", "banana", "strawberry", "kiwi", "watermelon"};
        
    //     // 길이가 6자를 초과하는 단어 개수 세기
    //     int longWords = std::count_if(words.begin(), words.end(), LengthGreaterThan(6));
        
    //     std::cout << "6자 초과 단어 수: " << longWords << std::endl;  // 출력: 2 (strawberry, watermelon)
        
    //     return 0;
    // }

//     #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <string>

// struct Person {
//     std::string name;
//     int age;
    
//     Person(const std::string& n, int a) : name(n), age(a) {}
// };

// int main() {
//     std::vector<Person> people = {
//         Person("Alice", 25),
//         Person("Bob", 42),
//         Person("Charlie", 18),
//         Person("Diana", 31),
//         Person("Edward", 55)
//     };
    
//     // 30세 이상인 사람 수 세기
//     int adultsOver30 = std::count_if(people.begin(), people.end(),
//                                     [](const Person& p) { return p.age >= 30; });
    
//     std::cout << "30세 이상 인원 수: " << adultsOver30 << std::endl;  // 출력: 3
    
//     return 0;
// }

