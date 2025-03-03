#include <iostream>
#include <string>

enum class Strategy { MTF, Transpose, Count };

template<typename T>
class SelfOrganizingList {
private:
    struct Node {
        T data;
        int count;
        Node* next;
        Node(const T& item) : data(item), count(0), next(nullptr) {}
    };
    
    Node* head;
    Strategy strategy;
    
public:
    SelfOrganizingList(Strategy s = Strategy::MTF) : head(nullptr), strategy(s) {}
    
    ~SelfOrganizingList() {
        while (head) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }
    
    void insert(const T& item) {
        Node* newNode = new Node(item);
        newNode->next = head;
        head = newNode;
    }
    
    bool find(const T& item) {
        if (!head) return false;
        
        // 첫 번째 노드인 경우 특별 처리
        if (head->data == item) {
            head->count++;
            return true;
        }
        
        Node* prev = head;
        Node* current = head->next;
        Node* prevPrev = nullptr;
        
        while (current) {
            if (current->data == item) {
                current->count++;
                
                switch (strategy) {
                    case Strategy::MTF:
                        // Move-to-Front 전략
                        prev->next = current->next;
                        current->next = head;
                        head = current;
                        break;
                        
                    case Strategy::Transpose:
                        // Transpose 전략
                        prev->next = current->next;
                        current->next = prev;
                        if (prevPrev) {
                            prevPrev->next = current;
                        } else {
                            head = current;
                        }
                        break;
                        
                    case Strategy::Count:
                        // Count 전략
                        Node* scan = head;
                        Node* scanPrev = nullptr;
                        
                        // 접근 횟수가 더 많은 노드 앞으로 이동
                        while (scan != current && scan->count >= current->count) {
                            scanPrev = scan;
                            scan = scan->next;
                        }
                        
                        // 이동이 필요한 경우
                        if (scan != current) {
                            prev->next = current->next;
                            current->next = scan;
                            
                            if (scanPrev) {
                                scanPrev->next = current;
                            } else {
                                head = current;
                            }
                        }
                        break;
                }
                
                return true;
            }
            
            prevPrev = prev;
            prev = current;
            current = current->next;
        }
        
        return false;
    }
    
    void display() const {
        Node* current = head;
        while (current) {
            std::cout << current->data;
            if (strategy == Strategy::Count) {
                std::cout << "(접근 횟수: " << current->count << ")";
            }
            std::cout << " -> ";
            current = current->next;
        }
        std::cout << "nullptr" << std::endl;
    }
    
    void setStrategy(Strategy s) {
        strategy = s;
    }
};

int main() {
    // 예제: 세 가지 전략 테스트
    std::cout << "=== Move-to-Front 전략 ===" << std::endl;
    SelfOrganizingList<std::string> mtfList(Strategy::MTF);
    mtfList.insert("D");
    mtfList.insert("C");
    mtfList.insert("B");
    mtfList.insert("A");
    
    std::cout << "초기 리스트: ";
    mtfList.display();
    
    std::cout << "C 찾기 후: ";
    mtfList.find("C");
    mtfList.display();
    
    std::cout << "D 찾기 후: ";
    mtfList.find("D");
    mtfList.display();
    
    std::cout << "\n=== Transpose 전략 ===" << std::endl;
    SelfOrganizingList<std::string> tList(Strategy::Transpose);
    tList.insert("D");
    tList.insert("C");
    tList.insert("B");
    tList.insert("A");
    
    std::cout << "초기 리스트: ";
    tList.display();
    
    std::cout << "C 찾기 후: ";
    tList.find("C");
    tList.display();
    
    std::cout << "D 찾기 후: ";
    tList.find("D");
    tList.display();
    
    std::cout << "\n=== Count 전략 ===" << std::endl;
    SelfOrganizingList<std::string> cList(Strategy::Count);
    cList.insert("D");
    cList.insert("C");
    cList.insert("B");
    cList.insert("A");
    
    std::cout << "초기 리스트: ";
    cList.display();
    
    std::cout << "C 찾기 후: ";
    cList.find("C");
    cList.display();
    
    std::cout << "C를 두 번 더 찾기 후: ";
    cList.find("C");
    cList.find("C");
    cList.display();
    
    std::cout << "D 찾기 후: ";
    cList.find("D");
    cList.display();
    
    return 0;
}