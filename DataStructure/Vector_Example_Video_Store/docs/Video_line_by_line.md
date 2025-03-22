# Step-by-Step Explanation: Video.cpp

Let’s break down the code **step by step** in a way that is accessible to everyone, even if you’re just starting to learn programming. I’ll explain each part of the code, why it’s written the way it is, and how it all fits together.

---

### **1. Header Files and Namespace**
```cpp
#include <iostream>
#include <string>
#include "Video.h"

#pragma once

#include <iostream>
#include <string>

using namespace std;
```

#### **What it does:**
- These lines include necessary libraries and declare the use of the `std` namespace.

#### **Explanation:**
1. **`#include <iostream>`**: This includes the **input/output stream library**, which allows the program to interact with the user via the console (e.g., printing text or reading input).
   - Example: `cout` (used later) is part of this library and is used to print text to the console.

2. **`#include <string>`**: This includes the **string library**, which allows the program to work with text (e.g., storing names, dates, etc.).
   - Example: The `string` data type is used to store text like `"Inception"` or `"Sci-Fi"`.

3. **`#include "Video.h"`**: This includes a custom header file, `Video.h`, which likely contains the **class definition** for `Videotape`. Header files are used to separate class declarations from their implementations.

4. **`#pragma once`**: This is a **preprocessor directive** that ensures the header file is only included once in the program, preventing duplicate definitions.

5. **`using namespace std;`**: This tells the compiler to use the **standard namespace**, which contains common C++ functions and objects like `cout` and `string`. Without this, you’d need to write `std::cout` instead of just `cout`.

#### **Why it’s used:**
- These lines set up the program to use essential tools for input/output and text manipulation. The `using namespace std;` line simplifies the code by avoiding repetitive `std::` prefixes.

---

### **2. Class Definition**
```cpp
Videotape::Videotape(string name_of_tape, string type_of_movie, string released_date, string cost, string availability, string location)
{
    NAME_OF_TAPE = name_of_tape;
    TYPE_OF_MOVIE = type_of_movie;
    RELEASED_DATE = released_date;
    COST = cost;
    AVAILABILITY = availability;
    LOCATION = location;
}
```

#### **What it does:**
- This is the **constructor** for the `Videotape` class. It initializes the attributes of a `Videotape` object when it is created.

#### **Explanation:**
1. **Constructor**: A constructor is a special function that runs automatically when an object is created. It sets up the object’s initial state.
   - Example: When you create a `Videotape` object, the constructor assigns values to its attributes (e.g., `NAME_OF_TAPE`, `TYPE_OF_MOVIE`).

2. **Parameters**: The constructor takes six parameters:
   - `name_of_tape`: The name of the video tape.
   - `type_of_movie`: The genre or type of the movie.
   - `released_date`: The date the movie was released.
   - `cost`: The cost of renting the video tape.
   - `availability`: Whether the tape is available for rent.
   - `location`: The physical location of the tape in the store.

3. **Assignment**: Inside the constructor, the values passed as parameters are assigned to the corresponding member variables of the class.

#### **Why it’s used:**
- The constructor ensures that every `Videotape` object is properly initialized with specific values when it’s created. This avoids having uninitialized or invalid data.

---

### **3. Setter Methods**
```cpp
void Videotape::setNAME_OF_TAPE(string a) { NAME_OF_TAPE = a; }
void Videotape::setTYPE_OF_MOVIE(string b) { TYPE_OF_MOVIE = b; }
void Videotape::setRELEASED_DATE(string c) { RELEASED_DATE = c; }
void Videotape::setCOST(string d) { COST = d; }
void Videotape::setAVAILABILITY(string e) { AVAILABILITY = e; }
void Videotape::setLOCATION(string f) { LOCATION = f; }
```

#### **What it does:**
- These are **setter methods** that allow you to modify the values of the `Videotape` object’s attributes after it has been created.

#### **Explanation:**
1. **Setter Methods**: A setter is a function that updates the value of a private member variable.
   - Example: `setNAME_OF_TAPE("Inception")` changes the `NAME_OF_TAPE` attribute to `"Inception"`.

2. **Parameters**: Each setter takes one parameter (e.g., `a`, `b`, `c`) and assigns it to the corresponding member variable.

3. **Why Private Variables?**: The member variables (e.g., `NAME_OF_TAPE`) are private, meaning they can’t be accessed directly from outside the class. Setters provide a controlled way to modify these variables.

#### **Why it’s used:**
- Setters allow you to change the state of an object after it’s created. They also enforce control over how data is modified, which can prevent errors or invalid states.

---

### **4. Getter Methods**
```cpp
string Videotape::getNAME_OF_TAPE() { return NAME_OF_TAPE; }
string Videotape::getTYPE_OF_MOVIE() { return TYPE_OF_MOVIE; }
string Videotape::getRELEASED_DATE() { return RELEASED_DATE; }
string Videotape::getCOST() { return COST; }
string Videotape::getAVAILABILITY() { return AVAILABILITY; }
string Videotape::getLOCATION() { return LOCATION; }
```

#### **What it does:**
- These are **getter methods** that allow you to retrieve the values of the `Videotape` object’s attributes.

#### **Explanation:**
1. **Getter Methods**: A getter is a function that returns the value of a private member variable.
   - Example: `getNAME_OF_TAPE()` returns the value of `NAME_OF_TAPE`.

2. **Return Type**: Each getter returns a `string`, which is the data type of the member variable.

3. **Why Private Variables?**: Since the member variables are private, getters provide a way to access their values without exposing them directly.

#### **Why it’s used:**
- Getters allow you to retrieve the state of an object without directly accessing its private data. This maintains encapsulation and control over how data is accessed.

---

### **5. Print Method**
```cpp
void Videotape::Print()
{
    cout << "\n\nName: " << NAME_OF_TAPE << endl;
    cout << "Type of movie: " << TYPE_OF_MOVIE << endl;
    cout << "Released date : " << RELEASED_DATE << endl;
    cout << "Cost: " << COST << endl;
    cout << "AVAILABILITY: " << AVAILABILITY << endl;
    cout << "Location: " << LOCATION << "\n\n";
}
```

#### **What it does:**
- This method prints all the attributes of the `Videotape` object to the console in a readable format.

#### **Explanation:**
1. **`cout`**: This is used to print text to the console.
   - Example: `cout << "Name: " << NAME_OF_TAPE` prints the label `"Name: "` followed by the value of `NAME_OF_TAPE`.

2. **`endl`**: This inserts a newline character, moving the cursor to the next line.

3. **Formatting**: The method uses `\n\n` to add extra blank lines before and after the output for better readability.

#### **Why it’s used:**
- The `Print` method provides a convenient way to display all the information about a `Videotape` object in one place. This is useful for debugging or showing data to the user.

---

### **6. Summary of Control Flow**
1. **Object Creation**: When a `Videotape` object is created, the constructor initializes its attributes.
2. **Modification**: Setters can be used to change the attributes.
3. **Retrieval**: Getters can be used to access the attributes.
4. **Display**: The `Print` method outputs all the attributes to the console.

---

### **7. Example Usage**
```cpp
int main() {
    // Create a Videotape object
    Videotape myTape("Inception", "Sci-Fi", "2010-07-16", "$5.99", "Available", "Aisle 3");

    // Display the tape's information
    myTape.Print();

    // Change the availability and display again
    myTape.setAVAILABILITY("Rented");
    myTape.Print();

    return 0;
}
```

#### **Output:**
```
Name: Inception
Type of movie: Sci-Fi
Released date : 2010-07-16
Cost: $5.99
AVAILABILITY: Available
Location: Aisle 3


Name: Inception
Type of movie: Sci-Fi
Released date : 2010-07-16
Cost: $5.99
AVAILABILITY: Rented
Location: Aisle 3
```

---

### **8. Why This Structure?**
- **Encapsulation**: The class encapsulates all the data and methods related to a video tape, making the code modular and easier to manage.
- **Reusability**: The `Videotape` class can be reused in other parts of the program or in other programs.
- **Control**: By using getters and setters, the class maintains control over how its data is accessed and modified.

---

This code is a great example of **object-oriented programming** in action. It’s simple, clear, and demonstrates how to structure a program using classes, constructors, getters, setters, and methods.