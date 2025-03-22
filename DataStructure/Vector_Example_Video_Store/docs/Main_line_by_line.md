# Step-by-Step Explanation: Main.cpp

Let’s break down the code **step by step** in extreme detail, explaining every significant section, concept, and decision. I’ll use simple language, examples, and diagrams to make everything clear.

---

### **1. Includes and Global Variables**

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "Video.h"

using namespace std;

int changeoptions;
bool videolist;

string temp_name_of_tape, temp_type_of_movie, temp_cost, temp_released_date, temp_location,  temp_availability;
```

#### **What It Does**
- This section includes necessary libraries and declares global variables.

#### **Explanation**
1. **`#include` Statements**:
   - These bring in external code libraries:
     - `<iostream>`: For input/output (e.g., `cin` and `cout`).
     - `<string>`: For working with text (e.g., `string` type).
     - `<vector>`: For using the `vector` data structure (a dynamic array).
     - `<cstdlib>`: For general utilities (not used in this code but often included for functions like `exit()`).
     - `"Video.h"`: A custom header file that defines the `Videotape` class.

2. **`using namespace std;`**:
   - This allows us to use standard library names (like `cout` and `vector`) without typing `std::` every time.

3. **Global Variables**:
   - These are variables that can be accessed anywhere in the program:
     - `changeoptions`: Stores the user’s menu choice (1 or 2).
     - `videolist`: A boolean flag (not used in this code but could be for future features).
     - `temp_name_of_tape`, `temp_type_of_movie`, etc.: Temporary storage for user input when adding a new video.

#### **Why This Approach?**
- **Includes**: These libraries provide essential tools for the program to work.
- **Global Variables**: While generally discouraged (they can make code harder to debug), they are used here for simplicity in a small program.

---

### **2. Main Function**

```cpp
int main()
{
    vector<Videotape> list; // Vector to store video tapes
    int i = 0; // Counter for the number of videos
```

#### **What It Does**
- The `main` function is the entry point of the program. It initializes a vector to store video tapes and a counter to track how many videos are in the system.

#### **Explanation**
1. **`vector<Videotape> list;`**:
   - A **vector** is a dynamic array that can grow in size. Here, it stores objects of the `Videotape` class.
   - **Dynamic Array**: Unlike a regular array, a vector doesn’t need a fixed size. It can expand as you add more items.

2. **`int i = 0;`**:
   - This counter keeps track of how many videos are in the vector. It starts at 0 because the vector is initially empty.

#### **Why This Approach?**
- **Vector**: Using a vector makes the program flexible. It can handle any number of videos without needing to predefine a size.
- **Counter (`i`)**: This helps track the current number of videos and is used to access specific elements in the vector.

---

### **3. Preloading Video Tapes**

```cpp
    cout << "_______LIST OF VIDEOS_______" << endl;

    list.push_back(Videotape("Good_will_hunting", "Drama", "10/02/1997", "$10","Available" , "F-17"));
    list.at(i).Print();
    i = i + 1;

    list.push_back(Videotape("Parasite", "Horror",  "10/13/2003", "$8" ,"Not available", "A-19"));
    list.at(i).Print();
    i = i + 1;
```

#### **What It Does**
- This section adds two preloaded video tapes to the vector and displays their details.

#### **Explanation**
1. **`list.push_back(...)`**:
   - Adds a new `Videotape` object to the end of the vector.
   - The `Videotape` constructor is called with the video’s details (name, type, release date, etc.).

2. **`list.at(i).Print();`**:
   - Calls the `Print` method of the `Videotape` object at index `i` in the vector.
   - This displays the video’s details.

3. **`i = i + 1;`**:
   - Increments the counter to reflect the addition of a new video.

#### **Why This Approach?**
- **Preloading**: Demonstrates how the system works by adding sample data.
- **`Print` Method**: Encapsulates the display logic within the `Videotape` class, making the code modular.

---

### **4. Main Menu Loop**

```cpp
    while (true)
    {
        cout << "\n______________________Options______________________\n 1.Add a video informations to save on the system \n 2.Show the current list of video files\nWrite 1 or 2  ";
        cin >> changeoptions;
```

#### **What It Does**
- Displays a menu and waits for the user to choose an option.

#### **Explanation**
1. **`while (true)`**:
   - Creates an infinite loop, meaning the program will keep running until manually stopped.

2. **Menu Display**:
   - The `cout` statement prints the menu options:
     - Option 1: Add a video.
     - Option 2: Show the list of videos.

3. **`cin >> changeoptions;`**:
   - Reads the user’s choice and stores it in `changeoptions`.

#### **Why This Approach?**
- **Infinite Loop**: Keeps the program running so the user can perform multiple actions.
- **Menu-Driven Interface**: Makes the program interactive and user-friendly.

---

### **5. Option 1: Add a Video**

```cpp
        if (changeoptions == 1)
        {
            cin.ignore();
            list.push_back(Videotape("", "", "", "", "", ""));

            cout << "\nName of the tape?  ";
            getline(cin, temp_name_of_tape);

            // Repeat for other fields...

            list.at(i).setNAME_OF_TAPE(temp_name_of_tape);
            // Repeat for other fields...

            i = i + 1;
            changeoptions = 0;
        }
```

#### **What It Does**
- Allows the user to add a new video tape to the system.

#### **Explanation**
1. **`cin.ignore();`**:
   - Clears the input buffer to avoid issues with `getline`.

2. **`list.push_back(Videotape("", "", "", "", "", ""));`**:
   - Adds a new, empty `Videotape` object to the vector.

3. **`getline(cin, temp_name_of_tape);`**:
   - Reads the user’s input for the video’s name and stores it in `temp_name_of_tape`.

4. **`list.at(i).setNAME_OF_TAPE(temp_name_of_tape);`**:
   - Updates the `Videotape` object’s properties using setter methods.

5. **`i = i + 1;`**:
   - Increments the counter to reflect the new video.

6. **`changeoptions = 0;`**:
   - Resets the menu choice to allow the user to make another selection.

#### **Why This Approach?**
- **Temporary Variables**: Store user input before updating the `Videotape` object.
- **Setter Methods**: Encapsulate the logic for updating object properties.

---

### **6. Option 2: Show the List**

```cpp
        else if (changeoptions == 2)
        {
            cout << "\n\n_______LIST OF VIDEOS_______" << endl;

            for (int j = 0; j < i; j++)
            {
                list.at(j).Print();
            }
            changeoptions = 0;
        }
```

#### **What It Does**
- Displays the details of all video tapes in the system.

#### **Explanation**
1. **`for (int j = 0; j < i; j++)`**:
   - Loops through the vector from index 0 to `i-1` (the number of videos).

2. **`list.at(j).Print();`**:
   - Calls the `Print` method for each `Videotape` object to display its details.

3. **`changeoptions = 0;`**:
   - Resets the menu choice.

#### **Why This Approach?**
- **Loop Through Vector**: Efficiently displays all videos.
- **`Print` Method**: Keeps the display logic within the `Videotape` class.

---

### **7. End of Program**

```cpp
    return 0;
}
```

#### **What It Does**
- Ends the program.

#### **Explanation**
- **`return 0;`**: Indicates that the program executed successfully.

---

### **Text-Based Diagram of Program Flow**

```
Start
  |
  v
Initialize Vector and Counter
  |
  v
Preload Two Videos
  |
  v
Display Menu
  |
  v
User Chooses Option
  |
  v
If Option 1: Add Video
  |       |
  |       v
  |     Prompt for Details
  |       |
  |       v
  |     Update Vector
  |       |
  |       v
  |     Reset Menu Choice
  |
  v
If Option 2: Show List
  |       |
  |       v
  |     Loop Through Vector
  |       |
  |       v
  |     Print Video Details
  |       |
  |       v
  |     Reset Menu Choice
  |
  v
Repeat Menu Loop
```

---

This breakdown should make the code **completely understandable**, even for beginners! Let me know if you’d like further clarification on any part.