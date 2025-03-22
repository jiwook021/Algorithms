# Suggested Improvements: Main.cpp

Here’s a detailed analysis of potential improvements to the code, categorized by **performance**, **readability**, **maintainability**, **error handling**, and **best practices**. Each suggestion includes **why it’s an improvement** and **how to implement it**.

---

### **1. Performance Improvements**

#### **a. Avoid Unnecessary Object Creation**
- **Problem**: In the "Add a Video" section, a default `Videotape` object is created with empty values, and then its properties are updated using setters. This is inefficient.
- **Improvement**: Create the `Videotape` object only after collecting all the necessary data.
- **Why**: Reduces unnecessary object creation and improves performance.
- **How**:
  ```cpp
  if (changeoptions == 1)
  {
      cin.ignore();

      cout << "\nName of the tape?  ";
      getline(cin, temp_name_of_tape);

      cout << "Type of the movie?  ";
      getline(cin, temp_type_of_movie);

      cout << "Released date of the movie?  ";
      getline(cin, temp_released_date);

      cout << "Cost?  ";
      getline(cin, temp_cost);

      cout << "Is the video available? (Write Available or Not available)  ";
      getline(cin, temp_availability);

      cout << "Location?  ";
      getline(cin, temp_location);

      // Create the object after collecting all data
      list.push_back(Videotape(temp_name_of_tape, temp_type_of_movie, temp_released_date, temp_cost, temp_availability, temp_location));
      i = i + 1;
      changeoptions = 0;
  }
  ```

---

### **2. Readability Improvements**

#### **a. Use Meaningful Variable Names**
- **Problem**: Variables like `i` and `j` are not descriptive.
- **Improvement**: Use meaningful names like `videoCount` and `index`.
- **Why**: Makes the code easier to understand.
- **How**:
  ```cpp
  int videoCount = 0; // Instead of int i = 0;

  for (int index = 0; index < videoCount; index++) // Instead of for (int j = 0; j < i; j++)
  {
      list.at(index).Print();
  }
  ```

#### **b. Add Comments and Documentation**
- **Problem**: The code lacks comments explaining its purpose and logic.
- **Improvement**: Add comments to describe the purpose of each section.
- **Why**: Helps other developers (or your future self) understand the code.
- **How**:
  ```cpp
  // Main menu loop: Allows the user to add videos or view the list
  while (true)
  {
      cout << "\n______________________Options______________________\n 1.Add a video informations to save on the system \n 2.Show the current list of video files\nWrite 1 or 2  ";
      cin >> changeoptions;
  ```

---

### **3. Maintainability Improvements**

#### **a. Encapsulate Video Input Logic**
- **Problem**: The code for collecting video details is repeated and tightly coupled with the main function.
- **Improvement**: Move this logic into a separate function.
- **Why**: Makes the code modular and easier to maintain.
- **How**:
  ```cpp
  Videotape GetVideoDetailsFromUser()
  {
      string name, type, date, cost, availability, location;

      cout << "\nName of the tape?  ";
      getline(cin, name);

      cout << "Type of the movie?  ";
      getline(cin, type);

      cout << "Released date of the movie?  ";
      getline(cin, date);

      cout << "Cost?  ";
      getline(cin, cost);

      cout << "Is the video available? (Write Available or Not available)  ";
      getline(cin, availability);

      cout << "Location?  ";
      getline(cin, location);

      return Videotape(name, type, date, cost, availability, location);
  }

  // In main function:
  if (changeoptions == 1)
  {
      cin.ignore();
      list.push_back(GetVideoDetailsFromUser());
      videoCount++;
      changeoptions = 0;
  }
  ```

#### **b. Use Constants for Magic Numbers**
- **Problem**: The menu options (`1` and `2`) are hardcoded, making the code less flexible.
- **Improvement**: Define constants for menu options.
- **Why**: Makes the code easier to update and less error-prone.
- **How**:
  ```cpp
  const int ADD_VIDEO_OPTION = 1;
  const int SHOW_LIST_OPTION = 2;

  if (changeoptions == ADD_VIDEO_OPTION)
  {
      // Add video logic
  }
  else if (changeoptions == SHOW_LIST_OPTION)
  {
      // Show list logic
  }
  ```

---

### **4. Error Handling**

#### **a. Validate User Input**
- **Problem**: The program doesn’t validate user input, which can lead to crashes or incorrect behavior.
- **Improvement**: Add input validation for menu choices and video details.
- **Why**: Prevents crashes and ensures the program behaves as expected.
- **How**:
  ```cpp
  // Validate menu choice
  while (true)
  {
      cout << "\n______________________Options______________________\n 1.Add a video informations to save on the system \n 2.Show the current list of video files\nWrite 1 or 2  ";
      if (cin >> changeoptions && (changeoptions == 1 || changeoptions == 2))
      {
          break;
      }
      else
      {
          cin.clear(); // Clear error flag
          cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Discard invalid input
          cout << "Invalid choice. Please enter 1 or 2.\n";
      }
  }
  ```

#### **b. Handle Empty Input**
- **Problem**: The program doesn’t check if the user enters empty strings for video details.
- **Improvement**: Add checks to ensure all fields are filled.
- **Why**: Ensures data integrity.
- **How**:
  ```cpp
  string GetNonEmptyInput(const string& prompt)
  {
      string input;
      while (true)
      {
          cout << prompt;
          getline(cin, input);
          if (!input.empty())
          {
              return input;
          }
          cout << "This field cannot be empty. Please try again.\n";
      }
  }

  // In GetVideoDetailsFromUser:
  string name = GetNonEmptyInput("\nName of the tape?  ");
  ```

---

### **5. Best Practices**

#### **a. Avoid Global Variables**
- **Problem**: Global variables like `changeoptions` and `videolist` make the code harder to debug and maintain.
- **Improvement**: Pass variables as function parameters or use local variables.
- **Why**: Reduces side effects and improves modularity.
- **How**:
  ```cpp
  int main()
  {
      int changeoptions = 0; // Local variable
      bool videolist = false; // Local variable
      // Rest of the code...
  }
  ```

#### **b. Use `const` for Unchanging Variables**
- **Problem**: Variables like `temp_name_of_tape` are not marked as `const` when they should be.
- **Improvement**: Use `const` for variables that don’t change after initialization.
- **Why**: Prevents accidental modification and improves clarity.
- **How**:
  ```cpp
  const string name = GetNonEmptyInput("\nName of the tape?  ");
  ```

#### **c. Use Range-Based For Loops**
- **Problem**: The loop for displaying videos uses an index-based approach.
- **Improvement**: Use a range-based `for` loop for cleaner code.
- **Why**: Simplifies iteration and reduces the chance of off-by-one errors.
- **How**:
  ```cpp
  for (const auto& video : list)
  {
      video.Print();
  }
  ```

---

### **6. Potential Bugs**

#### **a. Uninitialized Variables**
- **Problem**: `videolist` is declared but never initialized or used.
- **Improvement**: Remove unused variables.
- **Why**: Reduces clutter and potential confusion.
- **How**:
  ```cpp
  // Remove this line:
  bool videolist;
  ```

#### **b. Memory Leaks**
- **Problem**: If the `Videotape` class dynamically allocates memory, the program might leak memory.
- **Improvement**: Ensure proper memory management in the `Videotape` class.
- **Why**: Prevents memory leaks and crashes.
- **How**:
  - If `Videotape` uses dynamic memory, implement a destructor, copy constructor, and assignment operator (Rule of Three).

---

### **Final Improved Code Example**

Here’s a snippet of the improved code:

```cpp
int main()
{
    vector<Videotape> list;
    int videoCount = 0;
    int changeoptions = 0;

    // Preload videos
    list.push_back(Videotape("Good_will_hunting", "Drama", "10/02/1997", "$10", "Available", "F-17"));
    list.at(videoCount++).Print();

    list.push_back(Videotape("Parasite", "Horror", "10/13/2003", "$8", "Not available", "A-19"));
    list.at(videoCount++).Print();

    while (true)
    {
        cout << "\n______________________Options______________________\n 1.Add a video informations to save on the system \n 2.Show the current list of video files\nWrite 1 or 2  ";
        if (cin >> changeoptions && (changeoptions == 1 || changeoptions == 2))
        {
            cin.ignore();
            if (changeoptions == 1)
            {
                list.push_back(GetVideoDetailsFromUser());
                videoCount++;
            }
            else if (changeoptions == 2)
            {
                cout << "\n\n_______LIST OF VIDEOS_______" << endl;
                for (const auto& video : list)
                {
                    video.Print();
                }
            }
            changeoptions = 0;
        }
        else
        {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid choice. Please enter 1 or 2.\n";
        }
    }

    return 0;
}
```

---

These improvements make the code **more efficient**, **easier to read**, **more maintainable**, and **less prone to errors**. Let me know if you’d like further clarification!