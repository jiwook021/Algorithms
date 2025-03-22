# Code Overview: Main.cpp

### Purpose of the Code

This C++ program is designed to manage a **video rental system** or a **video library**. It allows users to:
1. **Store information about video tapes** (movies) in a collection.
2. **Add new video tapes** to the system by entering their details.
3. **Display the current list of video tapes** stored in the system.

The program uses a **vector** to dynamically store and manage the video tape entries, making it flexible to handle any number of videos. It also provides a **menu-driven interface** where users can choose between adding a new video or viewing the existing list.

---

### Main Functionality

1. **Storing Video Information**:
   - Each video tape is represented as an object of the `Videotape` class (defined in `Video.h`).
   - The program stores details such as:
     - Name of the tape
     - Type of movie (e.g., Drama, Horror)
     - Release date
     - Cost
     - Availability (Available or Not available)
     - Location (e.g., shelf location in the library)

2. **Adding New Videos**:
   - The user can input details for a new video tape, which is then added to the vector.

3. **Displaying the Video List**:
   - The program can display all the video tapes currently stored in the system.

4. **Menu-Driven Interface**:
   - The program runs in a loop, allowing the user to repeatedly choose between adding a video or viewing the list.

---

### Algorithms Used

1. **Vector Data Structure**:
   - The program uses a `vector<Videotape>` to store the video tape objects. Vectors are dynamic arrays that can grow in size as needed, making them ideal for this use case.

2. **Menu Loop**:
   - A `while (true)` loop keeps the program running indefinitely, displaying a menu and processing user input.

3. **Input Handling**:
   - The program uses `cin` and `getline` to read user input for video details.

4. **Object-Oriented Programming (OOP)**:
   - The `Videotape` class encapsulates the properties and methods related to a video tape. The program uses **setter methods** to update the properties of a video tape object.

---

### Overall Structure

The code is structured as follows:

1. **Includes and Global Variables**:
   - The program includes necessary headers (`<iostream>`, `<string>`, `<vector>`, `<cstdlib>`) and the `Video.h` file.
   - Global variables are declared for temporary storage of user input.

2. **Main Function**:
   - A vector `list` is created to store `Videotape` objects.
   - Two video tapes are preloaded into the vector for demonstration purposes.
   - A `while (true)` loop provides the main menu and handles user input.

3. **Menu Options**:
   - **Option 1**: Add a new video tape.
     - The user is prompted to enter details for the video tape.
     - A new `Videotape` object is created and added to the vector.
   - **Option 2**: Display the current list of video tapes.
     - The program iterates through the vector and prints the details of each video tape.

4. **Loop Continuation**:
   - After processing a menu option, the program resets `changeoptions` to 0 and continues the loop.

---

### How the Parts Work Together

1. **Initialization**:
   - The program starts by creating a vector and preloading two video tapes. These are displayed immediately to show the system is working.

2. **User Interaction**:
   - The user is presented with a menu to either add a video or view the list.
   - If the user chooses to add a video, the program collects the details, creates a `Videotape` object, and adds it to the vector.
   - If the user chooses to view the list, the program iterates through the vector and prints the details of each video tape.

3. **Dynamic Storage**:
   - The vector dynamically grows as new video tapes are added, ensuring the system can handle any number of entries.

4. **Looping**:
   - The program runs indefinitely, allowing the user to perform multiple actions without restarting.

---

### Problem Being Solved

The program addresses the need for a **simple video management system**. It solves the problem of:
- **Storing and organizing video tape information** in a structured way.
- **Providing an easy-to-use interface** for adding and viewing video details.
- **Scaling dynamically** to handle an increasing number of video tapes.

---

### Approach Taken

1. **Object-Oriented Design**:
   - The `Videotape` class encapsulates all the properties and behaviors of a video tape, making the code modular and reusable.

2. **Dynamic Data Structure**:
   - The use of a vector ensures the system can handle an arbitrary number of video tapes without requiring manual memory management.

3. **User-Friendly Interface**:
   - The menu-driven approach makes the program intuitive and easy to use, even for non-technical users.

4. **Input Validation**:
   - While not explicitly implemented in this code, the structure allows for easy addition of input validation (e.g., checking for valid dates or costs).

---

### Summary

This program is a **simple yet effective video management system** that demonstrates the use of:
- **Vectors** for dynamic storage.
- **Object-Oriented Programming** for modularity.
- **Menu-Driven Interfaces** for user interaction.

It provides a foundation that can be extended with additional features like input validation, file storage, or search functionality.