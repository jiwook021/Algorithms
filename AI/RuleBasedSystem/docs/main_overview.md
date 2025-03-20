# Code Overview: main.cpp

This code implements a **rule-based system** using a **forward-chaining inference engine**, which is a common pattern in artificial intelligence and expert systems. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code is designed to create a system that can reason about facts and apply rules to derive new facts. This is often used in expert systems, decision-making systems, and diagnostic tools. For example:
- **Medical diagnosis**: Rules like "If the patient has a fever and a cough, then they might have the flu" can be applied to a set of patient symptoms (facts) to infer possible diagnoses.
- **Business rules**: Rules like "If the customer's order exceeds $100, apply a 10% discount" can be applied to customer data (facts) to determine actions.

The system works by:
1. Storing **facts** (pieces of information, e.g., "temperature = 102").
2. Applying **rules** (conditions and actions, e.g., "If temperature > 100, then fever = true").
3. Deriving new facts based on the rules and existing facts.

---

### **Main Functionality**
The code implements the following key components:
1. **Fact**: Represents a piece of information with a name and value (e.g., "temperature = 102").
2. **FactBase**: A thread-safe collection of facts (working memory) that supports adding, removing, and querying facts.
3. **Rule**: Represents a condition-action pair (e.g., "If temperature > 100, then fever = true").
4. **RuleBase**: A collection of rules that can be evaluated against the facts.
5. **InferenceEngine**: The core logic that applies rules to facts to derive new facts (forward chaining).

---

### **Algorithms Used**
1. **Forward Chaining**:
   - The inference engine evaluates rules against the facts in the FactBase.
   - If a rule's conditions are satisfied, it derives new facts and adds them to the FactBase.
   - This process repeats until no new facts can be derived.

2. **Thread-Safe Data Structures**:
   - The FactBase uses a `std::mutex` to ensure thread safety when adding, removing, or querying facts.
   - This allows the system to be used in multi-threaded environments.

3. **Variant Types**:
   - The `Fact` class uses `std::variant` to support multiple data types (e.g., `bool`, `int`, `double`, `std::string`).
   - This makes the system flexible and capable of handling different types of facts.

---

### **Overall Structure**
The code is organized into several classes and components:

1. **Fact**:
   - Represents a single piece of information.
   - Stores a name (e.g., "temperature") and a value (e.g., 102).
   - Provides methods to get/set the value and convert the fact to a string.

2. **FactBase**:
   - A thread-safe collection of facts stored in an `std::unordered_map`.
   - Provides methods to add, remove, and query facts.

3. **Rule** (not fully shown in the code snippet):
   - Represents a condition-action pair.
   - Conditions are typically logical expressions (e.g., "temperature > 100").
   - Actions derive new facts (e.g., "fever = true").

4. **RuleBase** (not fully shown in the code snippet):
   - A collection of rules that can be evaluated against the FactBase.

5. **InferenceEngine** (not fully shown in the code snippet):
   - The core logic that applies rules to facts.
   - Uses forward chaining to derive new facts iteratively.

6. **Main Function**:
   - Runs tests to demonstrate the system's functionality.
   - Handles exceptions and errors gracefully.

---

### **How the Parts Work Together**
1. **Initialization**:
   - Facts are added to the FactBase (e.g., "temperature = 102").
   - Rules are added to the RuleBase (e.g., "If temperature > 100, then fever = true").

2. **Rule Evaluation**:
   - The InferenceEngine evaluates each rule against the facts in the FactBase.
   - If a rule's conditions are satisfied, it derives new facts and adds them to the FactBase.

3. **Iteration**:
   - The process repeats until no new facts can be derived.

4. **Output**:
   - The final set of facts represents the system's conclusions (e.g., "fever = true").

---

### **Problem Being Solved**
The code solves the problem of **automated reasoning**:
- Given a set of initial facts and rules, the system can derive new facts and make decisions.
- This is useful in domains like medical diagnosis, business rule engines, and expert systems.

---

### **Approach Taken**
1. **Modular Design**:
   - The system is divided into clear components (Fact, FactBase, Rule, RuleBase, InferenceEngine).
   - This makes the code maintainable and extensible.

2. **Thread Safety**:
   - The FactBase uses a mutex to ensure thread safety, making the system suitable for concurrent environments.

3. **Flexibility**:
   - The use of `std::variant` allows the system to handle different types of facts.
   - The rule-based approach makes it easy to add or modify rules without changing the core logic.

---

### **Example Use Case**
Imagine a medical diagnosis system:
1. **Facts**:
   - "temperature = 102"
   - "cough = true"

2. **Rules**:
   - "If temperature > 100, then fever = true"
   - "If fever = true and cough = true, then diagnosis = flu"

3. **Process**:
   - The InferenceEngine evaluates the rules and derives:
     - "fever = true" (from the first rule).
     - "diagnosis = flu" (from the second rule).

4. **Output**:
   - The system concludes that the patient has the flu.

---

### **Summary**
This code implements a rule-based system using a forward-chaining inference engine. It is designed to reason about facts and derive new facts based on rules. The system is modular, thread-safe, and flexible, making it suitable for a wide range of applications, including expert systems and decision-making tools.