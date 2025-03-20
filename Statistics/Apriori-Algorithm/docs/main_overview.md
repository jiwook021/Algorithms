# Code Overview: main.cpp

### Purpose of the Code

This C++ code implements the **Apriori Algorithm**, a classic algorithm used in **association rule mining**. Association rule mining is a technique used in data mining to discover interesting relationships between variables in large datasets. The most common application of this algorithm is in **market basket analysis**, where the goal is to find associations between products that customers frequently buy together.

For example, in a grocery store, the algorithm might discover that customers who buy **bread** are also likely to buy **milk**. This information can be used for product placement, promotions, or inventory management.

### Main Functionality

The code takes a dataset of transactions (where each transaction is a list of items purchased together) and identifies **frequent itemsets** (groups of items that appear together frequently) and **association rules** (rules that describe how items are associated with each other). The algorithm uses two key metrics:
1. **Support**: The number of transactions that contain a particular itemset. Itemsets with support above a specified threshold (`min_support`) are considered frequent.
2. **Confidence**: The likelihood that if one item (or set of items) is purchased, another item will also be purchased. Rules with confidence above a specified threshold (`min_confidence`) are considered strong.

### Algorithms Used

1. **Apriori Algorithm**:
   - The Apriori algorithm works in a **level-wise** manner, starting with single items and progressively building larger itemsets.
   - It uses the **downward closure property**: If an itemset is frequent, all its subsets must also be frequent. This property helps prune the search space and makes the algorithm efficient.

2. **Association Rule Generation**:
   - Once frequent itemsets are identified, the algorithm generates association rules and calculates their confidence.
   - Rules are filtered based on the minimum confidence threshold.

### Overall Structure

The code is structured into several key components:

1. **`generate_candidates` Function**:
   - Generates candidate itemsets of size 2 from frequent single items.
   - This is the first step in building larger itemsets.

2. **`count_occurrences` Function**:
   - Counts how many transactions contain a given itemset.
   - This is used to determine if an itemset is frequent.

3. **`apriori` Function**:
   - The main function that implements the Apriori algorithm.
   - It performs the following steps:
     - Counts occurrences of single items and identifies frequent ones.
     - Generates candidate pairs from frequent items.
     - Counts occurrences of candidate pairs and identifies frequent ones.
     - Generates association rules and filters them by confidence.
     - Outputs the results in a formatted table.

4. **`main` Function**:
   - Defines a dataset of transactions.
   - Sets the minimum support and confidence thresholds.
   - Calls the `apriori` function to run the algorithm and display the results.

### How the Parts Work Together

1. **Input Data**:
   - The dataset of transactions is defined in the `main` function. Each transaction is a vector of strings representing items.

2. **Frequent Itemset Generation**:
   - The `apriori` function first counts the occurrences of single items and identifies frequent ones based on the `min_support` threshold.
   - It then generates candidate pairs of items using the `generate_candidates` function.
   - The `count_occurrences` function is used to count how many transactions contain each candidate pair, and frequent pairs are identified.

3. **Association Rule Generation**:
   - For each frequent pair, the algorithm generates two association rules (e.g., `item1 -> item2` and `item2 -> item1`).
   - The confidence of each rule is calculated, and rules that meet the `min_confidence` threshold are displayed in a formatted table.

4. **Output**:
   - The results are displayed in a table showing the antecedent (left-hand side of the rule), consequent (right-hand side of the rule), and confidence.
   - Additional statistics, such as the total number of transactions, unique items, frequent items, and frequent pairs, are also displayed.

### Problem Being Solved

The code solves the problem of **discovering associations between items** in a transactional dataset. This is particularly useful in retail and e-commerce, where understanding customer purchasing behavior can lead to better decision-making.

### Approach Taken

The approach taken is **iterative and level-wise**:
1. Start with single items and identify frequent ones.
2. Build candidate pairs from frequent items and identify frequent pairs.
3. Generate association rules from frequent pairs and filter them based on confidence.
4. Output the results in a user-friendly format.

This approach ensures that the algorithm is efficient and scalable, even for large datasets.

### Summary

In summary, this code implements the Apriori algorithm to discover frequent itemsets and association rules in a transactional dataset. It uses support and confidence thresholds to filter results and outputs the findings in a structured and readable format. The code is modular, with separate functions for generating candidates, counting occurrences, and running the main algorithm, making it easy to understand and extend.