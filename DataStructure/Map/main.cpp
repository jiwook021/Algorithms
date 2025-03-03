/**
 * @file main.cpp
 * @brief Interactive driver for BSTMap (Binary Search Tree Map).
 *
 * Simulates an employee directory to show how a BST keeps
 * entries sorted by key, supports fast lookups, and enables
 * range queries.  Then lets the user experiment interactively.
 */

#include "BSTMap.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=============================================\n"
              << "  BST Map (Binary Search Tree)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  An ordered map stores key-value pairs sorted by key.\n"
              << "  Unlike a hash map, a BST map keeps entries in order,\n"
              << "  letting you iterate from smallest to largest key or\n"
              << "  search for ranges of keys efficiently.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Compare the key with the current node.\n"
              << "  2. If smaller, go left.  If larger, go right.\n"
              << "  3. If equal, found it (or it's a duplicate).\n"
              << "  4. Empty spot?  Insert a new node here.\n"
              << "  5. In-order traversal (left, node, right) visits\n"
              << "     all keys in ascending order.\n\n"

              << "KEY TERMS\n"
              << "  Root       : The top node of the tree.\n"
              << "  Left child : Has a smaller key than its parent.\n"
              << "  Right child: Has a larger key than its parent.\n"
              << "  In-order   : Visit left subtree, then node, then right.\n"
              << "  Successor  : Next larger key (used during deletion).\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Employee Directory
    //
    // A company tracks employees by ID number.  The BST map keeps
    // them sorted, making it easy to:
    //   - Look up any employee by ID in O(log n)
    //   - List all employees in ID order
    //   - Find employees in an ID range
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Employee Directory\n\n"
              << "  A company tracks employees by ID.  The BST map\n"
              << "  keeps them sorted automatically, so listing and\n"
              << "  range queries are natural.\n\n";

    BSTMap<int, std::string> employees;

    struct Employee {
        int id;
        std::string name;
    };
    std::vector<Employee> staff = {
        {1042, "Alice"},  {1007, "Bob"},   {1089, "Carol"},
        {1023, "Dave"},   {1056, "Eve"},   {1001, "Frank"},
        {1078, "Grace"},
    };

    std::cout << "  Adding employees (inserted out of order):\n\n";

    for (std::size_t i = 0; i < staff.size(); ++i) {
        const auto& e = staff[i];
        bool is_new = employees.insert(e.id, e.name);
        std::cout << "    insert(" << e.id << ", \""
                  << std::left << std::setw(6) << e.name << "\")"
                  << std::right
                  << "  -> " << (is_new ? "new   " : "update")
                  << "  (size: " << employees.size() << ")\n";
    }

    // -----------------------------------------------------------------
    // In-order traversal — the BST's defining property
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  In-Order Traversal (Sorted by ID)\n"
              << "---------------------------------------------\n\n"
              << "  Even though we inserted employees out of order,\n"
              << "  the BST keeps them sorted.  Iterating visits\n"
              << "  keys in ascending order:\n\n";

    for (const auto& pair : employees) {
        std::cout << "    ID " << pair.first << ": " << pair.second << "\n";
    }

    std::cout << "\n  " << employees.size() << " employees, sorted by ID.\n";

    // -----------------------------------------------------------------
    // Lookup, update, and delete
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Lookup, Update, and Delete\n"
              << "---------------------------------------------\n\n";

    // Lookup
    auto val = employees.find(1042);
    std::cout << "  find(1042)      -> "
              << (val ? val.value() : "not found") << "\n";
    auto miss = employees.find(9999);
    std::cout << "  find(9999)      -> "
              << (miss ? miss.value() : "not found") << "\n";
    std::cout << "  Contains(1007)  -> "
              << (employees.Contains(1007) ? "true" : "false") << "\n\n";

    // Update via insert (overwrites existing value)
    std::cout << "  Updating Alice's record:\n"
              << "    Before: find(1042) = "
              << employees.find(1042).value() << "\n";
    employees.insert(1042, "Alice Smith");
    std::cout << "    After:  find(1042) = "
              << employees.find(1042).value() << "\n\n";

    // Delete
    std::cout << "  Deleting Bob (ID 1007):\n"
              << "    remove(1007)  -> "
              << (employees.remove(1007) ? "removed" : "not found") << "\n"
              << "    Contains(1007) -> "
              << (employees.Contains(1007) ? "true" : "false") << "\n"
              << "    Size: " << employees.size() << "\n";

    // -----------------------------------------------------------------
    // Range queries — BST advantage over hash tables
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Range Queries (BST Advantage)\n"
              << "---------------------------------------------\n\n"
              << "  Unlike hash tables, BST maps support efficient\n"
              << "  range queries using lower_bound / upper_bound.\n\n";

    auto lo = employees.lower_bound(1020);
    auto hi = employees.upper_bound(1060);

    std::cout << "  lower_bound(1020) -> first ID >= 1020: ID "
              << lo->first << " (" << lo->second << ")\n";
    std::cout << "  upper_bound(1060) -> first ID >  1060: ID "
              << hi->first << " (" << hi->second << ")\n\n"
              << "  Employees with IDs in [1020, 1060]:\n\n";

    for (auto it = lo; it != hi; ++it) {
        std::cout << "    ID " << it->first << ": " << it->second << "\n";
    }

    std::cout << "\n  Hash tables would need to scan every element\n"
              << "  for a range query.  BSTs do it in O(log n + k)\n"
              << "  where k is the number of results.\n";

    // -----------------------------------------------------------------
    // Deletion cases — leaf, one child, two children
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Deletion Cases\n"
              << "---------------------------------------------\n\n"
              << "  BST deletion has three cases:\n"
              << "    1. Leaf node        : just remove it.\n"
              << "    2. One child        : replace node with its child.\n"
              << "    3. Two children     : replace with in-order successor,\n"
              << "                          then delete the successor.\n\n";

    {
        BSTMap<int, std::string> demo;
        demo.insert(50, "root");
        demo.insert(30, "left");
        demo.insert(70, "right");
        demo.insert(20, "leaf");
        demo.insert(40, "leaf");
        demo.insert(60, "leaf");
        demo.insert(80, "leaf");

        std::cout << "  Tree: 50(root), 30, 70, 20, 40, 60, 80\n\n";

        // Delete leaf
        demo.remove(20);
        std::cout << "    remove(20) - leaf node, just removed.\n";

        // Delete node with one child
        demo.remove(30);
        std::cout << "    remove(30) - one child (40), replaced by child.\n";

        // Delete node with two children
        demo.remove(50);
        std::cout << "    remove(50) - two children, replaced by successor.\n\n";

        std::cout << "  Remaining (in order): ";
        for (const auto& pair : demo) {
            std::cout << pair.first << " ";
        }
        std::cout << " (" << demo.size() << " nodes)\n";
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (using the employee directory)\n"
              << "  Commands:\n"
              << "    put ID NAME       -- add/update an employee\n"
              << "    get ID            -- look up by ID\n"
              << "    del ID            -- remove an employee\n"
              << "    range LOW HIGH    -- list employees in ID range\n"
              << "    show              -- list all employees\n"
              << "    q                 -- quit\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string cmd;
        std::getline(std::cin, cmd);

        if (cmd.empty() || cmd[0] == 'q' || cmd[0] == 'Q') break;

        if (cmd.rfind("put", 0) == 0) {
            try {
                std::size_t pos = 3;
                while (pos < cmd.size() && cmd[pos] == ' ') ++pos;
                int id = std::stoi(cmd.substr(pos), &pos);
                std::size_t name_start = cmd.find_first_not_of(' ', 3 + pos);
                if (name_start == std::string::npos) {
                    throw std::runtime_error("missing name");
                }
                std::string name = cmd.substr(name_start);

                bool existed = employees.Contains(id);
                if (existed) {
                    employees.insert(id, name);
                    std::cout << "    Updated: ID " << id << " -> "
                              << name << "\n";
                } else {
                    employees.insert(id, name);
                    std::cout << "    Added: ID " << id << " -> "
                              << name << "\n";
                }
                std::cout << "    Size: " << employees.size() << "\n";
            } catch (...) {
                std::cout << "    Invalid. Usage: put ID NAME"
                          << "  (e.g. put 2000 Zoe)\n";
            }
        } else if (cmd.rfind("get", 0) == 0) {
            try {
                int id = std::stoi(cmd.substr(3));
                auto result = employees.find(id);
                if (result) {
                    std::cout << "    ID " << id << " -> "
                              << result.value() << "\n";
                } else {
                    std::cout << "    Not found.\n";
                }
            } catch (...) {
                std::cout << "    Invalid. Usage: get ID"
                          << "  (e.g. get 1042)\n";
            }
        } else if (cmd.rfind("del", 0) == 0) {
            try {
                int id = std::stoi(cmd.substr(3));
                if (employees.remove(id)) {
                    std::cout << "    Deleted: ID " << id << "\n";
                } else {
                    std::cout << "    Not found.\n";
                }
            } catch (...) {
                std::cout << "    Invalid. Usage: del ID"
                          << "  (e.g. del 1007)\n";
            }
        } else if (cmd.rfind("range", 0) == 0) {
            try {
                std::size_t pos = 5;
                int low = std::stoi(cmd.substr(pos), &pos);
                int high = std::stoi(cmd.substr(5 + pos));
                auto lo_it = employees.lower_bound(low);
                auto hi_it = employees.upper_bound(high);
                bool found = false;
                for (auto it = lo_it; it != hi_it; ++it) {
                    std::cout << "    ID " << it->first << ": "
                              << it->second << "\n";
                    found = true;
                }
                if (!found) {
                    std::cout << "    No employees in range ["
                              << low << ", " << high << "].\n";
                }
            } catch (...) {
                std::cout << "    Invalid. Usage: range LOW HIGH"
                          << "  (e.g. range 1020 1060)\n";
            }
        } else if (cmd.rfind("show", 0) == 0) {
            if (employees.empty()) {
                std::cout << "    (empty)\n";
            } else {
                for (const auto& pair : employees) {
                    std::cout << "    ID " << pair.first << ": "
                              << pair.second << "\n";
                }
                std::cout << "    (" << employees.size()
                          << " employees)\n";
            }
        } else {
            std::cout << "    Unknown. Try: put, get, del, range, show, q\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
