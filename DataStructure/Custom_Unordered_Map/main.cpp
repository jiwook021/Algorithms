/**
 * @file main.cpp
 * @brief Interactive driver for custom UnorderedMap (hash table).
 *
 * Simulates building a contact list to show how hashing, collisions,
 * and rehashing work, then lets the user experiment interactively.
 */

#include "CustomUnorderedMap.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Show which buckets hold which contacts.
 */
static void PrintBuckets(const UnorderedMap<std::string, std::string>& map) {
    // Group elements by bucket for display
    std::vector<std::vector<std::pair<std::string, std::string>>>
        groups(map.BucketCount());

    for (auto it = map.begin(); it != map.end(); ++it) {
        auto [key, value] = *it;
        groups[map.Bucket(key)].push_back({key, value});
    }

    std::size_t occupied = 0;
    for (std::size_t b = 0; b < map.BucketCount(); ++b) {
        if (groups[b].empty()) continue;
        ++occupied;
        std::cout << "      Bucket " << std::setw(2) << b << ": ";
        for (std::size_t j = 0; j < groups[b].size(); ++j) {
            if (j > 0) std::cout << " -> ";
            std::cout << groups[b][j].first << "=" << groups[b][j].second;
        }
        if (groups[b].size() > 1) std::cout << "  (collision chain)";
        std::cout << "\n";
    }
    std::size_t empty_count = map.BucketCount() - occupied;
    if (empty_count > 0) {
        std::cout << "      (" << empty_count << " empty buckets)\n";
    }
}

int main() {
    std::cout << "=============================================\n"
              << "  Custom Unordered Map (Hash Table)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  An unordered map stores key-value pairs and lets you\n"
              << "  look up any value by its key in O(1) average time.\n"
              << "  It's the data structure behind Python's dict, Java's\n"
              << "  HashMap, and C++'s std::unordered_map.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Hash the key to get a number.\n"
              << "  2. Pick a bucket: hash % number_of_buckets.\n"
              << "  3. Store the key-value pair in that bucket.\n"
              << "  4. If two keys land in the same bucket (collision),\n"
              << "     both are stored in a linked list (separate chaining).\n"
              << "  5. When too many elements per bucket (high load factor),\n"
              << "     the table doubles its buckets and redistributes\n"
              << "     everything (rehashing).\n\n"

              << "KEY TERMS\n"
              << "  Bucket      : A slot in the array.  Each holds a chain.\n"
              << "  Collision   : Two keys land in the same bucket.\n"
              << "  Load factor : elements / buckets.  Higher = more\n"
              << "                collisions, slower lookups.\n"
              << "  Rehash      : Double the buckets and redistribute all\n"
              << "                elements to restore fast lookups.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Phone book (name -> phone number)
    //
    // Start with 4 buckets to make rehashing visible quickly.
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Building a Phone Book\n\n"
              << "  Starting with 4 buckets (small on purpose so we can\n"
              << "  see rehashing happen).\n\n";

    UnorderedMap<std::string, std::string> contacts(4);

    struct Contact {
        std::string name;
        std::string phone;
    };
    std::vector<Contact> people = {
        {"Alice",   "555-0101"}, {"Bob",     "555-0102"},
        {"Carol",   "555-0103"}, {"Dave",    "555-0104"},
        {"Eve",     "555-0105"}, {"Frank",   "555-0106"},
        {"Grace",   "555-0107"}, {"Henry",   "555-0108"},
        {"Ivy",     "555-0109"}, {"Jack",    "555-0110"},
    };

    std::cout << "  Adding contacts one by one:\n\n";

    std::size_t prev_buckets = contacts.BucketCount();
    for (std::size_t i = 0; i < people.size(); ++i) {
        const auto& c = people[i];
        contacts.Insert(c.name, c.phone);

        std::cout << "    Insert(\"" << std::left << std::setw(6)
                  << c.name << "\", \"" << c.phone << "\")"
                  << std::right
                  << "  -> bucket " << std::setw(2)
                  << contacts.Bucket(c.name);

        if (contacts.BucketCount() != prev_buckets) {
            std::cout << "  REHASH! " << prev_buckets << " -> "
                      << contacts.BucketCount() << " buckets";
            prev_buckets = contacts.BucketCount();
        }
        std::cout << "  (" << contacts.Size() << " contacts, LF="
                  << std::fixed << std::setprecision(2)
                  << contacts.LoadFactor() << ")\n";
    }

    std::cout << "\n  Final state: " << contacts.Size() << " contacts, "
              << contacts.BucketCount() << " buckets, load factor "
              << contacts.LoadFactor() << "\n";

    // -----------------------------------------------------------------
    // Inside the hash table — bucket distribution
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Inside the Hash Table\n"
              << "---------------------------------------------\n\n"
              << "  How the 10 contacts are distributed across buckets:\n\n";

    PrintBuckets(contacts);

    std::cout << "\n  Buckets with multiple entries are collision chains.\n"
              << "  Lookups in those buckets scan the chain (O(chain length)).\n"
              << "  Rehashing keeps chains short by spreading elements across\n"
              << "  more buckets.\n";

    // -----------------------------------------------------------------
    // Lookup, update, and delete
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Lookup, Update, and Delete\n"
              << "---------------------------------------------\n\n";

    // Lookup
    std::cout << "  Find(\"Alice\")  -> "
              << *contacts.Find("Alice") << "\n"
              << "  Find(\"Frank\")  -> "
              << *contacts.Find("Frank") << "\n"
              << "  Contains(\"Zoe\") -> "
              << (contacts.Contains("Zoe") ? "true" : "false")
              << "  (not in the book)\n\n";

    // Update via operator[]
    std::cout << "  Updating Bob's number:\n"
              << "    Before: contacts[\"Bob\"] = "
              << contacts["Bob"] << "\n";
    contacts["Bob"] = "555-9999";
    std::cout << "    After:  contacts[\"Bob\"] = "
              << contacts["Bob"] << "\n\n";

    // Delete
    std::cout << "  Deleting Carol:\n"
              << "    Erase(\"Carol\") -> "
              << (contacts.Erase("Carol") ? "removed" : "not found") << "\n"
              << "    Contains(\"Carol\") -> "
              << (contacts.Contains("Carol") ? "true" : "false") << "\n"
              << "    Size: " << contacts.Size() << "\n";

    // -----------------------------------------------------------------
    // Collision demo — force all keys into one bucket
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Collision Demo (1 bucket, no rehashing)\n"
              << "---------------------------------------------\n\n"
              << "  With only 1 bucket, every key collides.  All entries\n"
              << "  form a single linked list — lookup becomes O(n).\n\n";

    {
        UnorderedMap<std::string, std::string> tiny(1, 100.0);
        tiny.Insert("X", "100");
        tiny.Insert("Y", "200");
        tiny.Insert("Z", "300");

        std::cout << "    Bucket 0 chain: ";
        for (auto it = tiny.begin(); it != tiny.end(); ++it) {
            auto [k, v] = *it;
            std::cout << k << "=" << v << " -> ";
        }
        std::cout << "end\n"
                  << "    BucketSize(0) = " << tiny.BucketSize(0)
                  << "  (all 3 in one chain)\n\n"
                  << "  This is why hash tables need enough buckets!\n"
                  << "  With a good hash and load factor <= 1.0, most\n"
                  << "  buckets hold 0 or 1 element — O(1) lookup.\n";
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (using the phone book)\n"
              << "  Commands:\n"
              << "    put NAME PHONE   -- add/update a contact\n"
              << "    get NAME         -- look up a phone number\n"
              << "    del NAME         -- delete a contact\n"
              << "    show             -- list all contacts\n"
              << "    buckets          -- show bucket distribution\n"
              << "    q                -- quit\n"
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
                std::size_t space = cmd.find(' ', pos);
                if (space == std::string::npos) {
                    throw std::runtime_error("missing phone");
                }
                std::string name = cmd.substr(pos, space - pos);
                std::string phone = cmd.substr(space + 1);

                bool existed = contacts.Contains(name);
                if (existed) {
                    contacts[name] = phone;
                    std::cout << "    Updated: " << name << " -> "
                              << phone << "\n";
                } else {
                    contacts.Insert(name, phone);
                    std::cout << "    Added: " << name << " -> " << phone
                              << "  (bucket " << contacts.Bucket(name)
                              << ")\n";
                }
                std::cout << "    Size: " << contacts.Size()
                          << "  Buckets: " << contacts.BucketCount()
                          << "  LF: " << std::fixed << std::setprecision(2)
                          << contacts.LoadFactor() << "\n";
            } catch (...) {
                std::cout << "    Invalid. Usage: put NAME PHONE"
                          << "  (e.g. put Zoe 555-1234)\n";
            }
        } else if (cmd.rfind("get", 0) == 0) {
            std::size_t pos = 3;
            while (pos < cmd.size() && cmd[pos] == ' ') ++pos;
            std::string name = cmd.substr(pos);
            auto* val = contacts.Find(name);
            if (val) {
                std::cout << "    " << name << " -> " << *val
                          << "  (bucket " << contacts.Bucket(name) << ")\n";
            } else {
                std::cout << "    Not found.\n";
            }
        } else if (cmd.rfind("del", 0) == 0) {
            std::size_t pos = 3;
            while (pos < cmd.size() && cmd[pos] == ' ') ++pos;
            std::string name = cmd.substr(pos);
            if (contacts.Erase(name)) {
                std::cout << "    Deleted: " << name << "\n";
            } else {
                std::cout << "    Not found.\n";
            }
        } else if (cmd.rfind("show", 0) == 0) {
            if (contacts.Empty()) {
                std::cout << "    (empty)\n";
            } else {
                for (auto it = contacts.begin(); it != contacts.end(); ++it) {
                    auto [name, phone] = *it;
                    std::cout << "    " << name << " -> " << phone << "\n";
                }
                std::cout << "    (" << contacts.Size() << " contacts)\n";
            }
        } else if (cmd.rfind("buckets", 0) == 0) {
            std::cout << "    " << contacts.BucketCount() << " buckets, "
                      << contacts.Size() << " contacts, LF="
                      << std::fixed << std::setprecision(2)
                      << contacts.LoadFactor() << "\n";
            PrintBuckets(contacts);
        } else {
            std::cout << "    Unknown. Try: put, get, del, show, buckets, q\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
