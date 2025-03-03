/**
 * @file main.cpp
 * @brief Interactive driver for LRU Cache.
 *
 * Simulates a web browser page cache to show how LRU eviction
 * works, demonstrates edge cases, then lets the user experiment
 * with put/get/show commands.
 */

#include "Cache.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Print the cache contents in usage order (most recent first).
 */
static void PrintCache(const LruCache<std::string, std::string>& cache) {
    auto items = cache.GetItems();
    std::cout << "    Cache (" << cache.Size() << "/" << cache.Capacity()
              << "): [";
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << items[i].first;
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=============================================\n"
              << "  LRU Cache (Least Recently Used)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  A cache stores recently used data so you can get it\n"
              << "  back quickly without going to a slow source (disk,\n"
              << "  database, or network).\n\n"

              << "  When the cache is full and a new item arrives, the\n"
              << "  item that hasn't been used for the longest time\n"
              << "  (least recently used) gets evicted to make room.\n\n"

              << "HOW IT WORKS\n"
              << "  Put(key, value) -- store an item in the cache.\n"
              << "    If the cache is full, the least recently used\n"
              << "    item is removed first.\n"
              << "  Get(key) -- look up an item.\n"
              << "    If found (cache HIT), the item moves to \"most\n"
              << "    recently used\" so it won't be evicted soon.\n"
              << "    If not found (cache MISS), throws an error.\n\n"

              << "  Internally, LRU uses two data structures:\n"
              << "    1. A doubly-linked list to track usage order.\n"
              << "       Front = most recently used.\n"
              << "       Back  = least recently used (evicted next).\n"
              << "    2. A hash map for O(1) key lookup into the list.\n\n"

              << "WHY CACHES MATTER\n"
              << "  - Web browsers cache pages you've visited recently.\n"
              << "  - CPUs cache memory you've accessed recently.\n"
              << "  - Databases cache query results.\n"
              << "  - DNS resolvers cache domain-to-IP lookups.\n"
              << "  All use some form of LRU or similar eviction.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Web browser page cache
    //
    // The browser can hold 5 pages in memory.  Each visit either
    // finds the page (HIT) or loads it from the network (MISS).
    // When the cache is full on a MISS, the least recently visited
    // page is evicted.
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Web Browser Page Cache\n\n"
              << "  Your browser can hold 5 pages in memory.  When you\n"
              << "  visit a page, the browser checks the cache:\n"
              << "    HIT  -- page is cached, loads instantly.\n"
              << "    MISS -- not cached, loads from network and gets\n"
              << "            added to the cache.\n"
              << "  If the cache is full on a MISS, the page you haven't\n"
              << "  visited in the longest time is evicted.\n\n";

    // Page titles for the simulation
    std::unordered_map<std::string, std::string> page_titles = {
        {"google.com",        "Search Engine"},
        {"github.com",        "Code Repository"},
        {"reddit.com",        "Discussion Forum"},
        {"youtube.com",       "Video Platform"},
        {"stackoverflow.com", "Developer Q&A"},
        {"twitter.com",       "Social Media"},
        {"linkedin.com",      "Professional Network"},
        {"facebook.com",      "Social Network"},
        {"amazon.com",        "E-Commerce"},
        {"netflix.com",       "Streaming Service"},
        {"hulu.com",          "Streaming Service"},
        {"wikipedia.org",     "Encyclopedia"},
    };

    // 20 page visits -- github.com and twitter.com are visited
    // frequently, so LRU should keep them cached.
    std::vector<std::string> visits = {
        "google.com", "github.com", "reddit.com", "youtube.com",
        "stackoverflow.com",
        "twitter.com", "github.com", "linkedin.com", "google.com",
        "facebook.com", "github.com", "twitter.com", "amazon.com",
        "github.com", "netflix.com", "github.com", "hulu.com",
        "twitter.com", "wikipedia.org", "github.com",
    };

    LruCache<std::string, std::string> cache(5);
    int hits = 0;

    std::cout << "  Simulating " << visits.size()
              << " page visits (cache holds " << cache.Capacity()
              << " pages):\n\n";

    for (std::size_t i = 0; i < visits.size(); ++i) {
        const auto& url = visits[i];

        std::cout << "    Visit " << std::setw(2) << (i + 1) << ": "
                  << std::left << std::setw(24) << url << std::right;

        if (cache.Exists(url)) {
            // Cache hit: the page is already in memory.
            cache.Get(url);
            std::cout << "HIT";
            ++hits;
        } else {
            // Cache miss: need to load from network.
            // If full, note which page gets evicted.
            std::string evicted;
            if (cache.Size() == cache.Capacity()) {
                auto items = cache.GetItems();
                evicted = items.back().first;
            }
            cache.Put(url, page_titles.at(url));
            if (evicted.empty()) {
                std::cout << "MISS";
            } else {
                std::cout << "MISS  evicted " << evicted;
            }
        }

        if (i == 4) std::cout << "  <- cache full";
        std::cout << "\n";
    }

    // Summary
    int misses = static_cast<int>(visits.size()) - hits;
    std::cout << "\n  Final cache state:\n";
    PrintCache(cache);
    std::cout << "    ^most recent"
              << std::string(40, ' ') << "^least recent\n\n"
              << "  Summary: " << hits << " hits, " << misses
              << " misses out of " << visits.size() << " visits ("
              << std::fixed << std::setprecision(0)
              << (100.0 * hits / static_cast<double>(visits.size()))
              << "% hit rate)\n\n"
              << "  Key insight: github.com was visited 7 times but only\n"
              << "  loaded from the network once (visit 2).  Every later\n"
              << "  visit was a cache hit because frequent access kept it\n"
              << "  at the front, safe from eviction.  LRU naturally\n"
              << "  favors popular items.\n";

    // -----------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Edge Cases\n"
              << "---------------------------------------------\n\n";

    // Edge case 1: Update an existing key
    std::cout << "  1. Updating an existing entry\n\n";
    {
        LruCache<std::string, std::string> demo(3);
        demo.Put("page-A", "Old content");
        demo.Put("page-B", "Page B");
        demo.Put("page-C", "Page C");
        std::cout << "    After 3 inserts:\n";
        PrintCache(demo);

        demo.Put("page-A", "Updated content");
        std::cout << "    Put(\"page-A\", \"Updated content\")\n"
                  << "    After update:\n";
        PrintCache(demo);
        std::cout << "    Get(\"page-A\") = \"" << demo.Get("page-A") << "\"\n"
                  << "    Size = " << demo.Size()
                  << " (unchanged -- no eviction needed)\n"
                  << "    page-A moved to front with its new value.\n\n";
    }

    // Edge case 2: Capacity 1
    std::cout << "  2. Cache with capacity 1 (every insert evicts)\n\n";
    {
        LruCache<std::string, std::string> demo(1);
        demo.Put("A", "first");
        std::cout << "    Put(\"A\", \"first\")\n";
        PrintCache(demo);

        demo.Put("B", "second");
        std::cout << "    Put(\"B\", \"second\")  -> evicts A\n";
        PrintCache(demo);

        demo.Put("C", "third");
        std::cout << "    Put(\"C\", \"third\")   -> evicts B\n";
        PrintCache(demo);
        std::cout << "    Only one item survives at a time.\n\n";
    }

    // Edge case 3: Access after eviction
    std::cout << "  3. Accessing an evicted item\n\n";
    {
        LruCache<std::string, std::string> demo(2);
        demo.Put("X", "value-X");
        demo.Put("Y", "value-Y");
        std::cout << "    Cache is full:\n";
        PrintCache(demo);
        demo.Put("Z", "value-Z");
        std::cout << "    Put(\"Z\", ...) evicts X:\n";
        PrintCache(demo);
        std::cout << "    Get(\"X\") -> ";
        try {
            demo.Get("X");
        } catch (const std::out_of_range& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
        std::cout << "    Once evicted, the data must be reloaded from\n"
                  << "    the original source.\n";
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (cache capacity: 3)\n"
              << "  Commands:\n"
              << "    put KEY VALUE  -- store a key-value pair\n"
              << "    get KEY        -- look up a key\n"
              << "    show           -- display current cache state\n"
              << "    q              -- quit\n"
              << "---------------------------------------------\n";

    LruCache<std::string, std::string> interactive(3);

    while (true) {
        std::cout << "\n  > ";
        std::string cmd;
        std::getline(std::cin, cmd);

        if (cmd.empty() || cmd[0] == 'q' || cmd[0] == 'Q') {
            break;
        }

        if (cmd.rfind("put", 0) == 0) {
            try {
                std::size_t pos = 3;
                while (pos < cmd.size() && cmd[pos] == ' ') ++pos;
                std::size_t space = cmd.find(' ', pos);
                if (space == std::string::npos) {
                    throw std::runtime_error("missing value");
                }
                std::string key = cmd.substr(pos, space - pos);
                std::string value = cmd.substr(space + 1);

                std::string evicted;
                if (!interactive.Exists(key)
                    && interactive.Size() == interactive.Capacity()) {
                    auto items = interactive.GetItems();
                    evicted = items.back().first;
                }

                interactive.Put(key, value);
                if (!evicted.empty()) {
                    std::cout << "    Evicted: " << evicted << "\n";
                }
                std::cout << "    Stored: " << key << " = " << value << "\n";
                PrintCache(interactive);
            } catch (...) {
                std::cout << "    Invalid. Usage: put KEY VALUE"
                          << "  (e.g. put name Alice)\n";
            }
        } else if (cmd.rfind("get", 0) == 0) {
            try {
                std::size_t pos = 3;
                while (pos < cmd.size() && cmd[pos] == ' ') ++pos;
                std::string key = cmd.substr(pos);
                std::string value = interactive.Get(key);
                std::cout << "    " << key << " = " << value
                          << "  (moved to most recent)\n";
                PrintCache(interactive);
            } catch (const std::out_of_range&) {
                std::cout << "    Key not found (cache miss).\n";
            } catch (...) {
                std::cout << "    Invalid. Usage: get KEY"
                          << "  (e.g. get name)\n";
            }
        } else if (cmd.rfind("show", 0) == 0) {
            if (interactive.Size() == 0) {
                std::cout << "    Cache is empty.\n";
            } else {
                auto items = interactive.GetItems();
                std::cout << "    Cache (" << interactive.Size() << "/"
                          << interactive.Capacity() << "):\n";
                for (std::size_t i = 0; i < items.size(); ++i) {
                    std::cout << "      " << (i + 1) << ". "
                              << items[i].first << " = "
                              << items[i].second;
                    if (i == 0) {
                        std::cout << "  (most recent)";
                    }
                    if (i == items.size() - 1) {
                        std::cout << "  (least recent -- evicted next)";
                    }
                    std::cout << "\n";
                }
            }
        } else {
            std::cout << "    Unknown command. Try: put, get, show, q\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
