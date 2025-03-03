/**
 * @file main.cpp
 * @brief Educational driver for SegmentTree — stock price monitoring scenario.
 */

#include "SegmentTree.hpp"

#include <algorithm>
#include <climits>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main() {
    std::cout
        << "=============================================\n"
        << "       Segment Tree\n"
        << "=============================================\n\n"

        << "PURPOSE\n"
        << "  Given an array of N values, a segment tree lets you:\n"
        << "    1. Query any contiguous range [L..R] in O(log N)\n"
        << "       (sum, min, max, gcd -- any associative operation)\n"
        << "    2. Update a single element in O(log N)\n\n"

        << "HOW IT WORKS\n"
        << "  Build:  store leaves at the bottom, fill parents bottom-up.\n"
        << "  Query:  start at the root.  At each node, decide:\n"
        << "          - completely inside the query range?  return value.\n"
        << "          - completely outside?  return identity.\n"
        << "          - partial overlap?  recurse into both children.\n"
        << "  Update: change the leaf, walk up to root, recompute each parent.\n\n"

        << "WHEN TO USE IT\n"
        << "  Naive scan:     O(N) per query, O(1) update.\n"
        << "  Prefix sums:    O(1) per query, but O(N) update.\n"
        << "  Segment tree:   O(log N) for BOTH -- the sweet spot when\n"
        << "                  you need fast queries AND frequent updates.\n"
        << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Tree structure visualization
    // -----------------------------------------------------------------

    std::cout
        << "TREE STRUCTURE (for array [3, 1, 4, 1, 5, 9])\n\n"
        << "  The tree is stored in a flat array, 1-indexed:\n\n"
        << "                    [23]                tree[1]  root (sum of all)\n"
        << "                   /    \\\n"
        << "               [8]      [15]            tree[2..3]\n"
        << "              /   \\    /    \\\n"
        << "           [4]   [4] [5]   [10]         tree[4..7]\n"
        << "           / \\   / \\  / \\   / \\\n"
        << "          3   1 4   1 5   9 0   0       tree[8..15] (leaves)\n"
        << "          ^   ^ ^   ^ ^   ^\n"
        << "         [0] [1][2][3][4] [5]           original indices\n\n"
        << "  Parent i holds op(child 2i, child 2i+1).\n"
        << "  Leaf for index k lives at tree[size + k].\n"
        << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Weekly stock price monitor
    // -----------------------------------------------------------------

    std::cout
        << "SCENARIO: Weekly Stock Price Monitor\n\n"
        << "  A trading firm tracks daily closing prices for 8 days.\n"
        << "  They need instant answers to questions like:\n"
        << "    - What is the total revenue over days 2-5?\n"
        << "    - What was the lowest price this week?\n"
        << "    - What was the highest price in the last 3 days?\n\n";

    std::vector<std::string> day_name = {
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon2"
    };
    std::vector<int> prices = {142, 138, 145, 149, 137, 151, 148, 153};

    std::cout << "  Day prices:\n\n"
              << "    Day       Price\n"
              << "    --------  -----\n";
    for (std::size_t i = 0; i < prices.size(); ++i) {
        std::cout << "    " << std::left << std::setw(10) << day_name[i]
                  << std::right << std::setw(3) << prices[i] << "\n";
    }

    // -----------------------------------------------------------------
    // 1. Range-sum: total revenue
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  1. Range Sum -- total closing prices\n"
              << "---------------------------------------------\n\n";

    SegmentTree<int> sum_tree(prices);

    std::cout << "  SegmentTree<int> sum_tree(prices);\n"
              << "  (default operation: addition, identity: 0)\n\n";

    auto print_sum = [&](std::size_t l, std::size_t r) {
        std::cout << "    Sum [" << day_name[l] << ".." << day_name[r]
                  << "]  (indices " << l << ".." << r << ") = "
                  << sum_tree.Query(l, r) << "\n";
    };

    print_sum(0, 7);  // entire week
    print_sum(0, 4);  // Mon-Fri
    print_sum(5, 7);  // weekend + Mon2
    print_sum(2, 5);  // Wed-Sat

    // -----------------------------------------------------------------
    // 2. Range-min: lowest price
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  2. Range Min -- lowest closing price\n"
              << "---------------------------------------------\n\n";

    SegmentTree<int> min_tree(prices, INT_MAX,
                              [](int a, int b) { return std::min(a, b); });

    std::cout << "  SegmentTree<int> min_tree(prices, INT_MAX,\n"
              << "      [](int a, int b) { return std::min(a, b); });\n\n";

    auto print_min = [&](std::size_t l, std::size_t r) {
        std::cout << "    Min [" << day_name[l] << ".." << day_name[r]
                  << "]  (indices " << l << ".." << r << ") = "
                  << min_tree.Query(l, r) << "\n";
    };

    print_min(0, 7);  // global min
    print_min(0, 2);  // Mon-Wed
    print_min(3, 5);  // Thu-Sat

    std::cout << "\n    The lowest price all week was on "
              << day_name[4] << " at $" << min_tree.Query(0, 7) << ".\n";

    // -----------------------------------------------------------------
    // 3. Range-max: highest price
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  3. Range Max -- highest closing price\n"
              << "---------------------------------------------\n\n";

    SegmentTree<int> max_tree(prices, INT_MIN,
                              [](int a, int b) { return std::max(a, b); });

    std::cout << "  SegmentTree<int> max_tree(prices, INT_MIN,\n"
              << "      [](int a, int b) { return std::max(a, b); });\n\n";

    auto print_max = [&](std::size_t l, std::size_t r) {
        std::cout << "    Max [" << day_name[l] << ".." << day_name[r]
                  << "]  (indices " << l << ".." << r << ") = "
                  << max_tree.Query(l, r) << "\n";
    };

    print_max(0, 7);
    print_max(0, 4);  // weekdays
    print_max(5, 7);  // weekend

    std::cout << "\n    The highest price all week was on "
              << day_name[7] << " at $" << max_tree.Query(0, 7) << ".\n";

    // -----------------------------------------------------------------
    // 4. Point update -- breaking news!
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  4. Point Update -- breaking news!\n"
              << "---------------------------------------------\n\n"
              << "  After-hours correction: Wednesday's price revised\n"
              << "  from $" << prices[2] << " to $160.\n\n"
              << "  sum_tree.Update(2, 160);\n\n";

    sum_tree.Update(2, 160);
    min_tree.Update(2, 160);
    max_tree.Update(2, 160);

    std::cout << "  HOW UPDATE PROPAGATES (sum tree, index 2 -> 160):\n\n"
              << "    Step 1: Set leaf tree[size+2] = 160\n"
              << "            (was 145, now 160)\n\n"
              << "    Step 2: Recompute parent:\n"
              << "            tree[parent] = tree[left] + tree[right]\n"
              << "            Walk up to root, one level at a time.\n\n"
              << "    Before:          After:\n"
              << "       [1163]           [1178]\n"
              << "       /    \\           /    \\\n"
              << "    [574]  [589]     [589]  [589]\n"
              << "    / \\    / \\       / \\    / \\\n"
              << "  280 294 288 301  280 309 288 301\n"
              << "  /\\  /\\  /\\  /\\   /\\  /\\  /\\  /\\\n"
              << " ... 145 ...      ... 160 ...\n\n";

    std::cout << "  Updated results:\n\n";
    print_sum(0, 7);
    print_min(0, 7);
    print_max(0, 7);

    std::cout << "\n  The new max is now $160 (revised Wednesday price).\n";

    // -----------------------------------------------------------------
    // 5. How a query traverses the tree
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  5. How Query(2, 5) Traverses the Tree\n"
              << "---------------------------------------------\n\n"
              << "  Query range: indices 2..5 (Wed through Sat)\n\n"
              << "  Node [1] covers [0..7] -- partial overlap, recurse.\n"
              << "    Node [2] covers [0..3] -- partial overlap, recurse.\n"
              << "      Node [4] covers [0..1] -- OUTSIDE, return identity.\n"
              << "      Node [5] covers [2..3] -- INSIDE, return tree[5].\n"
              << "    Node [3] covers [4..7] -- partial overlap, recurse.\n"
              << "      Node [6] covers [4..5] -- INSIDE, return tree[6].\n"
              << "      Node [7] covers [6..7] -- OUTSIDE, return identity.\n\n"
              << "  Only 4 nodes visited out of 15.  That's O(log N)!\n\n"
              << "  Sum [Wed..Sat] = " << sum_tree.Query(2, 5)
              << "  (160 + 149 + 137 + 151 = 597)\n";

    // -----------------------------------------------------------------
    // 6. Range-GCD (bonus operation)
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  6. Bonus: Range GCD\n"
              << "---------------------------------------------\n\n"
              << "  GCD is also associative, so it works with a segment tree.\n"
              << "  Identity for GCD is 0 (because gcd(0, x) = x).\n\n";

    std::vector<int> gcd_data = {12, 18, 24, 36, 48, 60};
    SegmentTree<int> gcd_tree(gcd_data, 0,
                              [](int a, int b) { return std::gcd(a, b); });

    std::cout << "  Array: 12  18  24  36  48  60\n\n";

    auto print_gcd = [&](std::size_t l, std::size_t r) {
        std::cout << "    GCD [" << l << ".." << r << "] = "
                  << gcd_tree.Query(l, r) << "\n";
    };

    print_gcd(0, 5);  // gcd of all = 6
    print_gcd(0, 2);  // gcd(12,18,24) = 6
    print_gcd(2, 5);  // gcd(24,36,48,60) = 12
    print_gcd(3, 5);  // gcd(36,48,60) = 12
    print_gcd(4, 5);  // gcd(48,60) = 12

    // -----------------------------------------------------------------
    // 7. Same data, four operations -- side-by-side comparison
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  7. Same Data, Four Operations\n"
              << "---------------------------------------------\n\n"
              << "  Array: 12  18  24  36  48  60\n\n"
              << "    Range      Sum    Min   Max   GCD\n"
              << "    ---------  -----  ----  ----  ----\n";

    SegmentTree<int> s(gcd_data);
    SegmentTree<int> mn(gcd_data, INT_MAX,
                        [](int a, int b) { return std::min(a, b); });
    SegmentTree<int> mx(gcd_data, INT_MIN,
                        [](int a, int b) { return std::max(a, b); });

    struct Range { std::size_t l, r; };
    std::vector<Range> ranges = {{0,5}, {0,2}, {2,5}, {3,5}};
    for (const auto& rng : ranges) {
        std::cout << "    [" << rng.l << ".." << rng.r << "]     "
                  << std::setw(5) << s.Query(rng.l, rng.r)
                  << std::setw(6) << mn.Query(rng.l, rng.r)
                  << std::setw(6) << mx.Query(rng.l, rng.r)
                  << std::setw(6) << gcd_tree.Query(rng.l, rng.r) << "\n";
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (stock prices: 8 days)\n"
              << "  Commands:\n"
              << "    update INDEX VALUE   -- point update (e.g. update 2 160)\n"
              << "    sum    LEFT RIGHT    -- range sum    (e.g. sum 0 7)\n"
              << "    min    LEFT RIGHT    -- range min    (e.g. min 2 5)\n"
              << "    max    LEFT RIGHT    -- range max    (e.g. max 0 3)\n"
              << "    q                    -- quit\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string cmd;
        if (!std::getline(std::cin, cmd)) break;
        if (cmd.empty() || cmd[0] == 'q' || cmd[0] == 'Q') break;

        try {
            if (cmd.rfind("update", 0) == 0) {
                std::size_t pos = 6;
                unsigned long idx = std::stoul(cmd.substr(pos), &pos);
                int val = std::stoi(cmd.substr(6 + pos));
                sum_tree.Update(idx, val);
                min_tree.Update(idx, val);
                max_tree.Update(idx, val);
                std::cout << "    Updated index " << idx << " -> " << val << "\n";
            } else if (cmd.rfind("sum", 0) == 0) {
                std::size_t pos = 3;
                unsigned long l = std::stoul(cmd.substr(pos), &pos);
                unsigned long r = std::stoul(cmd.substr(3 + pos));
                std::cout << "    Sum [" << l << ".." << r << "] = "
                          << sum_tree.Query(l, r) << "\n";
            } else if (cmd.rfind("min", 0) == 0) {
                std::size_t pos = 3;
                unsigned long l = std::stoul(cmd.substr(pos), &pos);
                unsigned long r = std::stoul(cmd.substr(3 + pos));
                std::cout << "    Min [" << l << ".." << r << "] = "
                          << min_tree.Query(l, r) << "\n";
            } else if (cmd.rfind("max", 0) == 0) {
                std::size_t pos = 3;
                unsigned long l = std::stoul(cmd.substr(pos), &pos);
                unsigned long r = std::stoul(cmd.substr(3 + pos));
                std::cout << "    Max [" << l << ".." << r << "] = "
                          << max_tree.Query(l, r) << "\n";
            } else {
                std::cout << "    Unknown command. Try: update, sum, min, max, q\n";
            }
        } catch (...) {
            std::cout << "    Invalid input. See command formats above.\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
