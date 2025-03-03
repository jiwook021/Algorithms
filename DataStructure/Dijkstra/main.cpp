/**
 * @file main.cpp
 * @brief Interactive driver for Dijkstra and Bellman-Ford shortest path.
 *
 * Simulates a city delivery route planning scenario, shows how
 * Dijkstra finds shorter routes through intermediate locations,
 * demonstrates Bellman-Ford with negative edges, then lets the
 * user experiment interactively.
 */

#include "Dijkstra.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=============================================\n"
              << "  Dijkstra's Shortest Path Algorithm\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  Given a map of locations connected by roads with\n"
              << "  distances, Dijkstra finds the shortest route from\n"
              << "  a starting location to every other location.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Set distance to start = 0, all others = infinity.\n"
              << "  2. Pick the unvisited location with the smallest\n"
              << "     known distance (using a min-heap).\n"
              << "  3. For each neighbor, check: is going through the\n"
              << "     current location shorter?  If yes, update.\n"
              << "  4. Mark the current location as done.  Repeat.\n\n"

              << "DIJKSTRA vs BELLMAN-FORD\n"
              << "  Dijkstra:     Fast O(V log V + E), but requires\n"
              << "                all edge weights >= 0.\n"
              << "  Bellman-Ford: Slower O(V * E), but handles negative\n"
              << "                weights and detects negative cycles.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: City delivery route planning
    //
    // A delivery company at the Warehouse needs the shortest route
    // to 5 other city locations.
    //
    //  Warehouse(1) --3-- Downtown(2) --5-- Mall(4) --1-- University(6)
    //       |                 |                                |
    //       7                 1                                3
    //       |                 |                                |
    //  Airport(3) -----------+                          Hospital(5)
    //    /       |
    //   2        4
    //  /         |
    // Mall(4)   Hospital(5)
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: City Delivery Route Planning\n\n"
              << "  A delivery company at the Warehouse needs the\n"
              << "  shortest route to 5 other locations in the city.\n\n";

    std::vector<std::string> location = {
        "",             // index 0 unused (vertices are 1-indexed)
        "Warehouse",    // 1  (starting point)
        "Downtown",     // 2
        "Airport",      // 3
        "Mall",         // 4
        "Hospital",     // 5
        "University",   // 6
    };

    std::cout << "  Locations:\n";
    for (std::size_t i = 1; i <= 6; ++i) {
        std::cout << "    " << i << ": " << location[i];
        if (i == 1) std::cout << "  (starting point)";
        std::cout << "\n";
    }

    // Build the graph, printing each road as it's added
    Graph<int> graph(6);

    struct Road {
        std::size_t from;
        std::size_t to;
        int km;
    };
    std::vector<Road> roads = {
        {1, 2, 3}, {1, 3, 7}, {2, 3, 1}, {2, 4, 5},
        {3, 4, 2}, {3, 5, 4}, {4, 6, 1}, {5, 6, 3},
    };

    std::cout << "\n  Building the road network:\n\n";
    for (std::size_t i = 0; i < roads.size(); ++i) {
        const auto& r = roads[i];
        graph.AddEdge(r.from, r.to, r.km);
        std::cout << "    AddEdge(" << location[r.from] << ", "
                  << location[r.to] << ", " << r.km << " km)"
                  << "   road " << (i + 1) << "/" << roads.size() << "\n";
    }

    std::cout << "\n  Graph: " << graph.NumVertices() << " locations, "
              << roads.size() << " roads\n";

    // -----------------------------------------------------------------
    // Run Dijkstra from Warehouse (vertex 1)
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Running Dijkstra from Warehouse (vertex 1)\n"
              << "---------------------------------------------\n\n"
              << "  Finding shortest delivery routes...\n\n"
              << "    Location            Shortest Distance\n"
              << "    ------------------  -----------------\n";

    auto dist = graph.Dijkstra(1);

    for (std::size_t i = 1; i <= 6; ++i) {
        std::cout << "    " << std::setw(2) << i << "  "
                  << std::left << std::setw(16) << location[i]
                  << std::right;
        if (dist[i] == Graph<int>::kInfinity) {
            std::cout << "  unreachable\n";
        } else {
            std::cout << std::setw(5) << dist[i] << " km";
            if (i == 1) std::cout << "  (start)";
            std::cout << "\n";
        }
    }

    std::cout << "\n  Interesting findings:\n\n"
              << "    Direct road to Airport is 7 km, but Dijkstra\n"
              << "    found a shorter route through Downtown:\n"
              << "      Warehouse --3km--> Downtown --1km--> Airport = 4 km\n"
              << "      (saved 3 km!)\n\n"
              << "    Mall via Downtown is 3+5 = 8 km, but via Airport:\n"
              << "      Warehouse -> Downtown -> Airport -> Mall\n"
              << "      3 + 1 + 2 = 6 km  (saved 2 km!)\n";

    // -----------------------------------------------------------------
    // Bellman-Ford comparison (same graph)
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Bellman-Ford on the Same Graph\n"
              << "---------------------------------------------\n\n"
              << "  With non-negative weights, Bellman-Ford gives the\n"
              << "  same results as Dijkstra (just slower).\n\n";

    auto bf_dist = graph.BellmanFord(1);
    bool all_match = true;
    for (std::size_t i = 1; i <= 6; ++i) {
        if (dist[i] != bf_dist[i]) { all_match = false; break; }
    }
    std::cout << "    All distances match Dijkstra: "
              << (all_match ? "yes" : "no") << "\n";

    // -----------------------------------------------------------------
    // Bellman-Ford with negative edges
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Bellman-Ford with Negative Edges\n"
              << "---------------------------------------------\n\n"
              << "  Scenario: A delivery earns a $5 rebate when passing\n"
              << "  through a partner hub.  We model this as a negative\n"
              << "  edge weight (a cost reduction).\n\n"
              << "    Directed graph:\n"
              << "      1 --$10--> 2 --(-$5)--> 3\n"
              << "      1 ---$7-----------> 3\n\n";

    {
        Graph<int> directed(3);
        directed.AddEdge(1, 2, 10, false);   // directed
        directed.AddEdge(2, 3, -5, false);   // negative (rebate)
        directed.AddEdge(1, 3, 7,  false);   // direct route

        auto bf = directed.BellmanFord(1);
        std::cout << "    Bellman-Ford from vertex 1:\n"
                  << "      To 2:  $" << bf[2]
                  << "  (direct: 1 -> 2)\n"
                  << "      To 3:  $" << bf[3]
                  << "  (1 -> 2 -> 3 = 10 + (-5) = 5, shorter than direct 7)\n\n"
                  << "    The rebate at vertex 2 made the longer route\n"
                  << "    cheaper overall.  Dijkstra can't handle this\n"
                  << "    because it assumes costs never decrease.\n";
    }

    // -----------------------------------------------------------------
    // Bellman-Ford negative cycle detection
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Negative Cycle Detection\n"
              << "---------------------------------------------\n\n"
              << "  If a cycle has negative total weight, you could\n"
              << "  loop forever and reduce cost infinitely.  Bellman-\n"
              << "  Ford detects this and returns an empty result.\n\n"
              << "    Directed graph:\n"
              << "      1 --1--> 2 --1--> 3 --(-5)--> 1\n"
              << "      Cycle weight: 1 + 1 + (-5) = -3  (negative!)\n\n";

    {
        Graph<int> cycle_graph(3);
        cycle_graph.AddEdge(1, 2, 1,  false);
        cycle_graph.AddEdge(2, 3, 1,  false);
        cycle_graph.AddEdge(3, 1, -5, false);

        auto bf = cycle_graph.BellmanFord(1);
        if (bf.empty()) {
            std::cout << "    Bellman-Ford result: negative cycle detected!\n"
                      << "    No meaningful shortest path exists.\n";
        } else {
            std::cout << "    No negative cycle detected.\n";
        }
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (using the city graph)\n"
              << "  Commands:\n"
              << "    add FROM TO WEIGHT  -- add an edge (e.g. add 1 5 2)\n"
              << "    dijkstra SOURCE     -- shortest paths (e.g. dijkstra 1)\n"
              << "    bellman SOURCE      -- Bellman-Ford (e.g. bellman 1)\n"
              << "    q                   -- quit\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string cmd;
        std::getline(std::cin, cmd);

        if (cmd.empty() || cmd[0] == 'q' || cmd[0] == 'Q') {
            break;
        }

        if (cmd.rfind("add", 0) == 0) {
            try {
                std::size_t pos = 3;
                unsigned long from = std::stoul(cmd.substr(pos), &pos);
                unsigned long to = std::stoul(cmd.substr(3 + pos), &pos);
                int weight = std::stoi(cmd.substr(3 + pos));
                graph.AddEdge(from, to, weight);
                std::cout << "    Added: " << from << " <--" << weight
                          << "--> " << to << "\n";
            } catch (...) {
                std::cout << "    Invalid. Usage: add FROM TO WEIGHT"
                          << "  (e.g. add 1 5 2)\n";
            }
        } else if (cmd.rfind("dijkstra", 0) == 0) {
            try {
                unsigned long src = std::stoul(cmd.substr(8));
                auto d = graph.Dijkstra(src);
                std::cout << "    Shortest distances from vertex " << src << ":\n";
                for (std::size_t i = 1; i <= graph.NumVertices(); ++i) {
                    std::cout << "      -> " << i;
                    if (i <= 6) std::cout << " (" << location[i] << ")";
                    std::cout << ": ";
                    if (d[i] == Graph<int>::kInfinity) {
                        std::cout << "unreachable\n";
                    } else {
                        std::cout << d[i] << "\n";
                    }
                }
            } catch (...) {
                std::cout << "    Invalid. Usage: dijkstra SOURCE"
                          << "  (e.g. dijkstra 1)\n";
            }
        } else if (cmd.rfind("bellman", 0) == 0) {
            try {
                unsigned long src = std::stoul(cmd.substr(7));
                auto d = graph.BellmanFord(src);
                if (d.empty()) {
                    std::cout << "    Negative cycle detected!\n";
                } else {
                    std::cout << "    Bellman-Ford from vertex " << src << ":\n";
                    for (std::size_t i = 1; i <= graph.NumVertices(); ++i) {
                        std::cout << "      -> " << i;
                        if (i <= 6) std::cout << " (" << location[i] << ")";
                        std::cout << ": ";
                        if (d[i] == Graph<int>::kInfinity) {
                            std::cout << "unreachable\n";
                        } else {
                            std::cout << d[i] << "\n";
                        }
                    }
                }
            } catch (...) {
                std::cout << "    Invalid. Usage: bellman SOURCE"
                          << "  (e.g. bellman 1)\n";
            }
        } else {
            std::cout << "    Unknown command. Try: add, dijkstra, bellman, q\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
