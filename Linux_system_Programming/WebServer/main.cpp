#include "WebServer.hpp"
#include <sstream>

int main(int argc, char* argv[]) {
    try {
        int port = 8080;
        std::string docRoot = "./www";

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if ((arg == "-p" || arg == "--port") && i + 1 < argc)
                port = std::stoi(argv[++i]);
            else if ((arg == "-d" || arg == "--docroot") && i + 1 < argc)
                docRoot = argv[++i];
            else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -p, --port PORT      Port (default: 8080)\n"
                          << "  -d, --docroot PATH   Document root (default: ./www)\n"
                          << "  -h, --help           Show help\n";
                return 0;
            }
        }

        WebServer::Server server(port, docRoot);

        server.AddRoute("/hello",
            [](const std::unordered_map<std::string, std::string>& params) {
                std::string name = "World";
                auto it = params.find("name");
                if (it != params.end()) name = it->second;
                return "<html><body><h1>Hello, " + name + "!</h1></body></html>";
            });

        server.AddRoute("/info",
            [](const std::unordered_map<std::string, std::string>&) {
                std::stringstream ss;
                ss << "<html><body><h1>Server Information</h1><ul>"
                   << "<li>C++ Web Server</li>"
                   << "<li>Current Time: " << time(nullptr) << "</li>"
                   << "</ul></body></html>";
                return ss.str();
            });

        if (!server.Start()) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }

        std::cout << "Server running. Press Enter to stop." << std::endl;
        std::cin.get();
        server.Stop();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
