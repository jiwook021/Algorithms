# What happens when you **type a URL in the browser**?

Typing a URL into a browser initiates a sequence of steps that result in the display of the requested webpage on your device. Here is a detailed breakdown of what happens when you type a URL into your browser:

1. **URL Parsing**:
   - The browser parses the URL (Uniform Resource Locator) to identify the protocol, host, port, and path. For example, in the URL `https://www.example.com:80/path`, `https` is the protocol, `www.example.com` is the host, `80` is the port, and `/path` is the resource path.

2. **DNS Lookup**:
   - The browser needs to determine the IP address of the server hosting the website. It does this through a DNS (Domain Name System) lookup. If the IP address is not already cached in your system, the browser queries DNS servers to resolve the domain name (e.g., `www.example.com`) to its corresponding IP address.

3. **TCP Connection**:
   - Once the IP address is known, the browser initiates a TCP (Transmission Control Protocol) connection with the server. This involves a TCP handshake, which is a three-step process (SYN, SYN-ACK, ACK) used to establish a connection between the client (browser) and the server.

4. **Sending an HTTP Request**:
   - After establishing a TCP connection, the browser sends an HTTP (Hypertext Transfer Protocol) request to the server. This request includes the method (e.g., GET, POST), the path of the resource, and headers that carry metadata (like browser type, accepted response formats, etc.).

5. **Server Response**:
   - The server processes the HTTP request and sends back an HTTP response. This response contains a status code (indicating success or error), headers (providing metadata about the response), and often, the requested content (such as HTML, CSS, JavaScript, images, etc.).

6. **Rendering the Webpage**:
   - The browser begins rendering the webpage using the HTML, CSS, and JavaScript received. The rendering engine of the browser parses HTML and CSS, constructs the DOM (Document Object Model), and lays out the page accordingly. JavaScript is executed, potentially altering the DOM and modifying the appearance and behavior of the webpage.

7. **Resource Loading**:
   - As the browser parses the HTML, it may encounter tags that require it to load additional resources (like images, fonts, scripts, or stylesheets). These requests are made similarly to the initial request (steps 3 to 5), involving potentially more DNS lookups, TCP connections, and HTTP requests.

8. **Page Interactivity**:
   - Once all scripts are loaded and executed, and all resources are rendered, the page becomes fully interactive. This means you can now interact with elements like forms, buttons, and other interactive components.

9. **Persistent Connection**:
   - Modern HTTP/1.1 and HTTP/2 protocols support persistent connections, which means the TCP connection can be reused for multiple requests to the same server, reducing the overhead of setting up new connections.

10. **Secure Connections (HTTPS)**:
    - If the protocol is HTTPS (as opposed to HTTP), there is an additional layer of security. This involves setting up a TLS (Transport Layer Security) or SSL (Secure Sockets Layer) encryption layer on top of TCP, which encrypts and decrypts data sent over the connection to protect sensitive information from being intercepted.

Each of these steps involves complex interactions between your computer, network, and remote servers, all happening in just seconds to bring a webpage to your screen.