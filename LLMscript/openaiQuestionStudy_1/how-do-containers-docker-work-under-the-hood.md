# How do containers (Docker) work under the hood?

Containers, such as those managed by Docker, are a popular technology for deploying and running applications in a consistent and isolated environment. They use several features of the Linux kernel (and Windows, in the case of Windows containers) to provide the necessary isolation and resource management. Let's explore how containers work under the hood, particularly focusing on Linux-based containers, which are the most common.

### 1. **Namespaces**

Namespaces are a feature of the Linux kernel that provide isolation for running processes. They ensure that processes in one namespace cannot see or affect processes in another namespace. There are several types of namespaces, each isolating a different aspect of the operating system:

- **PID (Process ID) Namespace**: Isolates the process ID number space, meaning processes in different PID namespaces can have the same PID.
- **Network Namespace**: Provides isolated network environments, which include network devices, IP addresses, routing tables, etc.
- **Mount Namespace**: Isolates filesystem mount points seen by a group of processes, so that one set of processes does not see the same filesystem as another.
- **IPC (Interprocess Communication) Namespace**: Isolates interprocess communication between processes.
- **UTS (UNIX Time-sharing System) Namespace**: Isolates system identifiers, including the hostname and the domain name.
- **User Namespace**: Isolates user and group IDs, which is crucial for security and ensuring that root in a container does not mean root outside the container.

### 2. **Cgroups (Control Groups)**

Cgroups complement namespaces by controlling the amount of resources (CPU, memory, disk I/O, etc.) that a process or group of processes can use. This ensures that a single container cannot monopolize system resources, affecting other containers or the host system. It also supports fine-grained limits and monitoring of resources like memory usage, CPU utilization, and network bandwidth.

### 3. **Union File Systems**

Containers typically use union file systems like OverlayFS to layer file systems on top of each other. This allows a container to have its own private view of the filesystem, which can be modified without affecting other containers or the host filesystem. These layers can be shared between containers, which makes container images lighter as common layers are downloaded or stored only once.

### 4. **Container Runtime**

The container runtime is the software that uses the kernel's namespaces, cgroups, and union file systems to run containers. Docker, for example, uses containerd as its runtime. The runtime handles the lifecycle of containers, including creating, starting, stopping, and destroying containers.

### 5. **Container Image**

Container images are static files that contain all the code, libraries, dependencies, and other filesystem objects needed to execute an application. These images are built from a Dockerfile, which is a script containing commands that assemble the image. Once built, these images can be run as containers on any system that supports Docker, ensuring consistency across environments.

### 6. **Networking**

Docker provides a virtual network to connect containers to each other and to the outside world. Each container can be connected to multiple networks, can have IP addresses, and can use ports. Docker uses network namespaces to provide each container with its own network stack.

### 7. **Storage**

Docker provides several options for managing storage, including volumes and bind mounts. Volumes are managed by Docker and can be used to persist data independently of the container's life cycle. Bind mounts can be used to mount specific paths on the host to the container, allowing the container to read and write files on the host system.

By leveraging these technologies and components, Docker and other container platforms provide a powerful, efficient, and secure way to deploy and manage applications.