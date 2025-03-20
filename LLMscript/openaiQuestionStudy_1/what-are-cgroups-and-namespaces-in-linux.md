# What are cgroups and namespaces in Linux?

In Linux systems, `cgroups` (control groups) and `namespaces` are fundamental components used to implement resource management and isolation features, primarily used by container technologies such as Docker and Kubernetes. Both play crucial roles in managing and isolating system resources, but they do so in different ways:

### Cgroups (Control Groups)
Cgroups were introduced in Linux kernel 2.6.24 (circa 2008) to limit and isolate the resource usage (CPU, memory, disk I/O, etc.) of a collection of processes. They enable you to allocate resources—such as CPU time, system memory, network bandwidth, or combinations of these resources—among user-defined groups of tasks (processes) running on a system. By controlling and limiting the resources that a process or group of processes uses, cgroups help in ensuring that no single process can monopolize system resources, thus implementing resource fairness and preventing system crashes due to resource starvation.

Cgroups provide several key features:
- **Resource limiting**: You can set limits on the resources a group can use, ensuring that no single group uses more than its allocated share, which helps in maintaining system stability and performance.
- **Prioritization**: Some resources can be prioritized among various groups, which is crucial for performance-sensitive applications.
- **Accounting**: Cgroups keep track of the resource usage of different groups, which is useful for monitoring and billing purposes.
- **Control**: It provides the ability to freeze and resume groups of processes, which is useful for management and maintenance of services.

### Namespaces
Namespaces, introduced in Linux kernel 2.4.19 (circa 2002), are a feature that partitions kernel resources such that one set of processes sees one set of resources while another set of processes sees a different set of resources. Each namespace wraps a global system resource in an abstraction that makes it appear to the processes within the namespace that they have their own isolated instance of the global resource. 

Namespaces target the isolation of:
- **PID (Process ID) namespaces**: Isolate the process ID number space, meaning that processes in different PID namespaces can have the same PID.
- **Network namespaces**: Provide isolation of network controllers, IP address ports, etc., allowing systems to have multiple instances of network resources.
- **Mount namespaces**: Isolate filesystem mount points seen by a group of processes, so that one set of processes does not see the same filesystem view as another.
- **UTS (UNIX Time-sharing System) namespaces**: Isolate two system identifiers - the hostname and the NIS domain name. This is particularly useful in containers where each container needs to have its own hostname.
- **IPC (Inter-Process Communication) namespaces**: Separate inter-process communication between groups of processes.
- **User namespaces**: Isolate user IDs between different groups of processes, allowing a process to have a non-root user on the host but root inside the namespace.

### Usage in Containers
In the context of containerization, both cgroups and namespaces are essential:
- **Cgroups** manage the resources each container can use, ensuring that no single container can overuse resources at the expense of others.
- **Namespaces** ensure that each container only sees its own environment without interfering with or even being aware of other containers on the same host.

Together, cgroups and namespaces form the backbone of Linux container technologies, providing the isolation and resource control needed to securely and efficiently run multiple containers on a single Linux host.