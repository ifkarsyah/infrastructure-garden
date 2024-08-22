---
title: Infrastructure Engineer Learning Path
---
Step by step guide for DevOps, SRE or any other Operations Role in 2024:
https://roadmap.sh/devops

## Beginner
### Programming Language
Learn these [[programming-language]]. 
	- [[go]]
	- [[rust]]
	- [[c++]]
	- [[java]]
	- [[node.js]]
Focus on its concurrency features.

### Operating System

#### Concepts
* Learn OSTEP
* Learn How those concept implemented in Linux

#### Learn to Live in Terminal
* Basic Text Manipulation
	- [[cat]] - concatenate and print file
	- [[echo]] - displaying a line of text.
	- [[wc]] - printing newline, word, and byte counts for files
	- [[sort]] - sorting lines of text files.
	- [[cut]] - cutting sections from each line of files.
	- [[uniq]] - omitting repeated lines.
	- [[fmt]] - simple optimal text formatting.
- Advanced Text Manipulation
	- [[grep]] - searching plain-text data sets for lines that match a regular expression.
	- [[awk]] - programming language designed for text processing
	- [[sed]] - A stream editor for filtering and transforming text.
	- https://unix.stackexchange.com/questions/303044/when-to-use-grep-less-awk-sed
- Learn [[linux-io-redirection|Linux IO Redirection (>, >>, 2>&1)]]
* Learn VIM
	* https://vim-adventures.com/

#### Learn how to troubleshoot in Linux
* Process Monitoring
	* [[ps]] - report a snapshot of the current processes.
	- [[top]] - display Linux processes.
	- [[htop]] - interactive process viewer.
	- [[lsof]] - list open files.
* Performance Monitoring
	* [[nmon]] - system monitor tool for Linux and AIX systems.
	- [[iostat]] - reports CPU and IO statistics for devices, partitions and network filesystems.
	- [[sar]] - reports system loads, including CPU activity, memory/paging, device load, network
	- [[vmstat]] - reports virtual memory statistics

#### Reference
- https://linuxupskillchallenge.org/#table-of-contents
- https://linuxjourney.com/
- https://linuxcommand.org/tlcl.php

### Networking

#### Concepts
As a DevOps engineer you will need to understand the basics of networking protocols, how they work, and how they are used in the real world. To get you started, you should learn about, [TCP/IP](https://en.wikipedia.org/wiki/Internet_protocol_suite), [HTTP](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol), [HTTPS](https://en.wikipedia.org/wiki/HTTPS), [FTP](https://en.wikipedia.org/wiki/File_Transfer_Protocol), [SSH](https://en.wikipedia.org/wiki/Secure_Shell), [SMTP](https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol), [DNS](https://en.wikipedia.org/wiki/Domain_Name_System), [DHCP](https://en.wikipedia.org/wiki/Dynamic_Host_Configuration_Protocol), [NTP](https://en.wikipedia.org/wiki/Network_Time_Protocol).


#### Linux Networking Command
- [[ping]] - sends echo request packets to a host to test the Internet connection.
- [[traceroute]] - Traces the route taken by packets over an IP network.
- [[nmap]] - Scans hosts for open ports.
- [[netstat]] - displays network connections, routing tables, interface statistics, masquerade connections, and multicast memberships.
- [[ufw]] and `firewalld` - Firewall management tools.
- [[iptables]] and `nftables` - Firewall management tools.
- [[tcpdump]] - Dumps traffic on a network.
- [[dig]] - DNS lookup utility.

## Intermediate

### Container

#### Learn Container Concepts
Containers are a construct in which [cgroups](https://en.wikipedia.org/wiki/Cgroups), [namespaces](https://en.wikipedia.org/wiki/Linux_namespaces), and [chroot](https://en.wikipedia.org/wiki/Chroot) are used to fully encapsulate and isolate a process. This encapsulated process, called a container image, shares the kernel of the host with other containers.
- Container vs. Virtual Machine

#### Learn [[docker|Docker]]

Docker is a platform for working with containerized applications. Among its features are a daemon and client for managing and interacting with containers, registries for storing images, and a desktop application to package all these features together.

### What is and How to Setup X?

#### Load Balancer
Load balancing is the process of distributing traffic among multiple servers to improve a service or application's performance and reliability.
-  [What is Load Balancing?](https://www.nginx.com/resources/glossary/load-balancing/)
-  [Load Balancing concepts and algorithms](https://www.cloudflare.com/en-gb/learning/performance/what-is-load-balancing/)
#### Firewall
Firewall is a **network security device** that monitors and filters incoming and outgoing network traffic based on an organization’s previously established security policies.
- [Types of Firewall](https://www.cisco.com/c/en_in/products/security/firewalls/what-is-a-firewall.html)
- [Why do we need Firewalls?](https://www.tutorialspoint.com/what-is-a-firewall-and-why-do-you-need-one)
- [Firewalls and Network Security - SimpliLearn](https://www.youtube.com/watch?v=9GZlVOafYTg)
#### Caching
A cache server is a **dedicated network server** or service acting as a server that saves Web pages or other Internet content locally.
- [What is Caching?](https://www.cloudflare.com/en-gb/learning/cdn/what-is-caching/)
- [What is Cache Server?](https://networkencyclopedia.com/cache-server/)
- [Site Cache vs Browser Cache vs Server Cache](https://wp-rocket.me/blog/different-types-of-caching/)
#### Forward Proxy

Forward Proxy, often called proxy server is a server that sits in front of a group of **client machines**. When those computers make requests to sites and services on the Internet, the proxy server intercepts those requests and then communicates with web servers on behalf of those clients, like a middleman.
**Common Uses:**
- To block access to certain content
- To protect client identity online
- To provide restricted internet to organizations
#### Reverse Proxy
A Reverse Proxy server is a type of proxy server that typically sits behind the firewall in a private network and directs client requests to the appropriate backend server. It provides an additional level of security by hiding the server related details like `IP Address` to clients. It is also known as **server side proxy**.

**Common Uses:**
- Load balancing
- Web acceleration
- Security and anonymity
Reference
- [What is Reverse Proxy?](https://www.cloudflare.com/en-gb/learning/cdn/glossary/reverse-proxy/)
#### Web Server

##### NginX
NGINX is a powerful web server and uses a non-threaded, event-driven architecture that enables it to outperform Apache if configured correctly. It can also do other important things, such as load balancing, HTTP caching, or be used as a reverse proxy.
##### Caddy
Caddy is an open-source web server with automatic HTTPS written in Go. It is easy to configure and use, and it is a great choice for small to medium-sized projects.

##### Apache
Apache is a free, open-source HTTP server, available on many operating systems, but mainly used on Linux distributions. It is one of the most popular options for web developers, as it accounts for over 30% of all the websites

### Cloud Provider
Cloud providers provide a layer of APIs to abstract infrastructure and provision it based on security and billing boundaries. The cloud runs on servers in data centers, but the abstractions cleverly give the appearance of interacting with a single “platform” or large application.
#### AWS
Amazon Web Services has been the market leading cloud computing platform since 2011, ahead of Azure and Google Cloud. AWS offers over 200 services with data centers located all over the globe.
#### GCP
Google Cloud is Google’s cloud computing service offering, providing over 150 products/services to choose from.
#### Azure
Microsoft Azure is a cloud computing service operated by Microsoft. Azure currently provides more than 200 products and cloud services.

#### AlibabaCloud
Alibaba Cloud is a cloud computing service, offering over 100 products and services with data centers in 24 regions and 74 availability zones around the world.

### Serverless
- Cloudflare
- AWS Lambda
- Vercel
- Netlify
## IaaC
### Provisioning Tools

#### [[terraform|Terraform]]
Terraform is an extremely popular open source Infrastructure as Code (IaC) tool that can be used with many different cloud and service provider APIs. Terraform focuses on an immutable approach to infrastructure, with a terraform state file center to tracking the status of your real world infrastructure.

### Configuration Management
#### [[ansible|Ansible]]
Ansible is an open-source configuration management, application deployment and provisioning tool that uses its own declarative language in YAML. Ansible is agentless, meaning you only need remote connections via SSH
#### [[puppet|Puppet]]
Puppet, an automated administrative engine for your Linux, Unix, and Windows systems, performs administrative tasks (such as adding users, installing packages, and updating server configurations) based on a centralized specification.
### [[ci-cd|CI/CD]]
CI/CD is a method to frequently deliver apps to customers by introducing automation into the stages of app development. The main concepts attributed to CI/CD are continuous integration, continuous delivery, and continuous deployment
#### [[github-action|GitHub Action]]
ou can discover, create, and share actions to perform any job you’d like, including CI/CD, and combine actions in a completely customized workflow.
#### [[gitlab-ci|Gitlab CI]]
GitLab offers a CI/CD service that can be used as a SaaS offering or self-managed using your own resources.
#### [[jenkins|Jenkins]]
Jenkins is an open-source CI/CD automation server. Jenkins is primarily used for building projects, running tests, static code analysis and deployments.
### Logs Management
Log management is the process of handling log events generated by all software applications and infrastructure on which they run. It involves log collection, aggregation, parsing, storage, analysis, search, archiving, and disposal, with the ultimate goal of using the data for troubleshooting and gaining business insights
#### Elastic Stack
Elastic Stack is a group of open source products comprised of
- `Elastic Search` - Search and analytics engine
- `Logstash/fluentd` - Data processing pipeline
- `Kibana` - Dashboard to visualize data
### Secret Management
Secret management is an important aspect of DevOps, as it involves securely storing and managing sensitive information, such as passwords, API keys, and other secrets, that are used by applications and infrastructure.

#### [[vault|Vault]]
Vault is a tool for securely storing and managing secrets, such as passwords, API keys, and other sensitive information. It is developed and maintained by Hashicorp and is available as open-source software.

#### [[sealed-secret|Sealed Secret]]
Sealed Secrets is a tool for securely storing and managing secrets in a Kubernetes environment. It is developed and maintained by Bitnami and is available as open-source software.

## Kubernetes World
Kubernetes is an [open source](https://github.com/kubernetes/kubernetes) container management platform, and the dominant product in this space. Using Kubernetes, teams can deploy images across multiple underlying hosts, defining their desired availability, deployment logic, and scaling logic in YAML.

### Kubernetes Platform
- GKS
- EKS
- AKS
### Templating
- Kustomize
- Helm
### Service Mesh
- Istio
- Consul
- Linkerd
- Envoy
### Observability
#### Logging
- FluentD and Fluent-Bit
- Loki
#### Monitoring
- Prometheus
- Grafana
#### Tracing
- Jaeger
- OpenTelemetry
### GitOps
- ArgoCD
- FluxCD
