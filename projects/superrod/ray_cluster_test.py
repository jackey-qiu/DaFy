from collections import Counter
import os
import sys
import time
import ray
import psutil

redis_password = sys.argv[1]
num_cpus = int(sys.argv[2])

ray.init(address=os.environ["ip_head"], _redis_password=redis_password)

print("Nodes in the Ray cluster:")
print(ray.nodes())

print(ray.cluster_resources())

@ray.remote(num_cpus=1)
def f():
    print('hello')
    time.sleep(60)
    return ray._private.services.get_node_ip_address()

# The following takes one second (assuming that ray was able to access all of the allocated nodes).
for i in range(60):
    start = time.time()
    ip_addresses = ray.get([f.remote() for _ in range(60)])
    print(Counter(ip_addresses))
    end = time.time()
    print(end - start)
