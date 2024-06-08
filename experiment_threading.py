import time

from custom_threading import CustomThread


def print_hello(num):
    """Testing function

    Args:
        num (int): The number to print

    Returns:
        Str: The name
    """
    time.sleep(2)
    print("Hello", num)
    time.sleep(5)
    return f"Daniel {num}"


threads = []
for i in range(20):
    thread = CustomThread(target=print_hello, args=(i,))
    threads.append((thread, i))

num_threads_max = 5
num_threads_running = 0
running_threads = []
# Limit the number of threads running at the same time
while len(threads) > 0:
    if num_threads_running >= num_threads_max:
        # Wait until a thread finishes
        for thread in running_threads:  # pylint: disable=modified-iterating-list
            if not thread[0].is_alive():
                print(thread[0].join())
                running_threads.remove(thread)
                num_threads_running -= 1
    else:
        for i in range(num_threads_max - num_threads_running):
            thread = threads.pop(0)
            thread[0].start()
            num_threads_running += 1
            running_threads.append(thread)

# Wait for the remaining threads to finish
for thread in running_threads:
    print(thread[0].join())
