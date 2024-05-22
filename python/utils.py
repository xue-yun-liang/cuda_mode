import torch
import time


def show_time(func):
    ntest = 10
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    torch.cuda.empty_cache()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res