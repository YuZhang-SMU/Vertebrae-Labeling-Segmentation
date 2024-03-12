import torch

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LargestComponent(data_batch, dtype=torch.float32, device=my_device): # torch.bool
    num, dim = data_batch.shape

    max_start = torch.zeros(num, dtype=dtype, device=device)
    max_end = torch.zeros(num, dtype=dtype, device=device)
    max_length = torch.zeros(num, dtype=dtype, device=device)
    current_start = -1 * torch.ones(num, dtype=dtype, device=device)
    current_length = torch.zeros(num, dtype=dtype, device=device)

    for i in range(dim):
        indices = data_batch[:, i] == 1
        current_length[indices] += 1
        current_start[(indices & (current_start == -1))] = i

        indices = data_batch[:, i] == 0
        update_indices = indices & (current_length > max_length)
        max_start[update_indices] = current_start[update_indices]
        max_end[update_indices] = i
        max_length[update_indices] = current_length[update_indices]
        current_start[indices] = -1
        current_length[indices] = 0
    update_indices = (current_length > max_length)
    max_start[update_indices] = current_start[update_indices]
    max_end[update_indices] = dim
    max_length[update_indices] = current_length[update_indices]
    return max_start.to(torch.int8), max_end.to(torch.int8), max_length.to(torch.int8)