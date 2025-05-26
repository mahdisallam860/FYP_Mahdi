import torch
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a large tensor and move it to the GPU
tensor_size = 10**7  # Adjust the size as needed
a = torch.rand(tensor_size, device=device)
b = torch.rand(tensor_size, device=device)

# Perform a computation
start_time = time.time()
c = a + b
torch.cuda.synchronize()  # Wait for all kernels to finish
end_time = time.time()

print(f"Computation completed in {end_time - start_time:.4f} seconds")
