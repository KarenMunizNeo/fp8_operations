import torch
import transformer_engine as te
import transformer_engine.pytorch as te_pytorch
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format = fp8_format, amax_history_len = 16, amax_compute_algo ="max")

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

def print_memory(label=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e6  # MB
    reserved = torch.cuda.memory_reserved() / 1e6   # MB
    print(f"[{label}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def main():
    # Print the Transformer Engine version.
    print("Transformer Engine Version:", te.__version__)

    torch.manual_seed(12)
   
     # Create dummy matrices for matmul (C = A @ B)
    A = torch.rand((768, 1024), device="cuda", dtype=torch.float32)
    B = torch.rand((1024, 768), device="cuda", dtype=torch.float32)

    # Wrap B as weights in a Linear layer to simulate matmul
    linear_layer = te_pytorch.Linear(1024, 768, bias=False).cuda()
    with torch.no_grad():
        linear_layer.weight.copy_(B.T)

    print_memory("Before FP8 simulated matmul")

    start_event.record()

    #Autocasts input values to FP8 using Te an performs matmul
    with te_pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        C = linear_layer(A)
    
    end_event.record()


    elapsed_time = start_event.elapsed_time(end_event)
    print("Tempo de execução do Matmul: {:.3f} ms".format(elapsed_time))

    print_memory("After FP8 simulated matmul")
    print("Result (C = A @ B):")
    print(C.cpu().detach())
    print("C dtype:", C.dtype)


if __name__ == "__main__":
    main()
