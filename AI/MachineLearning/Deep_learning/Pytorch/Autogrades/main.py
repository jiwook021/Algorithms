import torch

def autograd_tutorial():
    """
    Demonstrates PyTorch's automatic differentiation capability (autograd).
    
    Autograd is PyTorch's automatic differentiation engine that powers
    neural network training. It calculates gradients automatically.
    
    Time Complexity: O(n) for forward and backward passes where n is the number
                    of operations in the computational graph
    Memory Complexity: O(n) for storing the computational graph
    """
    print("===== AUTOMATIC DIFFERENTIATION WITH AUTOGRAD =====")
    
    # 1. Creating tensors with requires_grad=True to track operations
    x = torch.ones(2, 2, requires_grad=True)
    print(f"Input tensor x: \n{x}")
    print(f"Requires gradient: {x.requires_grad}")
    
    # 2. Perform operations on the tensor
    y = x + 2
    print(f"\nIntermediate tensor y = x + 2: \n{y}")
    print(f"Requires gradient: {y.requires_grad}")
    print(f"y's grad_fn: {y.grad_fn}")  # Shows the operation that created this tensor
    
    # More complex operations
    z = y * y * 3
    out = z.mean()
    print(f"\nMore operations: z = 3 * y * y: \n{z}")
    print(f"Output tensor (mean of z): {out}")
    print(f"z's grad_fn: {z.grad_fn}")
    print(f"out's grad_fn: {out.grad_fn}")
    
    # 3. Compute gradients with backward pass
    print("\n===== COMPUTING GRADIENTS =====")
    out.backward()  # Equivalent to out.backward(torch.tensor(1.0))
    
    # After backward(), x.grad contains the gradient of out with respect to x
    print(f"Gradient of out with respect to x (∂out/∂x): \n{x.grad}")
    
    # Mathematical explanation
    print("\nMathematical explanation:")
    print("out = mean(3*(x+2)²)")
    print("∂out/∂x = 3*2*(x+2)/4 = 3*(x+2)/2 = 3/2 * (x+2)")
    print("With x = 1, ∂out/∂x = 3/2 * 3 = 4.5")
    
    # 4. Detaching tensors from computational graph
    print("\n===== DETACHING FROM COMPUTATIONAL GRAPH =====")
    x = torch.randn(3, requires_grad=True)
    print(f"New tensor x: {x}")
    
    # Create y without tracking history
    y = x * 2
    print(f"y = x * 2: {y}")
    
    # Detach y from computational graph
    y_detached = y.detach()
    print(f"y_detached: {y_detached}")
    print(f"y requires_grad: {y.requires_grad}")
    print(f"y_detached requires_grad: {y_detached.requires_grad}")
    
    # 5. Controlling gradient calculation with no_grad
    print("\n===== USING NO_GRAD CONTEXT =====")
    x = torch.randn(3, requires_grad=True)
    print(f"Tensor x: {x}")
    
    # Operations inside torch.no_grad() won't track gradients
    with torch.no_grad():
        y = x * 2
        print(f"y = x * 2 (inside no_grad): {y}")
        print(f"y requires_grad: {y.requires_grad}")
    
    # 6. Real-world example: Computing gradients for a simple function
    print("\n===== EXAMPLE: GRADIENT OF f(x) = x^2 =====")
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x = {x.item()}")
    
    # Forward pass: compute y = x^2
    y = x ** 2
    print(f"y = x^2 = {y.item()}")
    
    # Backward pass: compute dy/dx = 2x
    y.backward()
    print(f"dy/dx at x = {x.item()} is {x.grad.item()}")  # Should be 2 * x = 4
    
    # 7. Computing gradients for vector-valued functions
    print("\n===== VECTOR-VALUED FUNCTIONS =====")
    x = torch.randn(3, requires_grad=True)
    print(f"Vector x: {x}")
    
    # Function: y = x^2
    y = x ** 2
    print(f"y = x^2: {y}")
    
    # Create an external gradient to pass to backward
    # For vector-valued functions, we need to provide the gradient of the output
    # with respect to each output element
    external_grad = torch.tensor([1.0, 1.0, 1.0])
    
    # Compute the gradients
    y.backward(external_grad)
    print(f"Gradients (dx/dy): {x.grad}")  # Should be 2 * x
    
    # 8. Tips for using autograd in neural networks
    print("\n===== TIPS FOR USING AUTOGRAD =====")
    print("1. Zero gradients before backward(): optimizer.zero_grad() or x.grad.zero_()")
    print("2. Avoid in-place operations on tensors with requires_grad=True")
    print("3. Use no_grad() for inference to save memory and computation")
    print("4. For complex neural networks, PyTorch's nn module handles most autograd details")
    
    return "Autograd tutorial completed!"

# Run the tutorial
if __name__ == "__main__":
    autograd_tutorial()