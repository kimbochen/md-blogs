module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    ; %arg0: x_ptr, %arg1: y_ptr, %arg2: output_ptr, %arg3: n_elements

    %c1024_i32 = arith.constant 1024 : i32  ; BLOCK_SIZE = 1024

    %0 = tt.get_program_id x : i32          ; pid = tl.program_id(axis=0)

    %1 = arith.muli %0, %c1024_i32 : i32    ; block_start = pid * BLOCK_SIZE

    ; tl.arange(0, BLOCK_SIZE)
    ; The second line is the tensor layout for tl.arange(0, BLOCK_SIZE)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} :
    tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Broadcasts `block_start` shape to match tl.arange(0, BLOCK_SIZE)
    ; splat is an MLIR tensor dialect operation that broadcasts tensor shapes
    ; Source: https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorsplat-tensorsplatop
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; offsets = block_start + tl.arange(0, BLOCK_SIZE)
    %4 = arith.addi %3, %2 : tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Expand `n_elements` dimensions
    %5 = tt.splat %arg3 : (i32) ->
    tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; mask = offsets < n_elements
    ; cmpi is an MLIR arith dialect integer comparison operation. Predicate 2 is slt (signed less than)
    ; Source: https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop
    %6 = "triton_gpu.cmpi"(%4, %5) <{predicate = 2 : i64}> : 
      (
       tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>,
       tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>
      ) -> tensor<1024xi1, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Broadcasts `x_ptr` to a tensor of pointers with shape like `offsets`
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) ->
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Compute `x_ptr + offsets`
    %8 = tt.addptr %7, %4 :
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>,
      tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; x = tl.load(%8, mask=mask)
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} :
      tensor<1024xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Broadcasts `y_ptr` to a tensor of pointers with shape like `offsets`
    %10 = tt.splat %arg1 : (!tt.ptr<f32>) ->
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Compute `y_ptr + offsets`
    %11 = tt.addptr %10, %4 :
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>,
      tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; y = tl.load(%11, mask=mask)
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} :
      tensor<1024xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; output = x + y
    %13 = arith.addf %9, %12 : tensor<1024xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Broadcasts `output_ptr` to a tensor of pointers with shape like `offsets`
    %14 = tt.splat %arg2 : (!tt.ptr<f32>) ->
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; Compute `output_ptr + offsets`
    %15 = tt.addptr %14, %4 :
      tensor<1024x!tt.ptr<f32>, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>,
      tensor<1024xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    ; tl.store(output_ptr + offsets, output, mask=mask)
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} :
      tensor<1024xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>

    tt.return
  }
}
