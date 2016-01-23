# SIMD

Explicit SIMD vectorization in Julia

[![Build Status](https://travis-ci.org/eschnett/SIMD.jl.svg?branch=master)](https://travis-ci.org/eschnett/SIMD.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/xwaa3hm5wkiqrc54/branch/master?svg=true)](https://ci.appveyor.com/project/eschnett/simd-jl/branch/master)
[![codecov.io](https://codecov.io/github/eschnett/SIMD.jl/coverage.svg?branch=master)](https://codecov.io/github/eschnett/SIMD.jl?branch=master)

## Overview

This package allows programmers to explicitly SIMD-vectorize their Julia code. Ideally, the compiler (Julia and LLVM) should be able to do this automatically, especially for straightforwardly written code. In practice, this does not always work (for a variety of reasons), and the programmer is often left with some uncertainty as to whether the code was actually vectorized. It is usually necessary to look at the generated machine code to determine whether the compiler actually vectorized the code.

By exposing SIMD vector types and corresponding operations, the programmer can explicitly vectorize their code. While this does not guaratee that the generated machine code is efficient, it relieves the compiler from determining whether it is legal to vectorize the code, deciding whether it is beneficial to do so, and rearranging the code to synthesize vector instructions.

Here is a simple example for a manually vectorized code to add two arrays:
```Julia
using SIMD
function vadd!{T}(xs::Vector{T}, ys::Vector{T})
    @assert length(ys) == length(xs)
    N = 2   # SIMD vector length
    @assert length(xs) % N == 0
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        vstore(xv, xs, i)
    end
end
```

## SIMD vector operations

The SIMD package provides the usual arithmetic and logical operations for SIMD vectors:

`+ - * / % ~ & | $ << >> >>> == != < <= > >=`

`abs div ifelse inv muladd sqrt`

Others could (and should) be added.

These operators and functions are always applied element-wise, i.e. they are applied to each element in parallel, giving again a SIMD vector as result. This means that e.g. multiplying two vectors yields a vector, and comparing two vectors yields a boolean vector. This behaviour might seem strange and slightly unusual, but corresponds to the machine instructions provided by the hardware.

The SIMD package also provides conversion operators from scalars and tuples to SIMD vectors and from SIMD vectors to tuples. Additionally, there are `getindex` and `setindex` functions to access individual vector elements. Since SIMD vectors are immutable, `setindex` (note there is no exclamation mark at the end of the name) returns the modified vector:
```Julia
xs = Vec{2,Float64}(1)
ys = Vec{4,Float32}((1,2,3,4))
ys1 = NTuple{4,Float32}(ys)
y2 = ys[2]   # getindex
zs = setindex(ys, 4, 5)   # can't say ys[4] = 5
```

### Accessing arrays

When using explicit SIMD vectorization, it is convenient to allocate arrays still as arrays of scalars, not arrays of vectors. The `vload` and `vstore` functions allow reading vectors from and writing vectors into arrays, accessing several contiguous array elements simultaneously.

```Julia
arr = Vector{Float64}(100)
...
xs = vload(Vec{2,Float64}, arr, i)
...
vstore(xs, arr, i)
```
The `vload` call reads a vector of size 2 from the array, i.e. it reads `arr[i:i+1]`. Similarly, the `vstore` call writes the vector `xs` to the two array elements `arr[i:i+1]`.

## Representing SIMD vector types in Julia

In LLVM, SIMD vectors are represented via a special vector type. LLVM supports vectors of all "primitive" types, i.e. integers, floating-point numbers, and pointers. LLVM directly provides arithmetic and logic operations (add, subtract, bit shift, select, etc.) for vector types. For example, adding two numbers is represented in LLVM as
```LLVM
%res = fadd <double> %arg1, %arg2
```
and adding two vectors looks like
```LLVM
%res = fadd <2 x double> %arg1, %arg2
```

Thus, implementing SIMD operations in Julia is in principle a straightforward application of `llvmcall`. In principle, this should work:
```Julia
function +(x::Float64x2, y::Float64x2)
    llvmcall("""
        %res = fadd <2 x double> %0, %1
        return <2 x double> %res
        """, Float64x2, Tuple{Float64x2, Float64x2}, x, y)
end
```

This code would work if Julia supported a datatype (here called `Float64x2`) that is represented as LLVM vector. Unfortunately, there is no such type, so that we need to employ a work-around.

### Representing SIMD vectors as Julia tuples

Julia tuples (if all elements have the same type) are represented as LLVM arrays. LLVM arrays and vectors have similar representations, and it is straightforward to translate between them.

Currently, the SIMD package represents vectors as tuples, declared as
```Julia
immutable Vec{N,T}
    elts::NTuple{N,T}
end
```

Unfortunately, LLVM's optimizer is not quite clever enough to handle the conversion between vectors and arrays efficiently. While converting an array to a vector is effectively optimized away (resulting in zero machine instructions), converting a vector to an array currently leads to convoluted stretches of machine instructions. There is no reason for this -- this seems to be case of a missed optimization that could easily be remedied by someone with the respective knowledge of LLVM.

### Alternative representations

I experimented with representing SIMD vectors as `bitstype` types in Julia. These become integers of the corresponding size in LLVM. For example, `Vec{4,Float32}` would become a 128-bit integer `i128` in LLVM.

Since bitstypes are not parameteric in Julia (unlike `NTuple`), this is slightly inconvenient as one has to create the respective bitstypes as necessary, essentially simulating "generated type" (akin "generated functions"). Apart from this, LLVM has the same code generation issues -- there are missed optimizations that lead to even more convoluted code. (On x86-64, values are moved between vector and scalar registers, which is likely even slower than moving values between vector registers.)

An alternative would be to introduce a new, `NTuple`-like type to Julia that is translated to LLVM vectors instead of LLVM arrays. This would certainly be the cleanest approach from LLVM's perspective. Unfortunately, I do not know whether thsi would be feasible, or how much work this would be. I welcome feedback in this respect.
