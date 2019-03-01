# SIMD

Explicit SIMD vectorization in Julia

[![Build Status](https://travis-ci.org/eschnett/SIMD.jl.svg?branch=master)](https://travis-ci.org/eschnett/SIMD.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/xwaa3hm5wkiqrc54/branch/master?svg=true)](https://ci.appveyor.com/project/eschnett/simd-jl/branch/master)
[![codecov.io](https://codecov.io/github/eschnett/SIMD.jl/coverage.svg?branch=master)](https://codecov.io/github/eschnett/SIMD.jl?branch=master)
[![Dependency Status](https://dependencyci.com/github/eschnett/SIMD.jl/badge)](https://dependencyci.com/github/eschnett/SIMD.jl)
[![DOI](https://zenodo.org/badge/50004500.svg)](https://zenodo.org/badge/latestdoi/50004500)

## Overview

This package allows programmers to explicitly SIMD-vectorize their Julia code. Ideally, the compiler (Julia and LLVM) would be able to do this automatically, especially for straightforwardly written code. In practice, this does not always work (for a variety of reasons), and the programmer is often left with uncertainty as to whether the code was actually vectorized. It is usually necessary to look at the generated machine code to determine whether the compiler actually vectorized the code.

By exposing SIMD vector types and corresponding operations, the programmer can explicitly vectorize their code. While this does not guarantee that the generated machine code is efficient, it relieves the compiler from determining whether it is legal to vectorize the code, deciding whether it is beneficial to do so, and rearranging the code to synthesize vector instructions.

Here is a simple example for a manually vectorized code that adds two arrays:
```Julia
using SIMD
function vadd!(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}}) where {N, T}
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    lane = VecRange{N}(0)
    @inbounds for i in 1:N:length(xs)
        xs[lane + i] += ys[lane + i]
    end
end
```
To simplify this example code, the vector type that should be used (`Vec{N,T}`) is passed in explicitly as additional type argument. This routine is e.g. called as `vadd!(xs, ys, Vec{8,Float64})`.

## SIMD vector operations

SIMD vectors are similar to small fixed-size arrays of "simple" types. These element types are supported:

`Bool Int{8,16,32,64,128} UInt{8,16,32,64,128} Float{16,32,64}`

The SIMD package provides the usual arithmetic and logical operations for SIMD vectors:

`+ - * / % ^ ! ~ & | $ << >> >>> == != < <= > >=`

`abs cbrt ceil copysign cos div exp exp10 exp2 flipsign floor fma inv isfinite isinf isnan issubnormal log log10 log2 muladd rem round sign signbit sin sqrt trunc vifelse`

(Currently missing: `count_ones count_zeros exponent ldexp leading_ones leading_zeros significand trailing_ones trailing_zeros`, many trigonometric functions)

(Also currently missing: Type conversions, reinterpretation that changes the vector size)

These operators and functions are always applied element-wise, i.e. they are applied to each element in parallel, yielding again a SIMD vector as result. This means that e.g. multiplying two vectors yields a vector, and comparing two vectors yields a vector of booleans. This behaviour might seem strange and slightly unusual, but corresponds to the machine instructions provided by the hardware. It is also what is usually needed to vectorize loops.

The SIMD package also provides conversion operators from scalars and tuples to SIMD vectors and from SIMD vectors to tuples. Additionally, there are `getindex` and `setindex` functions to access individual vector elements.  SIMD vectors are immutable (like tuples), and `setindex` (note there is no exclamation mark at the end of the name) thus returns the modified vector.
```Julia
# Create a vector where all elements are Float64(1):
xs = Vec{4,Float64}(1)

# Create a vector from a tuple, and convert it back to a tuple:
ys = Vec{4,Float32}((1,2,3,4))
ys1 = NTuple{4,Float32}(ys)
y2 = ys[2]   # getindex

# Update one element of a vector:
ys = setindex(ys, 5, 3)   # cannot use ys[3] = 5
```

## Reduction operations

Reduction operations reduce a SIMD vector to a scalar. The following reduction operations are provided:

`all any maximum minimum sum prod`

Example:
```Julia
v = Vec{4,Float64}((1,2,3,4))
sum(v)
10.0
```

## Accessing arrays

When using explicit SIMD vectorization, it is convenient to allocate arrays still as arrays of scalars, not as arrays of vectors. The `vload` and `vstore` functions allow reading vectors from and writing vectors into arrays, accessing several contiguous array elements.

```Julia
arr = Vector{Float64}(undef, 100)
...
xs = vload(Vec{4,Float64}, arr, i)
...
vstore(xs, arr, i)
```
The `vload` call reads a vector of size 4 from the array, i.e. it reads `arr[i:i+3]`. Similarly, the `vstore` call writes the vector `xs` to the four array elements `arr[i:i+3]`.

When the values to be read are stored in non-contiguous locations, the `vgather` function can be used to load them into a vector (so-called gather operation).

```Julia
idx = Vec((1, 5, 6, 9))
vgather(arr, idx)
```

Likewise, storing to non-contiguous locations (scatter) can be done by the `vscatter` function.

```Julia
arr = zeros(10)
v = Vec((1.0, 2.0, 3.0, 4.0))
idx = Vec((1, 3, 4, 7))
vscatter(v, arr, idx)
```

Above `vload`, `vstore`, `vgather` and `vscatter` can be written using the indexing notation:

```Julia
i = 1
lane = VecRange{4}(0)
v = arr[lane + i]             # vload
arr[lane + i] = v             # vstore
idx = Vec((1, 3, 4, 7))
v = arr[idx]                  # vgather
arr[idx] = v                  # vscatter
```

## Vector shuffles

Vector shuffle is available through the `shufflevector` function.

Example:
```Julia
a = Vec{4, Int32}((1,2,3,4))
b = Vec{4, Int32}((5,6,7,8))
mask = (2,3,4,5)
shufflevector(a, b, Val{mask})
<4 x Int32>[3, 4, 5, 6]
```
The mask specifies vector elements counted across `a` and `b`,
starting at 0 to follow the LLVM convention. If you don't care about
some of the values in the result vector, you can use the symbol
`:undef`. `a` and `b` must be of the same SIMD vector type. The
result will be a SIMD vector with the same element type as `a` and `b`
and the same length as the mask. The function must be specialized on
the value of the mask, therefore the `Val{}` construction in the call.

There is also a one operand version of the function:
```Julia
a = Vec{4, Int32}((1,2,3,4))
mask = (0,3,1,2)
shufflevector(a, Val{mask})
<4 x Int32>[1, 4, 2, 3]
```

## Representing SIMD vector types in Julia

In LLVM, SIMD vectors are represented via a special vector type. LLVM supports vectors of all "primitive" types, i.e. integers (including booleans), floating-point numbers, and pointers. LLVM directly provides arithmetic and logic operations (add, subtract, bit shift, select, etc.) for vector types. For example, adding two numbers is represented in LLVM as
```LLVM
%res = fadd <double> %arg1, %arg2
```
and adding two vectors looks like
```LLVM
%res = fadd <4 x double> %arg1, %arg2
```

Thus, implementing SIMD operations in Julia is in principle a straightforward application of `llvmcall`. In principle, this should work:
```Julia
function +(x::Float64x4, y::Float64x4)
    llvmcall("""
        %res = fadd <4 x double> %0, %1
        return <4 x double> %res
        """, Float64x4, Tuple{Float64x4, Float64x4}, x, y)
end
```

The Julia representation of the datatype `Float64x4` is slightly
complex: It is an `NTuple{N,T}`, where the element type `T` is
specially marked by being wrapped in the type `Base.VecElement`:
`NTuple{4, Base.VecElement{Float64}}`. Julia implements a special rule
that translates tuples with element type `Base.VecElement` into LLVM
vectors. Other tuples are translated into LLVM arrays if all tuple
elements have the same type, otherwise into LLVM structures.

This representation has two drawbacks. First, it is rather tedious.
Second, while we want to define arithmetic operations for SIMD
vectors, we do not want to define arithmetic for Julia's tuple types
-- if we defined additional methods for generic tuples, who knows what
code would break as a result.

We thus define our own SIMD vector type `Vec{N,T}`:
```Julia
struct Vec{N,T}
    elts::NTuple{N,VecElement{T}}
end
```
