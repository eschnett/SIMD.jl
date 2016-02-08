# SIMD

Explicit SIMD vectorization in Julia

[![Build Status](https://travis-ci.org/eschnett/SIMD.jl.svg?branch=master)](https://travis-ci.org/eschnett/SIMD.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/xwaa3hm5wkiqrc54/branch/master?svg=true)](https://ci.appveyor.com/project/eschnett/simd-jl/branch/master)
[![codecov.io](https://codecov.io/github/eschnett/SIMD.jl/coverage.svg?branch=master)](https://codecov.io/github/eschnett/SIMD.jl?branch=master)

## Overview

This package allows programmers to explicitly SIMD-vectorize their Julia code. Ideally, the compiler (Julia and LLVM) would be able to do this automatically, especially for straightforwardly written code. In practice, this does not always work (for a variety of reasons), and the programmer is often left with uncertainty as to whether the code was actually vectorized. It is usually necessary to look at the generated machine code to determine whether the compiler actually vectorized the code.

By exposing SIMD vector types and corresponding operations, the programmer can explicitly vectorize their code. While this does not guaratee that the generated machine code is efficient, it relieves the compiler from determining whether it is legal to vectorize the code, deciding whether it is beneficial to do so, and rearranging the code to synthesize vector instructions.

Here is a simple example for a manually vectorized code that adds two arrays:
```Julia
using SIMD
function vadd!{N,T}(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}})
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        vstore(xv, xs, i)
    end
end
```
To simplify this example code, the vector type that should be used (`Vec{N,T}`) is passed in explicitly as additional type argument. This routine is e.g. called as `vadd!(xs, ys, Vec{8,Float64})`.

## SIMD vector operations

SIMD vectors are similar to small fixed-size arrays of "simple" types. These element types are supported:

`Bool Int{8,16,32,64,128} UInt{8,16,32,64,128} Float{16,32,64}`

The SIMD package provides the usual arithmetic and logical operations for SIMD vectors:

`+ - * / % ^ ! ~ & | $ << >> >>> == != < <= > >=`

`abs cbrt ceil copysign cos div exp exp10 exp2 flipsign floor fma ifelse inv isfinite isinf isnan issubnormal log log10 log2 muladd rem round sign signbit sin sqrt trunc`

(Currently missing: `count_ones count_zeros exponent ldexp leading_ones leading_zeros significand trailing_ones trailing_zeros`, many trigonometric functions)

(Also currently missing: Type conversions, reinterpretation that changes the vector size, vector shuffles, scatter/gather operations, masked load/store operations)

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
ys = setindex(ys, 3, 5)   # cannot use ys[3] = 5
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
arr = Vector{Float64}(100)
...
xs = vload(Vec{4,Float64}, arr, i)
...
vstore(xs, arr, i)
```
The `vload` call reads a vector of size 4 from the array, i.e. it reads `arr[i:i+3]`. Similarly, the `vstore` call writes the vector `xs` to the four array elements `arr[i:i+3]`.

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

This code would work if Julia supported a datatype (here called `Float64x4`) that is internally represented as LLVM vector. Unfortunately, there is no such type, so that we need to employ a work-around.

### Representing SIMD vectors as Julia tuples

Julia tuples (if all elements have the same type) are represented as LLVM arrays. LLVM arrays and vectors have similar representations, and it is straightforward to translate between them.

Currently, the SIMD package represents vectors as tuples, declared as
```Julia
immutable Vec{N,T} <: DenseArray{N,1}
    elts::NTuple{N,T}
end
```

### Possible alternative representations

(Julia's optimizer used to be deficient, and the generated vector code was inefficient. With a recent patch to LLVM this seems to have been remedied, and this section is now mostly of historic interest.)

I experimented with representing SIMD vectors as `bitstype` types in Julia. These become integers of the corresponding size in LLVM. For example, `Vec{4,Float32}` would become a 128-bit integer `i128` in LLVM.

Since bitstypes are not parameteric in Julia (unlike `NTuple`), this is slightly inconvenient as one has to create the respective bitstypes as necessary, essentially simulating "generated type" (akin "generated functions"). Apart from this, LLVM has certain generation issues -- there are missed optimizations that lead to  convoluted code. (On x86-64, values are moved between vector and scalar registers, which is a very slow operation.)

Another alternative would be to introduce a new, `NTuple`-like type to Julia that is translated to LLVM vectors instead of LLVM arrays. This would certainly be the cleanest approach from LLVM's perspective. Unfortunately, I do not know whether this would be feasible, or how much work this would be. I welcome feedback in this respect.
