using SIMD
using Test, InteractiveUtils

using Base: setindex

"""
    llvm_ir(f, args) :: String

Get LLVM IR of `f(args...)` as a string.
"""
llvm_ir(f, args) = sprint(code_llvm, f, Base.typesof(args...))

# The vector we are testing.
const global nbytes = 32

const global L8 = nbytes ÷ 4
const global L4 = nbytes ÷ 8

const global V8I32 = Vec{L8,Int32}
const global V8I64 = Vec{L8,Int64}
const global V4F64 = Vec{L4,Float64}

const global v8i32 = ntuple(i -> Int32(ifelse(isodd(i), i, -i)), L8)
const global v8i64 = ntuple(i -> Int64(ifelse(isodd(i), i, -i)), L8)
const global v4f64 = ntuple(i -> Float64(ifelse(isodd(i), i, -i)), L4)

global const arri32 = valloc(Int32, L8, 2*L8) do i i end
global const arrf64 = valloc(Float64, L4, 4*L4) do i i end

is_checking_bounds = Core.Compiler.inbounds_option() == :on

include("test_SIMD.jl")

VERSION >= v"1.9" && include("test_SLEEF_Ext.jl")
