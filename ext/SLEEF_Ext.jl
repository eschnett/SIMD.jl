module SLEEF_Ext

# Vectorized mathematical functions

# This module exports nothing, its purpose is to specialize
# mathematical functions in Base and Base.FastMath for SIMD.Vec arguments
# using vectorized implementations from SLEEFPirates

import SLEEFPirates as SP
import Base.FastMath as FM
import VectorizationBase as VB
import SIMD

# Since SLEEFPirates works with VB.Vec but not with SIMD.Vec,
# we convert between SIMD.Vec and VB.Vec.
# However constructing a VB.Vec of length exceeding the native vector length
# returns a VB.VecUnroll => we must handle also this type

# Constructors SIMD.Vec and VB.Vec accept x... as arguments where x is iterable
# so we make SIMD.Vec and VB.VecUnroll iterable (VB.Vec can be converted to Tuple).
# To avoid messing up existing behavior of Base.iterate for SIMD and VB types, we define a wrapper type Iter{V}

struct Iter{V}
    vec::V
end
@inline Base.iterate(v::Iter, args...) = iter(v.vec, args...)

# iterate over SIMD.Vec
@inline iter(v::SIMD.Vec) = v[1], 2
@inline iter(v::SIMD.Vec{N}, i) where {N} = (i > N ? nothing : (v[i], i + 1))

# iterate over VB.VecUnroll
@inline function iter(v::VB.VecUnroll)
    data = VB.data(v)
    return data[1](1), (1, 1)
end
@inline function iter(v::VB.VecUnroll{N,W}, (i, j)) where {N,W}
    data = VB.data(v)
    if j < W
        return data[i](j + 1), (i, j + 1)
    elseif i <= N # there are N+1 vectors
        return data[i+1](1), (i + 1, 1)
    else
        return nothing
    end
end

@inline SIMDVec(v::VB.Vec) = SIMD.Vec(Tuple(v)...)
@inline SIMDVec(vu::VB.VecUnroll) = SIMD.Vec(Iter(vu)...)
@inline VBVec(v::SIMD.Vec) = VB.Vec(Iter(v)...)

# some operators have a fast version in FastMath, but not all
# and some operators have a fast version in SP, but not all !
const not_unops = (:eval, :include, :evalpoly, :hypot, :ldexp, :sincos)
unop(n) = !occursin("#", string(n)) && !in(n, not_unops)

const unops_SP = filter(unop, names(SP; all = true))
const unops_FM = filter(unop, names(FM; all = true))

# "slow" operators provided by SP
const unops_Base_SP = intersect(unops_SP, names(Base))
# FastMath operators provided by SP
const unops_FM_SP = intersect(unops_SP, unops_FM)
# FastMath operators with only a slow version provided by SP
const unops_FM_SP_slow = filter(unops_SP) do op
    n = Symbol(op, :_fast)
    in(n, unops_FM) && !in(n, unops_SP)
end

const vec = SIMD.Vec{<:Any,<:Union{Float32,Float64}}

for op in unops_Base_SP
    @eval begin
        @inline Base.$op(x::$vec) = SIMDVec(SP.$op(VBVec(x)))
    end
end
for op in unops_FM_SP
    @eval @inline FM.$op(x::$vec) = SIMDVec(SP.$op(VBVec(x)))
end
for op in unops_FM_SP_slow
    op_fast = Symbol(op, :_fast)
    @eval @inline FM.$op_fast(x::$vec) = SIMDVec(SP.$op(VBVec(x)))
end

# two-argument functions : x^n with n scalar
@eval @inline FM.pow_fast(x::SIMD.Vec{<:Any,F}, n::F) where {F<:Union{Float32,Float64}} = FM.exp_fast(n * FM.log_fast(x))

for op in union(unops_FM_SP, unops_FM_SP_slow), F in (Float32, Float64), N in (4,8,16)
    op_fast = getfield(FM, op)
    precompile(op_fast, (SIMD.Vec{N,F},))
end

end
