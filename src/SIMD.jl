module SIMD

# A note on Val{} vs. Val():
#
# For historic reasoons, SIMD's API accepted compile-time constants as
# Val{N} instead of Val(N). The difference is that Val{N} is a type
# (Type{Val{N}}), whereas Val(N) is a value (of type Val{N}). This is
# against the intent of how Val is designed, and is also much slower
# at run time unless functions are either @inline'd or @generated.
#
# The API has now been cleaned up. To preserve backward compatibility,
# passing Val{N} instead of Val(N) is still supported. It might go
# away at the next major release.



#=

# Various boolean types

# Idea (from <Gaunard-simd.pdf>): Use Mask{N,T} instead of booleans
# with different sizes

abstract Boolean <: Integer

for sz in (8, 16, 32, 64, 128)
    Intsz = Symbol(:Int, sz)
    UIntsz = Symbol(:UInt, sz)
    Boolsz = Symbol(:Bool, sz)
    @eval begin
        immutable $Boolsz <: Boolean
            int::$UIntsz
            $Boolsz(b::Bool) =
                new(ifelse(b, typemax($UIntsz), typemin($UIntsz)))
        end
        booltype(::Val($sz)) = $Boolsz
        inttype(::Val($sz)) = $Intsz
        uinttype(::Val($sz)) = $UIntsz

        Base.convert(::Type{Bool}, b::$Boolsz) = b.int != 0

        Base.:~(b::$Boolsz) = $Boolsz(~b.int)
        Base.:!(b::$Boolsz) = ~b
        Base.:&(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int & b2.int)
        Base.:|(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int | b2.int)
        Base.$(:$)(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int $ b2.int)

        Base.:==(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int == b2.int)
        Base.:!=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int != b2.int)
        Base.:<(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int < b2.int)
        Base.:<=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int <= b2.int)
        Base.:>(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int > b2.int)
        Base.:>=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int >= b2.int)
    end
end
Base.convert(::Type{Bool}, b::Boolean) = error("impossible")
Base.convert{I<:Integer}(::Type{I}, b::Boolean) = I(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, b::Boolean) = B(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, i::Integer) = B(i!=0)

booltype{T}(::Type{T}) = booltype(Val(8*sizeof(T)))
inttype{T}(::Type{T}) = inttype(Val(8*sizeof(T)))
uinttype{T}(::Type{T}) = uinttype(Val(8*sizeof(T)))

=#

# Array types for SIMD

using Base: Slice, ScalarIndex

"""
    ContiguousSubArray{T,N,P,I,L}

Like `Base.FastContiguousSubArray` but without requirement for linear
indexing (i.e., type parameter `L` can be `false`).

# Examples
```
julia> A = view(ones(5, 5), :, [1,3]);

julia> A isa Base.FastContiguousSubArray
false

julia> A isa SIMD.ContiguousSubArray
true
```
"""
ContiguousSubArray{T,N,P,
                   I<:Union{Tuple{Union{Slice, AbstractUnitRange}, Vararg{Any}},
                            Tuple{Vararg{ScalarIndex}}},
                   L} = SubArray{T,N,P,I,L}

"""
    ContiguousArray{T,N}

Array types with contiguous first dimension.
"""
ContiguousArray{T,N} = Union{DenseArray{T,N}, ContiguousSubArray{T,N}}

"""
    FastContiguousArray{T,N}

This is the type of arrays that `pointer(A, i)` works.
"""
FastContiguousArray{T,N} = Union{DenseArray{T,N}, Base.FastContiguousSubArray{T,N}}
# https://github.com/eschnett/SIMD.jl/pull/40#discussion_r254131184
# https://github.com/JuliaArrays/MappedArrays.jl/pull/24#issuecomment-460568978

# The Julia SIMD vector type

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const IndexTypes = Union{IntegerTypes, Ptr}
const FloatingTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{IndexTypes, FloatingTypes}

const VE = Base.VecElement

export Vec
struct Vec{N,T<:ScalarTypes} # <: Number
    elts::NTuple{N,VE{T}}
    @inline Vec{N,T}(elts::NTuple{N, VE{T}}) where {N,T} = new{N,T}(elts)
end

function Base.show(io::IO, v::Vec{N,T}) where {N,T}
    print(io, "<$N x $T>[")
    join(io, [x.value for x in v.elts], ", ")
    print(io, "]")
end

# Type properties
Base.eltype(::Type{Vec{N,T}}) where {N,T} = T
Base.ndims( ::Type{Vec{N,T}}) where {N,T} = 1
Base.length(::Type{Vec{N,T}}) where {N,T} = N
Base.size(  ::Type{Vec{N,T}}) where {N,T} = (N,)
# TODO: This doesn't follow Base, e.g. `size([], 3) == 1`
Base.size(::Type{Vec{N,T}}, n::Integer) where {N,T} = (N,)[n]

Base.eltype(V::Vec) = eltype(typeof(V))
Base.ndims( V::Vec) = ndims(typeof(V))
Base.length(V::Vec) = length(typeof(V))
Base.size(  V::Vec) = size(typeof(V))
Base.size(  V::Vec, n::Integer) = size(typeof(V), n)

# Type conversion

# Create vectors from scalars or tuples
@generated function (::Type{Vec{N,T}})(x::S) where {N,T,S<:ScalarTypes}
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(tuple($([:(VE{T}(T(x))) for i in 1:N]...)))
    end
end
Vec{N,T}(xs::Tuple{}) where {N,T<:ScalarTypes} = error("illegal argument")
@generated function (::Type{Vec{N,T}})(xs::NTuple{N,S}) where {N,T,S<:ScalarTypes}
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(tuple($([:(VE{T}(T(xs[$i]))) for i in 1:N]...)))
    end
end
Vec(xs::NTuple{N,T}) where {N,T<:ScalarTypes} = Vec{N,T}(xs)

# Convert between vectors
@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v

@inline Base.convert(::Type{Vec{N,R}}, v::Vec{N}) where {N,R} =
    Vec{N,R}(NTuple{N, R}(v))

@inline Tuple(v::Vec{N}) where {N} = ntuple(i -> v.elts[i].value, Val(N))
@inline NTuple{N, T}(v::Vec{N}) where{N, T} = ntuple(i -> convert(T, v.elts[i].value), Val(N))

@generated function Base.:%(v::Vec{N,T}, ::Type{Vec{N,R}}) where {N,R,T}
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(tuple($([:(v.elts[$i].value % R) for i in 1:N]...)))
    end
end

# Convert vectors to tuples
@generated function Base.convert(::Type{NTuple{N,R}}, v::Vec{N,T}) where {N,R,T}
    quote
        $(Expr(:meta, :inline))
        tuple($([:(R(v.elts[$i].value)) for i in 1:N]...))
    end
end
@inline Base.convert(::Type{Tuple}, v::Vec{N,T}) where {N,T} =
    Base.convert(NTuple{N,T}, v)

# Promotion rules

# Note: Type promotion only works for subtypes of Number
# Base.promote_rule{N,T<:ScalarTypes}(::Type{Vec{N,T}}, ::Type{T}) = Vec{N,T}

Base.zero(::Type{Vec{N,T}}) where {N,T} = Vec{N,T}(zero(T))
Base.one(::Type{Vec{N,T}}) where {N,T} = Vec{N,T}(one(T))
Base.zero(::Vec{N,T}) where {N,T} = zero(Vec{N,T})
Base.one(::Vec{N,T}) where {N,T} = one(Vec{N,T})

# Floating point formats

int_type(::Type{Float16}) = Int16
int_type(::Type{Float32}) = Int32
int_type(::Type{Float64}) = Int64
# int_type(::Type{Float128}) = Int128
# int_type(::Type{Float256}) = Int256

uint_type(::Type{Float16}) = UInt16
uint_type(::Type{Float32}) = UInt32
uint_type(::Type{Float64}) = UInt64
# uint_type(::Type{Float128}) = UInt128
# uint_type(::Type{Float256}) = UInt256

significand_bits(::Type{Float16}) = 10
significand_bits(::Type{Float32}) = 23
significand_bits(::Type{Float64}) = 52
# significand_bits(::Type{Float128}) = 112
# significand_bits(::Type{Float256}) = 136

exponent_bits(::Type{T}) where {T<:FloatingTypes} =
    8*sizeof(T) - 1 - significand_bits(T)
sign_bits(::Type{T}) where {T<:FloatingTypes} = 1

significand_mask(::Type{T}) where {T<:FloatingTypes} =
    uint_type(T)(uint_type(T)(1) << significand_bits(T) - 1)
exponent_mask(::Type{T}) where {T<:FloatingTypes} =
    uint_type(T)(uint_type(T)(1) << exponent_bits(T) - 1) << significand_bits(T)
sign_mask(::Type{T}) where {T<:FloatingTypes} =
    uint_type(T)(1) << (significand_bits(T) + exponent_bits(T))

for T in (Float16, Float32, Float64)
    @assert sizeof(int_type(T)) == sizeof(T)
    @assert sizeof(uint_type(T)) == sizeof(T)
    @assert significand_bits(T) + exponent_bits(T) + sign_bits(T) == 8*sizeof(T)
    @assert significand_mask(T) | exponent_mask(T) | sign_mask(T) ==
        typemax(uint_type(T))
    @assert significand_mask(T) ⊻ exponent_mask(T) ⊻ sign_mask(T) ==
        typemax(uint_type(T))
end

# Convert Julia types to LLVM types

llvmtype(::Type{Bool}) = "i8"   # Julia represents Tuple{Bool} as [1 x i8]

# llvmtype(::Type{Bool8}) = "i8"
# llvmtype(::Type{Bool16}) = "i16"
# llvmtype(::Type{Bool32}) = "i32"
# llvmtype(::Type{Bool64}) = "i64"
# llvmtype(::Type{Bool128}) = "i128"

llvmtype(::Type{Int8}) = "i8"
llvmtype(::Type{Int16}) = "i16"
llvmtype(::Type{Int32}) = "i32"
llvmtype(::Type{Int64}) = "i64"
llvmtype(::Type{Int128}) = "i128"
llvmtype(::Type{<:Ptr}) = llvmtype(Int)

llvmtype(::Type{UInt8}) = "i8"
llvmtype(::Type{UInt16}) = "i16"
llvmtype(::Type{UInt32}) = "i32"
llvmtype(::Type{UInt64}) = "i64"
llvmtype(::Type{UInt128}) = "i128"

llvmtype(::Type{Float16}) = "half"
llvmtype(::Type{Float32}) = "float"
llvmtype(::Type{Float64}) = "double"

# Type-dependent optimization flags
# fastflags{T<:IntTypes}(::Type{T}) = "nsw"
# fastflags{T<:UIntTypes}(::Type{T}) = "nuw"
# fastflags{T<:FloatingTypes}(::Type{T}) = "fast"

suffix(N::Integer, ::Type{T}) where {T<:IntegerTypes} = "v$(N)i$(8*sizeof(T))"
suffix(N::Integer, ::Type{T}) where {T<:FloatingTypes} = "v$(N)f$(8*sizeof(T))"

# Type-dependent LLVM constants
function llvmconst(::Type{T}, val) where T
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "$typ $val"
end
function llvmconst(::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "$typ $(Int(val))"
end
function llvmconst(N::Integer, ::Type{T}, val) where T
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "<" * join(["$typ $val" for i in 1:N], ", ") * ">"
end
function llvmconst(N::Integer, ::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "<" * join(["$typ $(Int(val))" for i in 1:N], ", ") * ">"
end
function llvmtypedconst(::Type{T}, val) where T
    typ = llvmtype(T)
    T(val) === T(0) && return "$typ zeroinitializer"
    "$typ $val"
end
function llvmtypedconst(::Type{Bool}, val)
    typ = "i1"
    Bool(val) === false && return "$typ zeroinitializer"
    "$typ $(Int(val))"
end

# Type-dependent LLVM intrinsics
llvmins(::Val{:+}, N, ::Type{T}) where {T <: IndexTypes} = "add"
llvmins(::Val{:-}, N, ::Type{T}) where {T <: IndexTypes} = "sub"
llvmins(::Val{:*}, N, ::Type{T}) where {T <: IntegerTypes} = "mul"
llvmins(::Val{:div}, N, ::Type{T}) where {T <: IntTypes} = "sdiv"
llvmins(::Val{:rem}, N, ::Type{T}) where {T <: IntTypes} = "srem"
llvmins(::Val{:div}, N, ::Type{T}) where {T <: UIntTypes} = "udiv"
llvmins(::Val{:rem}, N, ::Type{T}) where {T <: UIntTypes} = "urem"

llvmins(::Val{:~}, N, ::Type{T}) where {T <: IntegerTypes} = "xor"
llvmins(::Val{:&}, N, ::Type{T}) where {T <: IntegerTypes} = "and"
llvmins(::Val{:|}, N, ::Type{T}) where {T <: IntegerTypes} = "or"
llvmins(::Val{:⊻}, N, ::Type{T}) where {T <: IntegerTypes} = "xor"

llvmins(::Val{:<<}, N, ::Type{T}) where {T <: IntegerTypes} = "shl"
llvmins(::Val{:>>>}, N, ::Type{T}) where {T <: IntegerTypes} = "lshr"
llvmins(::Val{:>>}, N, ::Type{T}) where {T <: UIntTypes} = "lshr"
llvmins(::Val{:>>}, N, ::Type{T}) where {T <: IntTypes} = "ashr"

llvmins(::Val{:(==)}, N, ::Type{T}) where {T <: IntegerTypes} = "icmp eq"
llvmins(::Val{:(!=)}, N, ::Type{T}) where {T <: IntegerTypes} = "icmp ne"
llvmins(::Val{:(>)}, N, ::Type{T}) where {T <: IntTypes} = "icmp sgt"
llvmins(::Val{:(>=)}, N, ::Type{T}) where {T <: IntTypes} = "icmp sge"
llvmins(::Val{:(<)}, N, ::Type{T}) where {T <: IntTypes} = "icmp slt"
llvmins(::Val{:(<=)}, N, ::Type{T}) where {T <: IntTypes} = "icmp sle"
llvmins(::Val{:(>)}, N, ::Type{T}) where {T <: UIntTypes} = "icmp ugt"
llvmins(::Val{:(>=)}, N, ::Type{T}) where {T <: UIntTypes} = "icmp uge"
llvmins(::Val{:(<)}, N, ::Type{T}) where {T <: UIntTypes} = "icmp ult"
llvmins(::Val{:(<=)}, N, ::Type{T}) where {T <: UIntTypes} = "icmp ule"

llvmins(::Val{:vifelse}, N, ::Type{T}) where {T} = "select"

llvmins(::Val{:+}, N, ::Type{T}) where {T <: FloatingTypes} = "fadd"
llvmins(::Val{:-}, N, ::Type{T}) where {T <: FloatingTypes} = "fsub"
llvmins(::Val{:*}, N, ::Type{T}) where {T <: FloatingTypes} = "fmul"
llvmins(::Val{:/}, N, ::Type{T}) where {T <: FloatingTypes} = "fdiv"
llvmins(::Val{:inv}, N, ::Type{T}) where {T <: FloatingTypes} = "fdiv"
llvmins(::Val{:rem}, N, ::Type{T}) where {T <: FloatingTypes} = "frem"

llvmins(::Val{:(==)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp oeq"
llvmins(::Val{:(!=)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp une"
llvmins(::Val{:(>)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp ogt"
llvmins(::Val{:(>=)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp oge"
llvmins(::Val{:(<)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp olt"
llvmins(::Val{:(<=)}, N, ::Type{T}) where {T <: FloatingTypes} = "fcmp ole"

llvmins(::Val{:^}, N, ::Type{T}) where {T <: FloatingTypes} =
    "@llvm.pow.$(suffix(N,T))"
llvmins(::Val{:abs}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.fabs.$(suffix(N,T))"
llvmins(::Val{:ceil}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.ceil.$(suffix(N,T))"
llvmins(::Val{:copysign}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.copysign.$(suffix(N,T))"
llvmins(::Val{:cos}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.cos.$(suffix(N,T))"
llvmins(::Val{:exp}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.exp.$(suffix(N,T))"
llvmins(::Val{:exp2}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.exp2.$(suffix(N,T))"
llvmins(::Val{:floor}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.floor.$(suffix(N,T))"
llvmins(::Val{:fma}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.fma.$(suffix(N,T))"
llvmins(::Val{:log}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.log.$(suffix(N,T))"
llvmins(::Val{:log10}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.log10.$(suffix(N,T))"
llvmins(::Val{:log2}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.log2.$(suffix(N,T))"
llvmins(::Val{:max}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.maxnum.$(suffix(N,T))"
llvmins(::Val{:min}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.minnum.$(suffix(N,T))"
# llvmins(::Val{:max}, N, ::Type{T}) where {T<:FloatingTypes} =
#     "@llvm.maximum.$(suffix(N,T))"
# llvmins(::Val{:min}, N, ::Type{T}) where {T<:FloatingTypes} =
#     "@llvm.minimum.$(suffix(N,T))"
llvmins(::Val{:muladd}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.fmuladd.$(suffix(N,T))"
llvmins(::Val{:powi}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.powi.$(suffix(N,T))"
llvmins(::Val{:round}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.rint.$(suffix(N,T))"
llvmins(::Val{:sin}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.sin.$(suffix(N,T))"
llvmins(::Val{:sqrt}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.sqrt.$(suffix(N,T))"
llvmins(::Val{:trunc}, N, ::Type{T}) where {T<:FloatingTypes} =
    "@llvm.trunc.$(suffix(N,T))"

# Convert between LLVM scalars, vectors, and arrays

function scalar2vector(vec, siz, typ, sca)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $sca, i32 $i")
    end
    instrs
end

function array2vector(vec, siz, typ, arr, tmp="$(arr)_av")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs, "$(tmp)_elem$i = extractvalue [$siz x $typ] $arr, $i")
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

function vector2array(arr, siz, typ, vec, tmp="$(vec)_va")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $i")
        push!(instrs,
            "$(accum(arr,i)) = "*
                "insertvalue [$siz x $typ] $(accum(arr,i-1)), " *
                "$typ $(tmp)_elem$i, $i")
    end
    instrs
end

# TODO: change argument order
function subvector(vec, siz, typ, rvec, rsiz, roff, tmp="$(rvec)_sv")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==rsiz-1 ? nam : "$(nam)_iter$i"
    @assert 0 <= roff
    @assert roff + rsiz <= siz
    for i in 0:rsiz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $(roff+i)")
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

function extendvector(vec, siz, typ, voff, vsiz, val, rvec, tmp="$(rvec)_ev")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz+vsiz-1 ? nam : "$(nam)_iter$i"
    rsiz = siz + vsiz
    for i in 0:siz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz x $typ> $vec, i32 $i")
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    for i in siz:siz+vsiz-1
        push!(instrs,
            "$(accum(rvec,i)) = " *
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), $val, i32 $i")
    end
    instrs
end

# Element-wise access

export setindex
@generated function setindex(v::Vec{N,T}, x::Number, ::Val{I}) where {N,T,I}
    @assert isa(I, Integer)
    1 <= I <= N || throw(BoundsError())
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    push!(instrs, "%res = insertelement $vtyp %0, $typ %1, $ityp $(I-1)")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, T}, v.elts, T(x)))
    end
end
@inline function setindex(v::Vec{N,T}, x::Number, ::Type{Val{I}}) where {N,T,I}
    setindex(v, x, Val(I))
end

@generated function setindex(v::Vec{N,T}, x::Number, i::Int) where {N,T}
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    push!(instrs, "%res = insertelement $vtyp %0, $typ %2, $ityp %1")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        @boundscheck 1 <= i <= N || throw(BoundsError())
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, Int, T},
            v.elts, i-1, T(x)))
    end
end
setindex(v::Vec{N,T}, x::Number, i) where {N,T} = setindex(v, Int(i), x)

Base.@propagate_inbounds Base.getindex(v::Vec{N,T}, ::Val{I}) where {N,T,I} = v.elts[I].value
Base.@propagate_inbounds Base.getindex(v::Vec{N,T}, ::Type{Val{I}}) where {N,T,I} = Base.getindex(v, Val(I))
Base.@propagate_inbounds Base.getindex(v::Vec{N,T}, i) where {N,T} = v.elts[i].value

# Type conversion

@generated function Base.reinterpret(::Type{Vec{N,R}},
        v1::Vec{N1,T1}) where {N,R,N1,T1}
    @assert N*sizeof(R) == N1*sizeof(T1)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N1 x $typ1>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    decls = []
    instrs = []
    push!(instrs, "%res = bitcast $vtyp1 %0 to $vtypr")
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N1,VE{T1}}}, v1.elts))
    end
end

# Generic function wrappers

# Functions taking one argument
@generated function llvmwrap(::Val{Op}, v1::Vec{N,T1},
                             ::Type{R} = T1) where {Op,N,T1,R}
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val(Op), N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %0)")
    else
        if Op === :~
            @assert T1 <: IntegerTypes
            otherval = -1
        elseif Op === :inv
            @assert T1 <: FloatingTypes
            otherval = 1.0
        else
            otherval = 0
        end
        otherarg = llvmconst(N, T1, otherval)
        push!(instrs, "%res = $ins $vtyp1 $otherarg, %0")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N,VE{T1}}}, v1.elts))
    end
end

# Functions taking one Bool argument
@generated function llvmwrap(::Val{Op}, v1::Vec{N,Bool},
                             ::Type{Bool} = Bool) where {Op,N}
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    ins = llvmins(Val(Op), N, Bool)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = trunc $vbtyp %0 to <$N x i1>")
    otherarg = llvmconst(N, Bool, true)
    push!(instrs, "%res = $ins <$N x i1> $otherarg, %arg1")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    push!(instrs, "ret $vbtyp %resb")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{Bool}}}, v1.elts))
    end
end

# Functions taking two arguments
@generated function llvmwrap(::Val{Op}, v1::Vec{N,T1}, v2::Vec{N,T2},
        ::Type{R} = T1) where {Op,N,T1,T2,R}
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val(Op), N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %0, $vtyp2 %1)")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}},
            v1.elts, v2.elts))
    end
end

# Functions taking two arguments, second argument is a scalar
@generated function llvmwrap(::Val{Op}, v1::Vec{N,T1}, s2::ScalarTypes,
        ::Type{R} = T1) where {Op,N,T1,R}
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(s2)
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val(Op), N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $typ2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %0, $typ2 %1)")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}}, Tuple{NTuple{N,VE{T1}}, $s2},
            v1.elts, s2))
    end
end

# Functions taking two arguments, returning Bool
@generated function llvmwrap(::Val{Op}, v1::Vec{N,T1}, v2::Vec{N,T2},
        ::Type{Bool}) where {Op,N,T1,T2}
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    atyp1 = "[$N x $typ1]"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    atyp2 = "[$N x $typ2]"
    ins = llvmins(Val(Op), N, T1)
    decls = []
    instrs = []
    if false && N == 1
        append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
        append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
        push!(instrs, "%cond = $ins $vtyp1 %arg1, %arg2")
        push!(instrs, "%res = zext <$N x i1> %cond to $vbtyp")
        append!(instrs, vector2array("%resarr", N, btyp, "%res"))
        push!(instrs, "ret $abtyp %resarr")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1")
        push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
        push!(instrs, "ret $vbtyp %resb")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}},
            v1.elts, v2.elts))
    end
end

# Functions taking a vector and a scalar argument
# @generated function llvmwrap{Op,N,T1,T2,R}(::Val{Op}, v1::Vec{N,T1},
#         x2::T2, ::Type{R} = T1)
#     @assert isa(Op, Symbol)
#     typ1 = llvmtype(T1)
#     atyp1 = "[$N x $typ1]"
#     vtyp1 = "<$N x $typ1>"
#     typ2 = llvmtype(T2)
#     typr = llvmtype(R)
#     atypr = "[$N x $typr]"
#     vtypr = "<$N x $typr>"
#     ins = llvmins(Val(Op), N, T1)
#     decls = []
#     instrs = []
#     append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
#     if ins[1] == '@'
#         push!(decls, "declare $vtypr $ins($vtyp1, $typ2)")
#         push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1, $typ2 %1)")
#     else
#         push!(instrs, "%res = $ins $vtyp1 %arg1, %1")
#     end
#     append!(instrs, vector2array("%resarr", N, typr, "%res"))
#     push!(instrs, "ret $atypr %resarr")
#     quote
#         $(Expr(:meta, :inline))
#         Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
#             NTuple{N,R}, Tuple{NTuple{N,T1}, T2}, v1.elts, x2))
#     end
# end

# Functions taking two Bool arguments, returning Bool
@generated function llvmwrap(::Val{Op}, v1::Vec{N,Bool}, v2::Vec{N,Bool},
        ::Type{Bool} = Bool) where {Op,N}
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    ins = llvmins(Val(Op), N, Bool)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = trunc $vbtyp %0 to <$N x i1>")
    push!(instrs, "%arg2 = trunc $vbtyp %1 to <$N x i1>")
    push!(instrs, "%res = $ins <$N x i1> %arg1, %arg2")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    push!(instrs, "ret $vbtyp %resb")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{Bool}}, Tuple{NTuple{N,VE{Bool}}, NTuple{N,VE{Bool}}},
            v1.elts, v2.elts))
    end
end

# Functions taking three arguments
@generated function llvmwrap(::Val{Op}, v1::Vec{N,T1}, v2::Vec{N,T2},
        v3::Vec{N,T3}, ::Type{R} = T1) where {Op,N,T1,T2,T3,R}
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    vtyp2 = "<$N x $typ2>"
    typ3 = llvmtype(T3)
    vtyp3 = "<$N x $typ3>"
    typr = llvmtype(R)
    vtypr = "<$N x $typr>"
    ins = llvmins(Val(Op), N, T1)
    decls = []
    instrs = []
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2, $vtyp3)")
        push!(instrs,
            "%res = call $vtypr $ins($vtyp1 %0, $vtyp2 %1, $vtyp3 %2)")
    else
        push!(instrs, "%res = $ins $vtyp1 %0, %1, %2")
    end
    push!(instrs, "ret $vtypr %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{R}},
            Tuple{NTuple{N,VE{T1}}, NTuple{N,VE{T2}}, NTuple{N,VE{T3}}},
            v1.elts, v2.elts, v3.elts))
    end
end

@generated function llvmwrapshift(::Val{Op}, v1::Vec{N,T},
        ::Val{I}) where {Op,N,T,I}
    @assert isa(Op, Symbol)
    if I >= 0
        op = Op
        i = I
    else
        if Op === :>> || Op === :>>>
            op = :<<
        else
            @assert Op === :<<
            if T <: Unsigned
                op = :>>>
            else
                op = :>>
            end
        end
        i = -I
    end
    @assert op in (:<<, :>>, :>>>)
    @assert i >= 0
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val(op), N, T)
    decls = []
    instrs = []
    nbits = 8*sizeof(T)
    if (op === :>> && T <: IntTypes) || i < nbits
        count = llvmconst(N, T, min(nbits-1, i))
        push!(instrs, "%res = $ins $vtyp %0, $count")
        push!(instrs, "ret $vtyp %res")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "return $vtyp $zero")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}}, v1.elts))
    end
end

@generated function llvmwrapshift(::Val{Op}, v1::Vec{N,T},
        x2::Unsigned) where {Op,N,T}
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val(Op), N, T)
    decls = []
    instrs = []
    append!(instrs, scalar2vector("%count", N, typ, "%1"))
    nbits = 8*sizeof(T)
    push!(instrs, "%tmp = $ins $vtyp %0, %count")
    push!(instrs, "%inbounds = icmp ult $typ %1, $nbits")
    if Op === :>> && T <: IntTypes
        nbits1 = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %0, $nbits1")
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        # Note that this function might be called with out-of-bounds
        # values for x2, assuming that the results are then ignored
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, T}, v1.elts, x2 % T))
    end
end

@generated function llvmwrapshift(::Val{Op}, v1::Vec{N,T},
        x2::Integer) where {Op,N,T}
    if Op === :>> || Op === :>>>
        NegOp = :<<
    else
        @assert Op === :<<
        if T <: Unsigned
            NegOp = :>>>
        else
            NegOp = :>>
        end
    end
    ValOp = Val(Op)
    ValNegOp = Val(NegOp)
    quote
        $(Expr(:meta, :inline))
        ifelse(x2 >= 0,
               llvmwrapshift($ValOp, v1, unsigned(x2)),
               llvmwrapshift($ValNegOp, v1, unsigned(-x2)))
    end
end

@generated function llvmwrapshift(::Val{Op}, v1::Vec{N,T},
        v2::Vec{N,U}) where {Op,N,T,U<:UIntTypes}
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    ins = llvmins(Val(Op), N, T)
    decls = []
    instrs = []
    push!(instrs, "%tmp = $ins $vtyp %0, %1")
    nbits = llvmconst(N, T, 8*sizeof(T))
    push!(instrs, "%inbounds = icmp ult $vtyp %1, $nbits")
    if Op === :>> && T <: IntTypes
        nbits1 = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %0, $nbits1")
        push!(instrs,
            "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs,
            "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, (v2 % Vec{N,T}).elts))
    end
end

@generated function llvmwrapshift(::Val{Op}, v1::Vec{N,T},
        v2::Vec{N,U}) where {Op,N,T,U<:IntegerTypes}
    if Op === :>> || Op === :>>>
        NegOp = :<<
    else
        @assert Op === :<<
        if T <: Unsigned
            NegOp = :>>>
        else
            NegOp = :>>
        end
    end
    ValOp = Val(Op)
    ValNegOp = Val(NegOp)
    quote
        $(Expr(:meta, :inline))
        vifelse(v2 >= 0,
                llvmwrapshift($ValOp, v1, v2 % Vec{N,unsigned(U)}),
                llvmwrapshift($ValNegOp, v1, -v2 % Vec{N,unsigned(U)}))
    end
end

# Conditionals

for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
    @eval begin
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T} =
            llvmwrap(Val($(QuoteNode(op))), v1, v2, Bool)
    end
end
@inline function Base.cmp(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T}
    I = int_type(T)
    vifelse(isequal(v1, v2), Vec{N,I}(0),
            vifelse(isless(v1, v2), Vec{N,I}(-1), Vec{N,I}(1)))
end
@inline function Base.isfinite(v1::Vec{N,T}) where {N,T<:FloatingTypes}
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & em != em
end
@inline Base.isinf(v1::Vec{N,T}) where {N,T<:FloatingTypes} = abs(v1) == Vec{N,T}(Inf)
@inline Base.isnan(v1::Vec{N,T}) where {N,T<:FloatingTypes} = v1 != v1
@inline function Base.issubnormal(v1::Vec{N,T}) where {N,T<:FloatingTypes}
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    sm = Vec{N,U}(significand_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    (iv & em == Vec{N,U}(0)) & (iv & sm != Vec{N,U}(0))
end
@inline function Base.signbit(v1::Vec{N,T}) where {N,T<:FloatingTypes}
    U = uint_type(T)
    sm = Vec{N,U}(sign_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & sm != Vec{N,U}(0)
end

export vifelse
vifelse(c::Bool, x, y) = ifelse(c, x, y)
@generated function vifelse(v1::Vec{N,Bool}, v2::Vec{N,T},
        v3::Vec{N,T}) where {N,T}
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    atyp = "[$N x $typ]"
    decls = []
    instrs = []
    if false && N == 1
        append!(instrs, array2vector("%arg1", N, btyp, "%0", "%arg1arr"))
        append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
        append!(instrs, array2vector("%arg3", N, typ, "%2", "%arg3arr"))
        push!(instrs, "%cond = trunc $vbtyp %arg1 to <$N x i1>")
        push!(instrs, "%res = select <$N x i1> %cond, $vtyp %arg2, $vtyp %arg3")
        append!(instrs, vector2array("%resarr", N, typ, "%res"))
        push!(instrs, "ret $atyp %resarr")
    else
        push!(instrs, "%cond = trunc $vbtyp %0 to <$N x i1>")
        push!(instrs, "%res = select <$N x i1> %cond, $vtyp %1, $vtyp %2")
        push!(instrs, "ret $vtyp %res")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}},
            Tuple{NTuple{N,VE{Bool}}, NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, v2.elts, v3.elts))
    end
end

# Integer arithmetic functions

for op in (:~, :+, :-)
    @eval begin
        @inline Base.$op(v1::Vec{N,T}) where {N,T<:IntegerTypes} =
            llvmwrap(Val($(QuoteNode(op))), v1)
    end
end
@inline Base.:!(v1::Vec{N,Bool}) where {N} = ~v1
@inline function Base.abs(v1::Vec{N,T}) where {N,T<:IntTypes}
    # s = -Vec{N,T}(signbit(v1))
    s = v1 >> Val(8*sizeof(T))
    # Note: -v1 == ~v1 + 1
    (s ⊻ v1) - s
end
@inline Base.abs(v1::Vec{N,T}) where {N,T<:UIntTypes} = v1
# TODO: Try T(v1>0) - T(v1<0)
#       use a shift for v1<0
#       evaluate v1>0 as -v1<0 ?
@inline Base.sign(v1::Vec{N,T}) where {N,T<:IntTypes} =
    vifelse(v1 == Vec{N,T}(0), Vec{N,T}(0),
        vifelse(v1 < Vec{N,T}(0), Vec{N,T}(-1), Vec{N,T}(1)))
@inline Base.sign(v1::Vec{N,T}) where {N,T<:UIntTypes} =
    vifelse(v1 == Vec{N,T}(0), Vec{N,T}(0), Vec{N,T}(1))
@inline Base.signbit(v1::Vec{N,T}) where {N,T<:IntTypes} = v1 < Vec{N,T}(0)
@inline Base.signbit(v1::Vec{N,T}) where {N,T<:UIntTypes} = Vec{N,Bool}(false)

for op in (:&, :|, :⊻, :+, :-, :*, :div, :rem)
    @eval begin
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
            llvmwrap(Val($(QuoteNode(op))), v1, v2)
    end
end
@inline Base.copysign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntTypes} =
    vifelse(signbit(v2), -abs(v1), abs(v1))
@inline Base.copysign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:UIntTypes} = v1
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntTypes} =
    vifelse(signbit(v2), -v1, v1)
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:UIntTypes} = v1
@inline Base.max(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    vifelse(v1>=v2, v1, v2)
@inline Base.min(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    vifelse(v1>=v2, v2, v1)

@inline function Base.muladd(v1::Vec{N,T}, v2::Vec{N,T},
        v3::Vec{N,T}) where {N,T<:IntegerTypes}
    v1*v2+v3
end

# TODO: Handle negative shift counts
#       use vifelse
#       ensure vifelse is efficient
for op in (:<<, :>>, :>>>)
    @eval begin
        @inline Base.$op(v1::Vec{N,T}, ::Val{I}) where {N,T<:IntegerTypes,I} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, Val(I))
        @inline Base.$op(v1::Vec{N,T}, ::Type{Val{I}}) where {N,T<:IntegerTypes,I} =
            Base.$op(v1, Val(I))
        @inline Base.$op(v1::Vec{N,T}, x2::Unsigned) where {N,T<:IntegerTypes} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, x2)
        @inline Base.$op(v1::Vec{N,T}, x2::Int) where {N,T<:IntegerTypes} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, x2)
        @inline Base.$op(v1::Vec{N,T}, x2::Integer) where {N,T<:IntegerTypes} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, x2)
        @inline Base.$op(v1::Vec{N,T},
                         v2::Vec{N,U}) where {N,T<:IntegerTypes,U<:UIntTypes} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, v2)
        @inline Base.$op(v1::Vec{N,T},
                         v2::Vec{N,U}) where {N,T<:IntegerTypes,U<:IntegerTypes} =
            llvmwrapshift(Val($(QuoteNode(op))), v1, v2)
        @inline Base.$op(x1::T, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
            $op(Vec{N,T}(x1), v2)
    end
end

# Floating point arithmetic functions

for op in (
        :+, :-,
        :abs, :ceil, :cos, :exp, :exp2, :floor, :inv, :log, :log10, :log2,
        :round, :sin, :sqrt, :trunc)
    @eval begin
        @inline Base.$op(v1::Vec{N,T}) where {N,T<:FloatingTypes} =
            llvmwrap(Val($(QuoteNode(op))), v1)
    end
end
@inline Base.exp10(v1::Vec{N,T}) where {N,T<:FloatingTypes} = Vec{N,T}(10)^v1
@inline Base.sign(v1::Vec{N,T}) where {N,T<:FloatingTypes} =
    vifelse(v1 == Vec{N,T}(0.0), Vec{N,T}(0.0), copysign(Vec{N,T}(1.0), v1))

for op in (:+, :-, :*, :/, :^, :copysign, :max, :min, :rem)
    @eval begin
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:FloatingTypes} =
            llvmwrap(Val($(QuoteNode(op))), v1, v2)
    end
end
# Using `IntegerTypes` here so that this definition "wins" against
# `^(::ScalarTypes, v2::Vec)`.
@inline Base.:^(v1::Vec{N,T}, x2::IntegerTypes) where {N,T<:FloatingTypes} =
    llvmwrap(Val(:powi), v1, Int(x2))
@inline Base.:^(v1::Vec{N,T}, x2::Integer) where {N,T<:FloatingTypes} =
    llvmwrap(Val(:powi), v1, Int(x2))
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:FloatingTypes} =
    vifelse(signbit(v2), -v1, v1)

# Do what Base does for HWNumber:
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{2}) = x*x
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{3}) = x*x*x

for op in (:fma, :muladd)
    @eval begin
        @inline function Base.$op(v1::Vec{N,T},
                v2::Vec{N,T}, v3::Vec{N,T}) where {N,T<:FloatingTypes}
            llvmwrap(Val($(QuoteNode(op))), v1, v2, v3)
        end
    end
end

# Type promotion

# Promote scalars of all IntegerTypes to vectors of IntegerTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :&, :|, :⊻, :+, :-, :*, :copysign, :div, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op(s1::Bool, v2::Vec{N,Bool}) where {N} =
            $op(Vec{N,Bool}(s1), v2)
        @inline Base.$op(s1::IntegerTypes, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op(v1::Vec{N,T}, s2::IntegerTypes) where {N,T<:IntegerTypes} =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline vifelse(c::Vec{N,Bool}, s1::IntegerTypes,
        v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    vifelse(c, Vec{N,T}(s1), v2)
@inline vifelse(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::IntegerTypes) where {N,T<:IntegerTypes} =
    vifelse(c, v1, Vec{N,T}(s2))

for op in (:muladd,)
    @eval begin
        @inline Base.$op(s1::IntegerTypes, v2::Vec{N,T},
                v3::Vec{N,T}) where {N,T<:IntegerTypes} =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op(v1::Vec{N,T}, s2::IntegerTypes,
                v3::Vec{N,T}) where {N,T<:IntegerTypes} =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op(s1::IntegerTypes, s2::IntegerTypes,
                v3::Vec{N,T}) where {N,T<:IntegerTypes} =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T},
                s3::IntegerTypes) where {N,T<:IntegerTypes} =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op(s1::IntegerTypes, v2::Vec{N,T},
                s3::IntegerTypes) where {N,T<:IntegerTypes} =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op(v1::Vec{N,T}, s2::IntegerTypes,
                s3::IntegerTypes) where {N,T<:IntegerTypes} =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Promote scalars of all ScalarTypes to vectors of FloatingTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :+, :-, :*, :/, :^, :copysign, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline vifelse(c::Vec{N,Bool}, s1::ScalarTypes,
        v2::Vec{N,T}) where {N,T<:FloatingTypes} =
    vifelse(c, Vec{N,T}(s1), v2)
@inline vifelse(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::ScalarTypes) where {N,T<:FloatingTypes} =
    vifelse(c, v1, Vec{N,T}(s2))

for op in (:fma, :muladd)
    @eval begin
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T},
                v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes,
                v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op(s1::ScalarTypes, s2::ScalarTypes,
                v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T},
                s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T},
                s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes,
                s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Poitner arithmetics between Ptr, IntegerTypes, and vectors of them.

for op in (:+, :-)
    @eval begin
        @inline Base.$op(v1::Vec{N,<:Ptr}, v2::Vec{N,<:IntegerTypes}) where {N} =
            llvmwrap(Val($(QuoteNode(op))), v1, v2)
        @inline Base.$op(v1::Vec{N,<:IntegerTypes}, v2::Vec{N,<:Ptr}) where {N} =
            llvmwrap(Val($(QuoteNode(op))), v1, v2)
        @inline Base.$op(s1::P, v2::Vec{N,<:IntegerTypes}) where {N,P<:Ptr} =
            $op(Vec{N,P}(s1), v2)
        @inline Base.$op(v1::Vec{N,<:IntegerTypes}, s2::P) where {N,P<:Ptr} =
            $op(v1, Vec{N,P}(s2))
    end
end


# Reduction operations

# TODO: map, mapreduce

function getneutral(op::Symbol, ::Type{T}) where T
    zs = Dict{Symbol,T}()
    if T <: IntegerTypes
        zs[:&] = ~T(0)
        zs[:|] = T(0)
    end
    zs[:max] = typemin(T)
    zs[:min] = typemax(T)
    zs[:+] = T(0)
    zs[:*] = T(1)
    zs[op]
end

if VERSION >= v"0.7.0-beta2.195"
    nextpow2(n) = nextpow(2, n)
end

# We cannot pass in the neutral element via Val{}; if we try, Julia refuses to
# inline this function, which is then disastrous for performance
@generated function llvmwrapreduce(::Val{Op}, v::Vec{N,T}) where {Op,N,T}
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    typ = llvmtype(T)
    decls = []
    instrs = []
    n = N
    nam = "%0"
    nold,n = n,nextpow2(n)
    if n > nold
        namold,nam = nam,"%vec_$n"
        append!(instrs,
            extendvector(namold, nold, typ, n, n-nold,
                llvmtypedconst(T,z), nam))
    end
    while n > 1
        nold,n = n, div(n, 2)
        namold,nam = nam,"%vec_$n"
        vtyp = "<$n x $typ>"
        ins = llvmins(Val(Op), n, T)
        append!(instrs, subvector(namold, nold, typ, "$(nam)_1", n, 0))
        append!(instrs, subvector(namold, nold, typ, "$(nam)_2", n, n))
        if ins[1] == '@'
            push!(decls, "declare $vtyp $ins($vtyp, $vtyp)")
            push!(instrs,
                "$nam = call $vtyp $ins($vtyp $(nam)_1, $vtyp $(nam)_2)")
        else
            push!(instrs, "$nam = $ins $vtyp $(nam)_1, $(nam)_2")
        end
    end
    push!(instrs, "%res = extractelement <$n x $typ> $nam, i32 0")
    push!(instrs, "ret $typ %res")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            T, Tuple{NTuple{N,VE{T}}}, v.elts)
    end
end

@inline Base.all(v::Vec{N,T}) where {N,T<:IntegerTypes} = llvmwrapreduce(Val(:&), v)
@inline Base.any(v::Vec{N,T}) where {N,T<:IntegerTypes} = llvmwrapreduce(Val(:|), v)
@inline Base.maximum(v::Vec{N,T}) where {N,T<:FloatingTypes} =
    llvmwrapreduce(Val(:max), v)
@inline Base.minimum(v::Vec{N,T}) where {N,T<:FloatingTypes} =
    llvmwrapreduce(Val(:min), v)
@inline Base.prod(v::Vec{N,T}) where {N,T} = llvmwrapreduce(Val(:*), v)
@inline Base.sum(v::Vec{N,T}) where {N,T} = llvmwrapreduce(Val(:+), v)

@generated function Base.reduce(::Val{Op}, v::Vec{N,T}) where {Op,N,T}
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    stmts = []
    n = N
    push!(stmts, :($(Symbol(:v,n)) = v))
    nold,n = n,nextpow2(n)
    if n > nold
        push!(stmts,
            :($(Symbol(:v,n)) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in 1:nold]...,
                [z for i in nold+1:n]...)))))
    end
    while n > 1
        nold,n = n, div(n, 2)
        push!(stmts,
            :($(Symbol(:v,n,"lo")) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in 1:n]...,)))))
        push!(stmts,
            :($(Symbol(:v,n,"hi")) = Vec{$n,T}($(Expr(:tuple,
                [:($(Symbol(:v,nold)).elts[$i]) for i in n+1:nold]...)))))
        push!(stmts,
            :($(Symbol(:v,n)) =
                $Op($(Symbol(:v,n,"lo")), $(Symbol(:v,n,"hi")))))
    end
    push!(stmts, :(v1[1]))
    Expr(:block, Expr(:meta, :inline), stmts...)
end
@inline function Base.reduce(::Type{Val{Op}}, v::Vec{N,T}) where {Op,N,T}
    Base.reduce(Val(Op), v)
end

@inline Base.maximum(v::Vec{N,T}) where {N,T<:IntegerTypes} = reduce(Val(:max), v)
@inline Base.minimum(v::Vec{N,T}) where {N,T<:IntegerTypes} = reduce(Val(:min), v)

# Load and store functions

export valloc
function valloc(::Type{T}, N::Int, sz::Int) where T
    @assert N > 0
    @assert sz >= 0
    # We use padding to align the address of the first element, and
    # also to ensure that we can access past the last element up to
    # the next full vector width
    padding = N-1 + mod(-sz, N)
    mem = Vector{T}(undef, sz + padding)
    addr = Int(pointer(mem))
    off = mod(-addr, N * sizeof(T))
    @assert mod(off, sizeof(T)) == 0
    off = fld(off, sizeof(T))
    @assert 0 <= off <= padding
    res = view(mem, off+1 : off+sz)
    addr2 = Int(pointer(res))
    @assert mod(addr2, N * sizeof(T)) == 0
    res
end
function valloc(f, ::Type{T}, N::Int, sz::Int) where T
    mem = valloc(T, N, sz)
    @inbounds for i in 1:sz
        mem[i] = f(i)
    end
    mem
end

export vload, vloada, vloadnt
@generated function vload(::Type{Vec{N,T}}, ptr::Ptr{T},
                          ::Val{Aligned} = Val(false),
                          ::Val{Nontemporal} = Val(false)) where {N,T,Aligned,Nontemporal}
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    if Nontemporal
        push!(flags, "!nontemporal !{i32 1}")
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %0 to $vtyp*")
    end
    push!(instrs, "%res = load $vtyp, $vtyp* %ptr" * join(flags, ", "))
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{Ptr{T}}, ptr))
    end
end
@inline function vload(::Type{Vec{N,T}}, ptr::Ptr{T},
                       ::Type{Val{Aligned}},
                       ::Type{Val{Nontemporal}} = Val{false}) where {N,T,Aligned,Nontemporal}
    vload(Vec{N, T}, ptr, Val(Aligned), Val(Nontemporal))
end

@inline vloada(::Type{Vec{N,T}}, ptr::Ptr{T}) where {N,T} =
    vload(Vec{N,T}, ptr, Val(true))

@inline vloadnt(::Type{Vec{N,T}}, ptr::Ptr{T}) where {N,T} =
    vload(Vec{N,T}, ptr, Val(true), Val(true))

@inline function vload(::Type{Vec{N,T}},
                       arr::FastContiguousArray{T,1},
                       i::Integer,
                       ::Val{Aligned} = Val(false),
                       ::Val{Nontemporal} = Val(false)) where {N,T,Aligned,Nontemporal}
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), Val(Aligned), Val(Nontemporal))
end
@inline function vload(::Type{Vec{N,T}},
                       arr::FastContiguousArray{T,1},
                       i::Integer,
                       ::Type{Val{Aligned}},k = Val{false},
                       ::Type{Val{Nontemporal}} = Val{false}) where {N,T,Aligned,Nontemporal}
    vload(Vec{N,T}, arr, i, Val(Aligned), Val(Nontemporal))
end
@inline function vloada(::Type{Vec{N,T}},
                        arr::FastContiguousArray{T,1},
                        i::Integer) where {N,T}
    vload(Vec{N,T}, arr, i, Val(true))
end
@inline function vloadnt(::Type{Vec{N,T}},
                        arr::Union{Array{T,1},SubArray{T,1}},
                        i::Integer) where {N,T}
    vload(Vec{N,T}, arr, i, Val(true), Val(true))
end

@inline vload(::Type{Vec{N,T}}, ptr::Ptr{T}, mask::Nothing,
              ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vload(Vec{N,T}, ptr, Val(Aligned))
@inline vload(::Type{Vec{N,T}}, ptr::Ptr{T}, mask::Nothing,
              ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vload(Vec{N,T}, ptr, make, Val(Aligned))

@generated function vload(::Type{Vec{N,T}}, ptr::Ptr{T},
                          mask::Vec{N,Bool},
                          ::Val{Aligned} = Val(false)) where {N,T,Aligned}
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end

    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %0 to $vtyp*")
    end
    push!(instrs, "%mask = trunc $vbtyp %1 to <$N x i1>")
    push!(decls,
        "declare $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp*, i32, " *
            "<$N x i1>, $vtyp)")
    push!(instrs,
        "%res = call $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp* %ptr, " *
            "i32 $align, <$N x i1> %mask, $vtyp $(llvmconst(N, T, 0)))")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{Ptr{T}, NTuple{N,VE{Bool}}}, ptr, mask.elts))
    end
end
@inline function vload(::Type{Vec{N,T}}, ptr::Ptr{T},
                       mask::Vec{N,Bool},
                       ::Type{Val{Aligned}}) where {N,T,Aligned}
    vload(Vec{N,T}, ptr, mask, Val(Aligned))
end

@inline vloada(::Type{Vec{N,T}}, ptr::Ptr{T},
               mask::Union{Vec{N,Bool}, Nothing}) where {N,T} =
    vload(Vec{N,T}, ptr, mask, Val(true))

@inline function vload(::Type{Vec{N,T}},
                       arr::FastContiguousArray{T,1},
                       i::Integer, mask::Union{Vec{N,Bool}, Nothing},
                       ::Val{Aligned} = Val(false)) where {N,T,Aligned}
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), mask, Val(Aligned))
end
@inline function vload(::Type{Vec{N,T}},
                       arr::FastContiguousArray{T,1},
                       i::Integer, mask::Union{Vec{N,Bool}, Nothing},
                       ::Type{Val{Aligned}}) where {N,T,Aligned}
    vload(Vec{N,T}, arr, i, mask, Val(Aligned))
end
@inline function vloada(::Type{Vec{N,T}},
                        arr::FastContiguousArray{T,1}, i::Integer,
                        mask::Union{Vec{N,Bool}, Nothing}) where {N,T}
    vload(Vec{N,T}, arr, i, mask, Val(true))
end

export vstore, vstorea, vstorent
@generated function vstore(v::Vec{N,T}, ptr::Ptr{T},
                           ::Val{Aligned} = Val(false),
                           ::Val{Nontemporal} = Val(false)) where {N,T,Aligned,Nontemporal}
    @assert isa(Aligned, Bool)
    @assert isa(Nontemporal, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    if Nontemporal
        push!(flags, "!nontemporal !{i32 1}")
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %1 to $vtyp*")
    end
    push!(instrs, "store $vtyp %0, $vtyp* %ptr" * join(flags, ", "))
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
                      Cvoid, Tuple{NTuple{N,VE{T}}, Ptr{T}}, v.elts, ptr)
    end
end
@inline function vstore(v::Vec{N,T}, ptr::Ptr{T},
                        ::Type{Val{Aligned}},
                        ::Type{Val{Nontemporal}} = Val{false}) where {N,T,Aligned,Nontemporal}
    vstore(v, ptr, Val(Aligned), Val(Nontemporal))
end

@inline vstorea(v::Vec{N,T}, ptr::Ptr{T}) where {N,T} = vstore(v, ptr, Val{true})

@inline vstorent(v::Vec{N,T}, ptr::Ptr{T}) where {N,T} = vstore(v, ptr, Val{true}, Val{true})

@inline function vstore(v::Vec{N,T},
                        arr::FastContiguousArray{T,1},
                        i::Integer,
                        ::Val{Aligned} = Val(false),
                        ::Val{Nontemporal} = Val(false)) where {N,T,Aligned,Nontemporal}
    @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), Val{Aligned}, Val{Nontemporal})
end
@inline function vstore(v::Vec{N,T},
                        arr::FastContiguousArray{T,1},
                        i::Integer,
                        ::Type{Val{Aligned}},
                        ::Type{Val{Nontemporal}} = Val{false}) where {N,T,Aligned,Nontemporal}
    vstore(v, arr, i, Val(Aligned), Val(Nontemporal))
end
@inline function vstorea(v::Vec{N,T}, arr::FastContiguousArray{T,1},
                         i::Integer) where {N,T}
    vstore(v, arr, i, Val{true})
end
@inline function vstorent(v::Vec{N,T}, arr::FastContiguousArray{T,1},
                         i::Integer) where {N,T}
    vstore(v, arr, i, Val{true}, Val{true})
end

@inline vstore(v::Vec{N,T}, ptr::Ptr{T}, mask::Nothing,
               ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vstore(v, ptr, Val{Aligned})
@inline vstore(v::Vec{N,T}, ptr::Ptr{T}, mask::Nothing,
               ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vstore(v, ptr, mask, Val(Aligned))

@generated function vstore(v::Vec{N,T}, ptr::Ptr{T},
                           mask::Vec{N,Bool},
                           ::Val{Aligned} = Val(false)) where {N,T,Aligned}
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %1 to $vtyp*")
    end
    push!(instrs, "%mask = trunc $vbtyp %2 to <$N x i1>")
    push!(decls,
        "declare void @llvm.masked.store.$(suffix(N,T))($vtyp, $vtyp*, i32, " *
            "<$N x i1>)")
    push!(instrs,
        "call void @llvm.masked.store.$(suffix(N,T))($vtyp %0, $vtyp* %ptr, " *
            "i32 $align, <$N x i1> %mask)")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid, Tuple{NTuple{N,VE{T}}, Ptr{T}, NTuple{N,VE{Bool}}},
            v.elts, ptr, mask.elts)
    end
end
@inline function vstore(v::Vec{N,T}, ptr::Ptr{T},
                        mask::Vec{N,Bool},
                        ::Type{Val{Aligned}}) where {N,T,Aligned}
    vstore(v, ptr, mask, Val(Aligned))
end

@inline vstorea(v::Vec{N,T}, ptr::Ptr{T},
                mask::Union{Vec{N,Bool}, Nothing}) where {N,T} =
    vstore(v, ptr, mask, Val{true})

@inline function vstore(v::Vec{N,T},
                        arr::FastContiguousArray{T,1},
                        i::Integer,
                        mask::Union{Vec{N,Bool}, Nothing},
                        ::Val{Aligned} = Val(false),
                        ::Val{Nontemporal} = Val(false)) where {N,T,Aligned,Nontemporal}
    #TODO @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), mask, Val{Aligned}, Val{Nontemporal})
end
@inline function vstore(v::Vec{N,T},
                        arr::FastContiguousArray{T,1},
                        i::Integer,
                        mask::Union{Vec{N,Bool}, Nothing},
                        ::Type{Val{Aligned}},
                        ::Type{Val{Nontemporal}} = Val{false}) where {N,T,Aligned,Nontemporal}
    vstore(v, arr, i, mask, Val(Aligned), Val(Nontemporal))
end
@inline function vstorea(v::Vec{N,T},
                         arr::FastContiguousArray{T,1},
                         i::Integer,
                         mask::Union{Vec{N,Bool}, Nothing}) where {N,T}
    vstore(v, arr, i, mask, Val{true})
end

export vgather, vgathera

@inline vgather(
        ::Type{Vec{N,T}}, ptrs::Vec{N,Ptr{T}}, mask::Nothing,
        ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vgather(Vec{N,T}, ptrs, Vec(ntuple(_ -> true, N)), Val(Aligned))
@inline vgather(
        ::Type{Vec{N,T}}, ptrs::Vec{N,Ptr{T}}, mask::Nothing,
        ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vgather(Vec{N,T}, ptrs, mask, Val(Aligned))

@generated function vgather(
        ::Type{Vec{N,T}}, ptrs::Vec{N,Ptr{T}}, mask::Vec{N,Bool},
        ::Val{Aligned} = Val(false)) where {N,T,Aligned}
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vptyp = "<$N x $typ*>"
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end

    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptrs = bitcast <$N x $typ*> %0 to $vptyp")
    else
        push!(instrs, "%ptrs = inttoptr <$N x $ptyp> %0 to $vptyp")
    end
    push!(instrs, "%mask = trunc $vbtyp %1 to <$N x i1>")
    push!(decls,
        "declare $vtyp @llvm.masked.gather.$(suffix(N,T))($vptyp, i32, " *
            "<$N x i1>, $vtyp)")
    push!(instrs,
        "%res = call $vtyp @llvm.masked.gather.$(suffix(N,T))($vptyp %ptrs, " *
            "i32 $align, <$N x i1> %mask, $vtyp $(llvmconst(N, T, 0)))")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,VE{T}}, Tuple{NTuple{N,VE{Ptr{T}}}, NTuple{N,VE{Bool}}},
            ptrs.elts, mask.elts))
    end
end
@inline function vgather(
        ::Type{Vec{N,T}}, ptrs::Vec{N,Ptr{T}}, mask::Vec{N,Bool},
        ::Type{Val{Aligned}}) where {N,T,Aligned}
    vgather(Vec{N,T}, ptrs, mask, Val(Aligned))
end

@inline vgathera(::Type{Vec{N,T}}, ptrs::Vec{N,Ptr{T}},
                 mask::Union{Vec{N,Bool}, Nothing}) where {N,T} =
    vgather(Vec{N,T}, ptrs, mask, Val{true})

@inline vgather(arr::FastContiguousArray{T,1},
                idx::Vec{N,<:Integer},
                mask::Union{Vec{N,Bool}, Nothing} = nothing,
                ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vgather(Vec{N,T},
            pointer(arr) + sizeof(T) * (idx - 1),
            mask, Val{Aligned})
@inline vgather(arr::FastContiguousArray{T,1},
                idx::Vec{N,<:Integer},
                mask::Union{Vec{N,Bool}, Nothing},
                ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vgather(arr, idx, mask, Val(Aligned))

@inline vgathera(arr::FastContiguousArray{T,1},
                 idx::Vec{N,<:Integer},
                 mask::Union{Vec{N,Bool}, Nothing} = nothing) where {N,T} =
    vgather(arr, idx, mask, Val{true})

export vscatter, vscattera

@inline vscatter(
        v::Vec{N,T}, ptrs::Vec{N,Ptr{T}}, mask::Nothing,
        ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vscatter(v, ptrs, Vec(ntuple(_ -> true, N)), Val(Aligned))
@inline vscatter(
        v::Vec{N,T}, ptrs::Vec{N,Ptr{T}}, mask::Nothing,
        ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vscatter(v, ptrs, mask, Val(Aligned))

@generated function vscatter(
        v::Vec{N,T}, ptrs::Vec{N,Ptr{T}}, mask::Vec{N,Bool},
        ::Val{Aligned} = Val(false)) where {N,T,Aligned}
    @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    vptyp = "<$N x $typ*>"
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    if VERSION < v"v0.7.0-DEV"
        push!(instrs, "%ptrs = bitcast <$N x $typ*> %1 to $vptyp")
    else
        push!(instrs, "%ptrs = inttoptr <$N x $ptyp> %1 to $vptyp")
    end
    push!(instrs, "%mask = trunc $vbtyp %2 to <$N x i1>")
    push!(decls,
        "declare void @llvm.masked.scatter.$(suffix(N,T))" *
            "($vtyp, $vptyp, i32, <$N x i1>)")
    push!(instrs,
        "call void @llvm.masked.scatter.$(suffix(N,T))" *
            "($vtyp %0, $vptyp %ptrs, i32 $align, <$N x i1> %mask)")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid,
            Tuple{NTuple{N,VE{T}}, NTuple{N,VE{Ptr{T}}}, NTuple{N,VE{Bool}}},
            v.elts, ptrs.elts, mask.elts)
    end
end
@inline function vscatter(
        v::Vec{N,T}, ptrs::Vec{N,Ptr{T}}, mask::Vec{N,Bool},
        ::Type{Val{Aligned}}) where {N,T,Aligned}
    vscatter(v, ptrs, mask, Val(Aligned))
end

@inline vscattera(v::Vec{N,T}, ptrs::Vec{N,Ptr{T}},
                  mask::Union{Vec{N,Bool}, Nothing}) where {N,T} =
    vscatter(v, ptrs, mask, Val{true})

@inline vscatter(v::Vec{N,T}, arr::FastContiguousArray{T,1},
                 idx::Vec{N,<:Integer},
                 mask::Union{Vec{N,Bool}, Nothing} = nothing,
                 ::Val{Aligned} = Val(false)) where {N,T,Aligned} =
    vscatter(v, pointer(arr) + sizeof(T) * (idx - 1), mask, Val(Aligned))
@inline vscatter(v::Vec{N,T}, arr::FastContiguousArray{T,1},
                 idx::Vec{N,<:Integer},
                 mask::Union{Vec{N,Bool}, Nothing},
                 ::Type{Val{Aligned}}) where {N,T,Aligned} =
    vscatter(v, arr, idx, mask, Val(Aligned))

@inline vscattera(v::Vec{N,T}, arr::FastContiguousArray{T,1},
                  idx::Vec{N,<:Integer},
                  mask::Union{Vec{N,Bool}, Nothing} = nothing) where {N,T} =
    vscatter(v, arr, idx, mask, Val{true})

# Vector shuffles

function shufflevector_instrs(N, T, I, two_operands)
    typ = llvmtype(T)
    vtyp2 = vtyp1 = "<$N x $typ>"
    M = length(I)
    vtyp3 = "<$M x i32>"
    vtypr = "<$M x $typ>"
    mask = "<" * join(map(x->string("i32 ", x), I), ", ") * ">"
    instrs = []
    v2 = two_operands ? "%1" : "undef"
    push!(instrs, "%res = shufflevector $vtyp1 %0, $vtyp2 $v2, $vtyp3 $mask")
    push!(instrs, "ret $vtypr %res")
    return M, [], instrs
end

export shufflevector
@generated function shufflevector(v1::Vec{N,T}, v2::Vec{N,T},
                                  ::Val{I}) where {N,T,I}
    M, decls, instrs = shufflevector_instrs(N, T, I, true)
    quote
        $(Expr(:meta, :inline))
        Vec{$M,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{$M,VE{T}},
            Tuple{NTuple{N,VE{T}}, NTuple{N,VE{T}}},
            v1.elts, v2.elts))
    end
end
@inline function shufflevector(v1::Vec{N,T}, v2::Vec{N,T},
                               ::Type{Val{I}}) where {N,T,I}
    shufflevector(v1, v2, Val(I))
end

@generated function shufflevector(v1::Vec{N,T}, ::Val{I}) where {N,T,I}
    M, decls, instrs = shufflevector_instrs(N, T, I, false)
    quote
        $(Expr(:meta, :inline))
        Vec{$M,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{$M,VE{T}},
            Tuple{NTuple{N,VE{T}}},
            v1.elts))
    end
end
@inline function shufflevector(v1::Vec{N,T}, ::Type{Val{I}}) where {N,T,I}
    shufflevector(v1, Val(I))
end

export VecRange

"""
    VecRange{N}(i::Int)

Analogous to `UnitRange` but for loading SIMD vector of width `N` at
index `i`.

# Examples
```jldoctest
julia> xs = ones(4);

julia> xs[VecRange{4}(1)]  # calls `vload(Vec{4,Float64}, xs, 1)`
<4 x Float64>[1.0, 1.0, 1.0, 1.0]
```
"""
struct VecRange{N}
    i::Int
end

@inline Base.length(idx::VecRange{N}) where {N} = N
@inline Base.first(idx::VecRange) = idx.i
@inline Base.last(idx::VecRange) = idx.i + length(idx) - 1

@inline Base.:+(idx::VecRange{N}, j::Integer) where N = VecRange{N}(idx.i + j)
@inline Base.:+(j::Integer, idx::VecRange{N}) where N = VecRange{N}(idx.i + j)
@inline Base.:-(idx::VecRange{N}, j::Integer) where N = VecRange{N}(idx.i - j)

Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, idx::VecRange) =
    (first(inds) <= first(idx)) && (last(idx) <= last(inds))

Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, idx::Vec) =
    all(first(inds) <= idx) && all(idx <= last(inds))

@inline _checkarity(::AbstractArray{<:Any,N}, ::Vararg{<:Any,N}) where {N} =
    nothing

@inline _checkarity(::T, ::Any) where {T <: AbstractArray} =
    if IndexStyle(T) isa IndexLinear
        nothing
    else
        throw(ArgumentError("""
        Array type $T does not support indexing with a single index.
        Exactly $(ndims(T)) (non-mask) indices have to be specified.
        """))
    end

_checkarity(::AbstractArray{<:Any,N}, ::Vararg{<:Any,M}) where {N,M} =
    throw(ArgumentError("""
    $M indices are given to $N-dimensional array.
    Exactly $N (non-mask) indices have to be specified when using SIMD.
    """))

# Combined with `_preprocessindices`, helper function `_extractmask`
# extracts `mask` in the tail position.  As slicing tuple is not
# type-stable, we use reverse-of-tail-of-reverse hack to extract
# `mask` at the end of `args`.
@inline _extractmask(mask::Vec{N,Bool}, R::Vararg{Integer}) where N =
    (reverse(R), mask)
@inline _extractmask(R::Vararg{Integer}) = (reverse(R), nothing)
@inline _extractmask(mask::Vec{N,Bool}) where {N} = ((), mask)
@inline _extractmask() = ((), nothing)

@noinline _extractmask(rargs...) =
    throw(ArgumentError("""
    Using SIMD indexing `array[idx, i2, ..., iN, mask]` for `N`-dimensional
    array requires `i2` to `iN` to be all integers and `mask` to be optionally
    a SIMD vector `Vec` of `Bool`s.  Given `(i2, ..., iN, mask)` is
    $(summary(reverse(rargs)))
    """))

_maskedidx(idx, ::Nothing, ::Any) = idx
_maskedidx(idx::Vec, mask::Vec, fst) = vifelse(mask, idx, fst)
_maskedidx(idx::VecRange, mask::Vec, fst) =
    _maskedidx(Vec(ntuple(i -> i - 1 + idx.i, length(mask))), mask, fst)

Base.@propagate_inbounds function _preprocessindices(arr, idx, args)
    I, mask = _extractmask(reverse(args)...)
    _checkarity(arr, idx, I...)
    @boundscheck checkbounds(arr,
                             _maskedidx(idx, mask, first(axes(arr, 1))),
                             I...)
    return I, mask
end

"""
    _pointer(arr, i, I)

Pointer to the element `arr[i, I...]`.
"""
Base.@propagate_inbounds _pointer(arr::Array, i, I) =
    pointer(arr, LinearIndices(arr)[i, I...])
Base.@propagate_inbounds _pointer(arr::Base.FastContiguousSubArray, i, I) =
    pointer(arr, (i, I...))
Base.@propagate_inbounds _pointer(arr::SubArray, i, I) =
    pointer(Base.unsafe_view(arr, 1, I...), i)

Base.@propagate_inbounds function Base.getindex(
        arr::ContiguousArray{T}, idx::VecRange{N},
        args::Vararg{Union{Integer,Vec{N,Bool}}}) where {N,T}
    I, mask = _preprocessindices(arr, idx, args)
    return vload(Vec{N,T}, _pointer(arr, idx.i, I), mask)
end

Base.@propagate_inbounds function Base.setindex!(
        arr::ContiguousArray{T}, v::Vec{N,T}, idx::VecRange{N},
        args::Vararg{Union{Integer,Vec{N,Bool}}}) where {N,T}
    I, mask = _preprocessindices(arr, idx, args)
    vstore(v, _pointer(arr, idx.i, I), mask)
    return arr
end

Base.@propagate_inbounds function Base.getindex(
        arr::ContiguousArray{T}, idx::Vec{N,<:Integer},
        args::Vararg{Union{Integer,Vec{N,Bool}}}) where {N,T}
    I, mask = _preprocessindices(arr, idx, args)
    ptrs = _pointer(arr, 1, I) - sizeof(T) + sizeof(T) * idx
    return vgather(Vec{N,T}, ptrs, mask)
end

Base.@propagate_inbounds function Base.setindex!(
        arr::ContiguousArray{T}, v::Vec{N,T}, idx::Vec{N,<:Integer},
        args::Vararg{Union{Integer,Vec{N,Bool}}}) where {N,T}
    I, mask = _preprocessindices(arr, idx, args)
    ptrs = _pointer(arr, 1, I) - sizeof(T) + sizeof(T) * idx
    vscatter(v, ptrs, mask)
    return arr
end

end
