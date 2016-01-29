module SIMD

# The Julia SIMD vector type

export Vec
immutable Vec{N,T} <: DenseArray{T,1}
    elts::NTuple{N,T}
    Vec(elts::NTuple{N,T}) = new(elts)
end

# Type properties

# Note: eltype and ndims are provided by DenseArray

import Base: length
length{N,T}(::Type{Vec{N,T}}) = N
length{N,T}(::Vec{N,T}) = N

import Base: size
size{N,T}(::Vec{N,T}) = (N,)
size{N,T}(::Type{Vec{N,T}}) = (N,)
size{N,T}(::Vec{N,T}, n::Int) = size(Vec{N,T})[n]
size{N,T}(::Type{Vec{N,T}}, n::Int) = size(Vec{N,T})[n]

# Type conversion

import Base: convert
@generated function create{N,T}(::Type{Vec{N,T}}, x::T)
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple, [:x for i in 1:N]...)))
    end
end
convert{N,T}(::Type{Vec{N,T}}, x::T) = create(Vec{N,T}, x)
convert{N,T}(::Type{Vec{N,T}}, x::Number) = create(Vec{N,T}, T(x))
convert{N,T}(::Type{Vec{N,T}}, xs::NTuple{N}) = Vec{N,T}(NTuple{N,T}(xs))

convert{N,T}(::Type{NTuple{N,T}}, v::Vec{N,T}) = v.elts

# Convert Julia types to LLVM types

llvmtype(::Type{Bool}) = "i8"   # Julia represents Tuple{Bool} as [1 x i8]

llvmtype(::Type{Int8}) = "i8"
llvmtype(::Type{Int16}) = "i16"
llvmtype(::Type{Int32}) = "i32"
llvmtype(::Type{Int64}) = "i64"
llvmtype(::Type{Int128}) = "i128"

llvmtype(::Type{UInt8}) = "i8"
llvmtype(::Type{UInt16}) = "i16"
llvmtype(::Type{UInt32}) = "i32"
llvmtype(::Type{UInt64}) = "i64"
llvmtype(::Type{UInt128}) = "i128"

llvmtype(::Type{Float16}) = "half"
llvmtype(::Type{Float32}) = "float"
llvmtype(::Type{Float64}) = "double"

# Type-dependent optimization flags
fastflags{T<:Signed}(::Type{T}) = "nsw"
fastflags{T<:Unsigned}(::Type{T}) = "nuw"
fastflags{T<:AbstractFloat}(::Type{T}) = "fast"

suffix{T}(N::Integer, ::Type{T}) = "v$(N)f$(8*sizeof(T))"

# Type-dependent LLVM constants
function llvmconst{T}(N::Integer, ::Type{T}, val)
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "<" * join(["$typ $val" for i in 1:N], ", ") * ">"
end
function llvmconst(N::Integer, ::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "<" * join(["$typ $(Int(val))" for i in 1:N], ", ") * ">"
end

# Type-dependent LLVM intrinsics
llvmins{T<:Integer}(::Type{Val{:+}}, N, ::Type{T}) = "add"
llvmins{T<:Integer}(::Type{Val{:-}}, N, ::Type{T}) = "sub"
llvmins{T<:Integer}(::Type{Val{:*}}, N, ::Type{T}) = "mul"
llvmins{T<:Signed}(::Type{Val{:div}}, N, ::Type{T}) = "sdiv"
llvmins{T<:Signed}(::Type{Val{:rem}}, N, ::Type{T}) = "srem"
llvmins{T<:Unsigned}(::Type{Val{:div}}, N, ::Type{T}) = "udiv"
llvmins{T<:Unsigned}(::Type{Val{:rem}}, N, ::Type{T}) = "urem"

llvmins{T<:Integer}(::Type{Val{:~}}, N, ::Type{T}) = "xor"
llvmins{T<:Integer}(::Type{Val{:&}}, N, ::Type{T}) = "and"
llvmins{T<:Integer}(::Type{Val{:|}}, N, ::Type{T}) = "or"
llvmins{T<:Integer}(::Type{Val{:$}}, N, ::Type{T}) = "xor"

llvmins{T<:Integer}(::Type{Val{:<<}}, N, ::Type{T}) = "shl"
llvmins{T<:Integer}(::Type{Val{:>>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:Unsigned}(::Type{Val{:>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:Signed}(::Type{Val{:>>}}, N, ::Type{T}) = "ashr"

llvmins{T<:Integer}(::Type{Val{:(==)}}, N, ::Type{T}) = "icmp eq"
llvmins{T<:Integer}(::Type{Val{:(!=)}}, N, ::Type{T}) = "icmp ne"
llvmins{T<:Signed}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp sgt"
llvmins{T<:Signed}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp sge"
llvmins{T<:Signed}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp slt"
llvmins{T<:Signed}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp sle"
llvmins{T<:Unsigned}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp ugt"
llvmins{T<:Unsigned}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp uge"
llvmins{T<:Unsigned}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp ult"
llvmins{T<:Unsigned}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp ule"

llvmins{T}(::Type{Val{:ifelse}}, N, ::Type{T}) = "select"

llvmins{T<:AbstractFloat}(::Type{Val{:+}}, N, ::Type{T}) = "fadd"
llvmins{T<:AbstractFloat}(::Type{Val{:-}}, N, ::Type{T}) = "fsub"
llvmins{T<:AbstractFloat}(::Type{Val{:*}}, N, ::Type{T}) = "fmul"
llvmins{T<:AbstractFloat}(::Type{Val{:/}}, N, ::Type{T}) = "fdiv"
llvmins{T<:AbstractFloat}(::Type{Val{:inv}}, N, ::Type{T}) = "fdiv"
llvmins{T<:AbstractFloat}(::Type{Val{:rem}}, N, ::Type{T}) = "frem"

llvmins{T<:AbstractFloat}(::Type{Val{:(==)}}, N, ::Type{T}) = "fcmp oeq"
llvmins{T<:AbstractFloat}(::Type{Val{:(!=)}}, N, ::Type{T}) = "fcmp une"
llvmins{T<:AbstractFloat}(::Type{Val{:(>)}}, N, ::Type{T}) = "fcmp ogt"
llvmins{T<:AbstractFloat}(::Type{Val{:(>=)}}, N, ::Type{T}) = "fcmp oge"
llvmins{T<:AbstractFloat}(::Type{Val{:(<)}}, N, ::Type{T}) = "fcmp olt"
llvmins{T<:AbstractFloat}(::Type{Val{:(<=)}}, N, ::Type{T}) = "fcmp ole"

llvmins{T<:AbstractFloat}(::Type{Val{:^}}, N, ::Type{T}) =
    "@llvm.pow.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:abs}}, N, ::Type{T}) =
    "@llvm.fabs.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:ceil}}, N, ::Type{T}) =
    "@llvm.ceil.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:copysign}}, N, ::Type{T}) =
    "@llvm.copysign.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:cos}}, N, ::Type{T}) =
    "@llvm.cos.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:exp}}, N, ::Type{T}) =
    "@llvm.exp.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:exp2}}, N, ::Type{T}) =
    "@llvm.exp2.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:floor}}, N, ::Type{T}) =
    "@llvm.floor.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:fma}}, N, ::Type{T}) =
    "@llvm.fma.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:log}}, N, ::Type{T}) =
    "@llvm.log.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:log10}}, N, ::Type{T}) =
    "@llvm.log10.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:log2}}, N, ::Type{T}) =
    "@llvm.log2.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:max}}, N, ::Type{T}) =
    "@llvm.maxnum.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:min}}, N, ::Type{T}) =
    "@llvm.minnum.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:muladd}}, N, ::Type{T}) =
    "@llvm.fmuladd.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:powi}}, N, ::Type{T}) =
    "@llvm.powi.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:round}}, N, ::Type{T}) =
    "@llvm.rint.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:sin}}, N, ::Type{T}) =
    "@llvm.sin.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:sqrt}}, N, ::Type{T}) =
    "@llvm.sqrt.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:trunc}}, N, ::Type{T}) =
    "@llvm.trunc.$(suffix(N,T))"

# Convert between LLVM scalars, vectors, and arrays

function scalar2vector(vec, siz, typ, sca)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_$i"
    for i in 0:siz-1
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $sca, i32 $i")
    end
    instrs
end

function scalar2array(varrec, siz, typ, sca)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_$i"
    for i in 0:siz-1
        push!(instrs,
            "$(accum(arr,i)) = " *
                "insertvalue [$siz x $typ] $(accum(arr,i-1)), $typ $sca, $i")
    end
    instrs
end

function array2vector(vec, siz, typ, arr, tmp=arr)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_$i"
    for i in 0:siz-1
        push!(instrs, "$(tmp)_$i = extractvalue [$siz x $typ] $arr, $i")
        push!(instrs,
            "$(accum(vec,i)) = " *
                "insertelement <$siz x $typ> $(accum(vec,i-1)), " *
                "$typ $(tmp)_$i, i32 $i")
    end
    instrs
end

function vector2array(arr, siz, typ, vec, tmp=vec)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_$i"
    for i in 0:siz-1
        push!(instrs, "$(tmp)_$i = extractelement <$siz x $typ> $vec, i32 $i")
        push!(instrs,
            "$(accum(arr,i)) = "*
                "insertvalue [$siz x $typ] $(accum(arr,i-1)), " *
                "$typ $(tmp)_$i, $i")
    end
    instrs
end

# Element-wise access

export setindex

@generated function setindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}, x)
    @assert isa(I, Integer)
    1 <= I <= N || throw(BoundsError())
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple,
            [i == I ? :(T(x)) : :(v.elts[$i]) for i in 1:N]...)))
    end
end

# Note: Julia has no equivalent of LLVM's "insertelement" function
@generated function setindex{N,T}(v::Vec{N,T}, i::Integer, x::Number)
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    push!(instrs, "%res = insertelement $vtyp %arg1, $typ %2, $ityp %1")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        let j = Int(i)
            @boundscheck 1 <= j <= N || throw(BoundsError())
            Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
                NTuple{N,T}, Tuple{NTuple{N,T}, Int, T}, v.elts, j-1, T(x)))
        end
    end
end

import Base: getindex
getindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}) = v.elts[I]
getindex{N,T}(v::Vec{N,T}, i::Integer) = v.elts[i]

# Type conversion

# Note: This can also change the number of vector elements if the size of the
# vector elements changes correspondingly; it is not possible to do this in
# plain Julia
@generated function Base.reinterpret{N,R,N1,T1}(::Type{Vec{N,R}},
        v1::Vec{N1,T1})
    @assert N*sizeof(R) == N1*sizeof(T1)
    typ1 = llvmtype(T1)
    atyp1 = "[$N1 x $typ1]"
    vtyp1 = "<$N1 x $typ1>"
    typr = llvmtype(R)
    atypr = "[$N x $typr]"
    vtypr = "<$N x $typr>"
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N1, typ1, "%0", "%arg1arr"))
    push!(instrs, "%res = bitcast $vtyp1 %arg1 to $vtypr")
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N1,T1}}, v1.elts))
    end
end

# Conditionals

for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
    @eval begin
        import Base: $op
        @generated function $op{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
            op = $op
            quote
                $(Expr(:meta, :inline))
                Vec{N,Bool}($(Expr(:tuple,
                    [:($op(v1.elts[$i], v2.elts[$i])) for i in 1:N]...)))
            end
        end
    end
end

import Base: ifelse
@generated function ifelse{N,T}(v1::Vec{N,Bool}, v2::Vec{N,T}, v3::Vec{N,T})
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple,
            [:(ifelse(v1.elts[$i], v2.elts[$i], v3.elts[$i]))
                for i in 1:N]...)))
    end
end

# Boolean functions

for op in (:!,)
    @eval begin
        import Base: $op
        @generated function $op{N}(v1::Vec{N,Bool})
            op = $op
            quote
                $(Expr(:meta, :inline))
                Vec{N,Bool}($(Expr(:tuple,
                    [:($op(v1.elts[$i])) for i in 1:N]...)))
            end
        end
    end
end

# Arithmetic functions

for op in (
        :~, :+, :-,
        :abs, :ceil, :cos, :exp, :exp2, :floor, :inv, :log, :log10, :log2,
        :round, :sin, :sqrt, :trunc)
    @eval begin
        import Base: $op
        @generated function $op{N,T}(v1::Vec{N,T})
            op = $op
            quote
                $(Expr(:meta, :inline))
                Vec{N,T}($(Expr(:tuple, [:($op(v1.elts[$i])) for i in 1:N]...)))
            end
        end
    end
end
import Base: exp10
@inline exp10{N,T}(v1::Vec{N,T}) = Vec{N,T}(10)^v1

for op in (
        :&, :|, :$, :+, :-, :*, :/, :^,
        :cld, :copysign, :div, :fld, :max, :min, :mod, :rem)
    @eval begin
        import Base: $op
        @generated function $op{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
            op = $op
            quote
                $(Expr(:meta, :inline))
                Vec{N,T}($(Expr(:tuple,
                    [:($op(v1.elts[$i], v2.elts[$i])) for i in 1:N]...)))
            end
        end
    end
end

for op in (:fma, :muladd)
    @eval begin
        import Base: $op
        @generated function $op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}, v3::Vec{N,T})
            op = $op
            quote
                $(Expr(:meta, :inline))
                Vec{N,T}($(Expr(:tuple,
                    [:($op(v1.elts[$i], v2.elts[$i], v3.elts[$i]))
                        for i in 1:N]...)))
            end
        end
    end
end

# TODO: add missing functions
# TODO: add load/store functions

end
