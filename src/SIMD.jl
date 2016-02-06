module SIMD

#=

# Various boolean types

abstract Boolean <: Integer

for sz in (8, 16, 32, 64, 128)
    Intsz = symbol(:Int, sz)
    UIntsz = symbol(:UInt, sz)
    Boolsz = symbol(:Bool, sz)
    @eval begin
        immutable $Boolsz <: Boolean
            int::$UIntsz
            $Boolsz(b::Bool) =
                new(ifelse(b, typemax($UIntsz), typemin($UIntsz)))
        end
        booltype(::Type{Val{$sz}}) = $Boolsz
        inttype(::Type{Val{$sz}}) = $Intsz
        uinttype(::Type{Val{$sz}}) = $UIntsz

        Base.convert(::Type{Bool}, b::$Boolsz) = b.int != 0

        Base. ~(b::$Boolsz) = $Boolsz(~b.int)
        Base. !(b::$Boolsz) = ~b
        Base. &(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int & b2.int)
        Base. |(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int | b2.int)
        Base.$(:$)(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int $ b2.int)

        Base. ==(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int == b2.int)
        Base. !=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int != b2.int)
        Base. <(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int < b2.int)
        Base. <=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int <= b2.int)
        Base. >(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int > b2.int)
        Base. >=(b1::$Boolsz, b2::$Boolsz) = $Boolsz(b1.int >= b2.int)
    end
end
Base.convert(::Type{Bool}, b::Boolean) = error("impossible")
Base.convert{I<:Integer}(::Type{I}, b::Boolean) = I(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, b::Boolean) = B(Bool(b))
Base.convert{B<:Boolean}(::Type{B}, i::Integer) = B(i!=0)

booltype{T}(::Type{T}) = booltype(Val{8*sizeof(T)})
inttype{T}(::Type{T}) = inttype(Val{8*sizeof(T)})
uinttype{T}(::Type{T}) = uinttype(Val{8*sizeof(T)})

=#

# The Julia SIMD vector type

const BoolTypes = Union{Bool}
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{BoolTypes, IntTypes, UIntTypes}
const FloatingTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{IntegerTypes, FloatingTypes}

export Vec
immutable Vec{N,T<:ScalarTypes} <: DenseArray{T,1}   # <: Number
    elts::NTuple{N,T}
    Vec(elts::NTuple{N,T}) = new(elts)
end

function Base.show{N,T}(io::IO, v::Vec{N,T})
    print(io, T, "<")
    for i in 1:N
        i>1 && print(io, ",")
        print(io, v.elts[i])
    end
    print(io, ">")
end

# Base.print_matrix wants to access a second dimension that doesn't exist for
# Vec. (In Julia, every array can be accessed as N-dimensional array, for
# arbitrary N.) Instead of implementing this, output our Vec the usual way.
function Base.print_matrix(io::IO, X::Vec,
                      pre::AbstractString = " ",  # pre-matrix string
                      sep::AbstractString = "  ", # separator between elements
                      post::AbstractString = "",  # post-matrix string
                      hdots::AbstractString = "  \u2026  ",
                      vdots::AbstractString = "\u22ee",
                      ddots::AbstractString = "  \u22f1  ",
                      hmod::Integer = 5, vmod::Integer = 5)
    print(io, X)
end

# Type properties

# eltype and ndims is provided by DenseArray
# Base.eltype{N,T}(::Type{Vec{N,T}}) = T
# Base.ndims{N,T}(::Type{Vec{N,T}}) = 1
Base.length{N,T}(::Type{Vec{N,T}}) = N
Base.size{N,T}(::Type{Vec{N,T}}) = (N,)
Base.size{N,T}(::Type{Vec{N,T}}, n::Integer) = (N,)[n]
# Base.eltype{N,T}(::Vec{N,T}) = T
# Base.ndims{N,T}(::Vec{N,T}) = 1
Base.length{N,T}(::Vec{N,T}) = N
Base.size{N,T}(::Vec{N,T}) = (N,)
Base.size{N,T}(::Vec{N,T}, n::Integer) = (N,)[n]

# Type conversion

@generated function create{N,T<:ScalarTypes}(::Type{Vec{N,T}}, x::T)
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple, [:x for i in 1:N]...)))
    end
end
Base.convert{N,T}(::Type{Vec{N,T}}, x::T) = create(Vec{N,T}, x)
Base.convert{N,T}(::Type{Vec{N,T}}, x::Number) = create(Vec{N,T}, T(x))
Base.convert{N,T}(::Type{Vec{N,T}}, xs::NTuple{N}) = Vec{N,T}(NTuple{N,T}(xs))

Base.convert{N,T}(::Type{NTuple{N,T}}, v::Vec{N,T}) = v.elts

# Promotion rules

# Type promotion only works for subtypes of Number!
# Base.promote_rule{N,T<:ScalarTypes}(::Type{Vec{N,T}}, ::Type{T}) = Vec{N,T}

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

exponent_bits{T<:FloatingTypes}(::Type{T}) = 8*sizeof(T) - 1 - significand_bits(T)
sign_bits{T<:FloatingTypes}(::Type{T}) = 1

significand_mask{T<:FloatingTypes}(::Type{T}) =
    uint_type(T)(uint_type(T)(1) << significand_bits(T) - 1)
exponent_mask{T<:FloatingTypes}(::Type{T}) =
    uint_type(T)(uint_type(T)(1) << exponent_bits(T) - 1) << significand_bits(T)
sign_mask{T<:FloatingTypes}(::Type{T}) =
    uint_type(T)(1) << (significand_bits(T) + exponent_bits(T))

for T in (Float16, Float32, Float64)
    @assert sizeof(int_type(T)) == sizeof(T)
    @assert sizeof(uint_type(T)) == sizeof(T)
    @assert significand_bits(T) + exponent_bits(T) + sign_bits(T) == 8*sizeof(T)
    @assert significand_mask(T) | exponent_mask(T) | sign_mask(T) ==
        typemax(uint_type(T))
    @assert significand_mask(T) $ exponent_mask(T) $ sign_mask(T) ==
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

llvmtype(::Type{UInt8}) = "i8"
llvmtype(::Type{UInt16}) = "i16"
llvmtype(::Type{UInt32}) = "i32"
llvmtype(::Type{UInt64}) = "i64"
llvmtype(::Type{UInt128}) = "i128"

llvmtype(::Type{Float16}) = "half"
llvmtype(::Type{Float32}) = "float"
llvmtype(::Type{Float64}) = "double"

# Type-dependent optimization flags
fastflags{T<:IntTypes}(::Type{T}) = "nsw"
fastflags{T<:UIntTypes}(::Type{T}) = "nuw"
fastflags{T<:FloatingTypes}(::Type{T}) = "fast"

suffix{T<:IntegerTypes}(N::Integer, ::Type{T}) = "v$(N)i$(8*sizeof(T))"
suffix{T<:FloatingTypes}(N::Integer, ::Type{T}) = "v$(N)f$(8*sizeof(T))"

# Type-dependent LLVM constants
function llvmconst{T}(::Type{T}, val)
    T(val) === T(0) && return "zeroinitializer"
    typ = llvmtype(T)
    "$typ $val"
end
function llvmconst(::Type{Bool}, val)
    Bool(val) === false && return "zeroinitializer"
    typ = "i1"
    "$typ $(Int(val))"
end
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
llvmins{T<:IntegerTypes}(::Type{Val{:+}}, N, ::Type{T}) = "add"
llvmins{T<:IntegerTypes}(::Type{Val{:-}}, N, ::Type{T}) = "sub"
llvmins{T<:IntegerTypes}(::Type{Val{:*}}, N, ::Type{T}) = "mul"
llvmins{T<:IntTypes}(::Type{Val{:div}}, N, ::Type{T}) = "sdiv"
llvmins{T<:IntTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "srem"
llvmins{T<:UIntTypes}(::Type{Val{:div}}, N, ::Type{T}) = "udiv"
llvmins{T<:UIntTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "urem"

llvmins{T<:IntegerTypes}(::Type{Val{:~}}, N, ::Type{T}) = "xor"
llvmins{T<:IntegerTypes}(::Type{Val{:&}}, N, ::Type{T}) = "and"
llvmins{T<:IntegerTypes}(::Type{Val{:|}}, N, ::Type{T}) = "or"
llvmins{T<:IntegerTypes}(::Type{Val{:$}}, N, ::Type{T}) = "xor"

llvmins{T<:IntegerTypes}(::Type{Val{:<<}}, N, ::Type{T}) = "shl"
llvmins{T<:IntegerTypes}(::Type{Val{:>>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:UIntTypes}(::Type{Val{:>>}}, N, ::Type{T}) = "lshr"
llvmins{T<:IntTypes}(::Type{Val{:>>}}, N, ::Type{T}) = "ashr"

llvmins{T<:IntegerTypes}(::Type{Val{:(==)}}, N, ::Type{T}) = "icmp eq"
llvmins{T<:IntegerTypes}(::Type{Val{:(!=)}}, N, ::Type{T}) = "icmp ne"
llvmins{T<:IntTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp sgt"
llvmins{T<:IntTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp sge"
llvmins{T<:IntTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp slt"
llvmins{T<:IntTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp sle"
llvmins{T<:UIntTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "icmp ugt"
llvmins{T<:UIntTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "icmp uge"
llvmins{T<:UIntTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "icmp ult"
llvmins{T<:UIntTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "icmp ule"

llvmins{T}(::Type{Val{:ifelse}}, N, ::Type{T}) = "select"

llvmins{T<:FloatingTypes}(::Type{Val{:+}}, N, ::Type{T}) = "fadd"
llvmins{T<:FloatingTypes}(::Type{Val{:-}}, N, ::Type{T}) = "fsub"
llvmins{T<:FloatingTypes}(::Type{Val{:*}}, N, ::Type{T}) = "fmul"
llvmins{T<:FloatingTypes}(::Type{Val{:/}}, N, ::Type{T}) = "fdiv"
llvmins{T<:FloatingTypes}(::Type{Val{:inv}}, N, ::Type{T}) = "fdiv"
llvmins{T<:FloatingTypes}(::Type{Val{:rem}}, N, ::Type{T}) = "frem"

llvmins{T<:FloatingTypes}(::Type{Val{:(==)}}, N, ::Type{T}) = "fcmp oeq"
llvmins{T<:FloatingTypes}(::Type{Val{:(!=)}}, N, ::Type{T}) = "fcmp une"
llvmins{T<:FloatingTypes}(::Type{Val{:(>)}}, N, ::Type{T}) = "fcmp ogt"
llvmins{T<:FloatingTypes}(::Type{Val{:(>=)}}, N, ::Type{T}) = "fcmp oge"
llvmins{T<:FloatingTypes}(::Type{Val{:(<)}}, N, ::Type{T}) = "fcmp olt"
llvmins{T<:FloatingTypes}(::Type{Val{:(<=)}}, N, ::Type{T}) = "fcmp ole"

llvmins{T<:FloatingTypes}(::Type{Val{:^}}, N, ::Type{T}) =
    "@llvm.pow.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:abs}}, N, ::Type{T}) =
    "@llvm.fabs.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:ceil}}, N, ::Type{T}) =
    "@llvm.ceil.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:copysign}}, N, ::Type{T}) =
    "@llvm.copysign.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:cos}}, N, ::Type{T}) =
    "@llvm.cos.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:exp}}, N, ::Type{T}) =
    "@llvm.exp.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:exp2}}, N, ::Type{T}) =
    "@llvm.exp2.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:floor}}, N, ::Type{T}) =
    "@llvm.floor.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:fma}}, N, ::Type{T}) =
    "@llvm.fma.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log}}, N, ::Type{T}) =
    "@llvm.log.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log10}}, N, ::Type{T}) =
    "@llvm.log10.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:log2}}, N, ::Type{T}) =
    "@llvm.log2.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:max}}, N, ::Type{T}) =
    "@llvm.maxnum.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:min}}, N, ::Type{T}) =
    "@llvm.minnum.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:muladd}}, N, ::Type{T}) =
    "@llvm.fmuladd.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:powi}}, N, ::Type{T}) =
    "@llvm.powi.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:round}}, N, ::Type{T}) =
    "@llvm.rint.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:sin}}, N, ::Type{T}) =
    "@llvm.sin.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:sqrt}}, N, ::Type{T}) =
    "@llvm.sqrt.$(suffix(N,T))"
llvmins{T<:FloatingTypes}(::Type{Val{:trunc}}, N, ::Type{T}) =
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

function scalar2array(varrec, siz, typ, sca)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    for i in 0:siz-1
        push!(instrs,
            "$(accum(arr,i)) = " *
                "insertvalue [$siz x $typ] $(accum(arr,i-1)), $typ $sca, $i")
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
            "$(accum(rvec,i)) = "*
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

function extendvector(vec, siz, typ, voff, vsiz, val, rvec)
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    rsiz = siz1 + siz2
    for i in 0:siz-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz1 x $typ> $vec1, i32 $i")
        push!(instrs,
            "$(accum(rvec,i)) = "*
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    for i in siz:siz+vsiz-1
        push!(instrs,
            "$(accum(rvec,i)) = "*
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), $val, i32 $i")
    end
    instrs
end

function combinevectors(vec1, siz1, vec2, siz2, typ, rvec, tmp="$(rvec)_cv")
    instrs = []
    accum(nam, i) = i<0 ? "undef" : i==siz-1 ? nam : "$(nam)_iter$i"
    rsiz = siz1 + siz2
    for i in 0:siz1-1
        push!(instrs,
            "$(tmp)_elem$i = extractelement <$siz1 x $typ> $vec1, i32 $i")
        push!(instrs,
            "$(accum(rvec,i)) = "*
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, $i")
    end
    for i in siz1:siz2-1
        push!(instrs,
            "$(tmp)_elem$i = " *
                "extractelement <$siz2 x $typ> $vec2, i32 $(i-siz1)")
        push!(instrs,
            "$(accum(rvec,i)) = "*
                "insertelement <$rsiz x $typ> $(accum(rvec,i-1)), " *
                "$typ $(tmp)_elem$i, i32 $i")
    end
    instrs
end

# Element-wise access

export setindex
@generated function setindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}, x::Number)
    @assert isa(I, Integer)
    1 <= I <= N || throw(BoundsError())
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    decls = []
    instrs = []
    push!(instrs, "%resarr = insertvalue $atyp %0, $typ %1, $(I-1)")
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,T}, T}, v.elts, T(x)))
    end
end

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

Base.getindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}) = v.elts[I]
Base.getindex{N,T}(v::Vec{N,T}, i::Integer) = v.elts[i]

# Type conversion

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

# Generic function wrappers

@generated function llvmwrap{Op,N,T1,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    atyp1 = "[$N x $typ1]"
    vtyp1 = "<$N x $typ1>"
    typr = llvmtype(R)
    atypr = "[$N x $typr]"
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1)")
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
        push!(instrs, "%res = $ins $vtyp1 $otherarg, %arg1")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}}, v1.elts))
    end
end

@generated function llvmwrap{Op,N}(::Type{Val{Op}}, v1::Vec{N,Bool},
        ::Type{Bool} = Bool)
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    ins = llvmins(Val{Op}, N, Bool)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1b", N, btyp, "%0", "%arg1arr"))
    push!(instrs, "%arg1 = trunc $vbtyp %arg1b to <$N x i1>")
    otherarg = llvmconst(N, Bool, true)
    push!(instrs, "%res = $ins <$N x i1> $otherarg, %arg1")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    append!(instrs, vector2array("%resarr", N, btyp, "%resb"))
    push!(instrs, "ret $abtyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,Bool}, Tuple{NTuple{N,Bool}}, v1.elts))
    end
end

@generated function llvmwrap{Op,N,T1,T2,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    atyp1 = "[$N x $typ1]"
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    atyp2 = "[$N x $typ2]"
    vtyp2 = "<$N x $typ2>"
    typr = llvmtype(R)
    atypr = "[$N x $typr]"
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1, $vtyp2 %arg2)")
    else
        push!(instrs, "%res = $ins $vtyp1 %arg1, %arg2")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}, NTuple{N,T2}}, v1.elts, v2.elts))
    end
end

@generated function llvmwrap{Op,N,T1,T2}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, ::Type{Bool})
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ1 = llvmtype(T1)
    atyp1 = "[$N x $typ1]"
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    atyp2 = "[$N x $typ2]"
    vtyp2 = "<$N x $typ2>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
    push!(instrs, "%cond = $ins $vtyp1 %arg1, %arg2")
    push!(instrs, "%res = zext <$N x i1> %cond to $vbtyp")
    append!(instrs, vector2array("%resarr", N, btyp, "%res"))
    push!(instrs, "ret $abtyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,Bool}, Tuple{NTuple{N,T1}, NTuple{N,T2}},
            v1.elts, v2.elts))
    end
end

@generated function llvmwrap{Op,N,T1,T2,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        x2::T2, ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    atyp1 = "[$N x $typ1]"
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    typr = llvmtype(R)
    atypr = "[$N x $typr]"
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $typ2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1, $typ2 %1)")
    else
        push!(instrs, "%res = $ins $vtyp1 %arg1, %1")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}, T2}, v1.elts, x2))
    end
end

@generated function llvmwrap{Op,N}(::Type{Val{Op}}, v1::Vec{N,Bool},
        v2::Vec{N,Bool}, ::Type{Bool} = Bool)
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    ins = llvmins(Val{Op}, N, Bool)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1b", N, btyp, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2b", N, btyp, "%1", "%arg2arr"))
    push!(instrs, "%arg1 = trunc $vbtyp %arg1b to <$N x i1>")
    push!(instrs, "%arg2 = trunc $vbtyp %arg2b to <$N x i1>")
    push!(instrs, "%res = $ins <$N x i1> %arg1, %arg2")
    push!(instrs, "%resb = zext <$N x i1> %res to $vbtyp")
    append!(instrs, vector2array("%resarr", N, btyp, "%resb"))
    push!(instrs, "ret $abtyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,Bool}, Tuple{NTuple{N,Bool}, NTuple{N,Bool}},
            v1.elts, v2.elts))
    end
end

@generated function llvmwrap{Op,N,T1,T2,T3,R}(::Type{Val{Op}}, v1::Vec{N,T1},
        v2::Vec{N,T2}, v3::Vec{N,T3}, ::Type{R} = T1)
    @assert isa(Op, Symbol)
    typ1 = llvmtype(T1)
    atyp1 = "[$N x $typ1]"
    vtyp1 = "<$N x $typ1>"
    typ2 = llvmtype(T2)
    atyp2 = "[$N x $typ2]"
    vtyp2 = "<$N x $typ2>"
    typ3 = llvmtype(T3)
    atyp3 = "[$N x $typ3]"
    vtyp3 = "<$N x $typ3>"
    typr = llvmtype(R)
    atypr = "[$N x $typr]"
    vtypr = "<$N x $typr>"
    ins = llvmins(Val{Op}, N, T1)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
    append!(instrs, array2vector("%arg3", N, typ3, "%2", "%arg3arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2, $vtyp3)")
        push!(instrs,
            "%res = call $vtypr $ins($vtyp1 %arg1, $vtyp2 %arg2, $vtyp3 %arg3)")
    else
        push!(instrs, "%res = $ins $vtyp1 %arg1, %arg2, %arg3")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}, NTuple{N,T2}, NTuple{N,T3}},
            v1.elts, v2.elts, v3.elts))
    end
end

@generated function llvmwrapshift{Op,N,T,I}(::Type{Val{Op}}, v1::Vec{N,T},
        ::Type{Val{I}})
    @assert isa(Op, Symbol)
    @assert isa(I, Integer)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    nbits = 8*sizeof(T)
    if (Op === :>> && T <: IntTypes) || (I>=0 && I<nbits)
        append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
        count = llvmconst(N, T, I>=0 && I<nbits ? I : nbits-1)
        push!(instrs, "%res = $ins $vtyp %arg1, $count")
        append!(instrs, vector2array("%resarr", N, typ, "%res"))
        push!(instrs, "ret $atyp %resarr")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "return $atyp $zero")
    end
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,T}}, v1.elts))
    end
end

@generated function llvmwrapshift{Op,N,T}(::Type{Val{Op}}, v1::Vec{N,T},
        x2::T)
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    append!(instrs, scalar2vector("%arg2", N, typ, "%1"))
    nbits = 8*sizeof(T)
    push!(instrs, "%tmp = $ins $vtyp %arg1, %arg2")
    push!(instrs, "%inbounds = icmp ult $typ %1, $nbits")
    if Op === :>> && T <: IntTypes
        nbits = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %arg1, $nbits")
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "%res = select i1 %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,T}, T}, v1.elts, x2))
    end
end

@generated function llvmwrapshift{Op,N,T}(::Type{Val{Op}}, v1::Vec{N,T},
        v2::Vec{N,T})
    @assert isa(Op, Symbol)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
    push!(instrs, "%tmp = $ins $vtyp %arg1, %arg2")
    nbits = llvmconst(N, T, 8*sizeof(T))
    push!(instrs, "%inbounds = icmp ult $vtyp %arg2, $nbits")
    if Op === :>> && T <: IntTypes
        nbits = llvmconst(N, T, 8*sizeof(T)-1)
        push!(instrs, "%limit = $ins $vtyp %arg1, $nbits")
        push!(instrs, "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp %limit")
    else
        zero = llvmconst(N, T, 0)
        push!(instrs, "%res = select <$N x i1> %inbounds, $vtyp %tmp, $vtyp $zero")
    end
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts))
    end
end

# Conditionals

for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
    @eval begin
        @inline Base.$op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, Bool)
    end
end
# @inline Base.isfinite{N,T<:FloatingTypes}(v1::Vec{N,T}) =
#     !(isinf(v1) | isnan(v1))
@inline function Base.isfinite{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & em != em
end
@inline Base.isinf{N,T<:FloatingTypes}(v1::Vec{N,T}) = abs(v1) == Vec{N,T}(Inf)
# @inline function Base.isinf{N,T<:FloatingTypes}(v1::Vec{N,T})
#     U = uint_type(T)
#     em = Vec{N,U}(exponent_mask(T))
#     sm = Vec{N,U}(significand_mask(T))
#     iv = reinterpret(Vec{N,U}, v1)
#     iv & (em|sm) == em
# end
@inline Base.isnan{N,T<:FloatingTypes}(v1::Vec{N,T}) = v1 != v1
# @inline function Base.isnan{N,T<:FloatingTypes}(v1::Vec{N,T})
#     U = uint_type(T)
#     em = Vec{N,U}(exponent_mask(T))
#     sm = Vec{N,U}(significand_mask(T))
#     iv = reinterpret(Vec{N,U}, v1)
#     (iv & em == em)  & (iv & sm != Vec{N,U}(0))
# end
# @inline Base.isnormal{N,T<:FloatingTypes}(v1::Vec{N,T}) = ???
@inline function Base.issubnormal{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    em = Vec{N,U}(exponent_mask(T))
    sm = Vec{N,U}(significand_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    (iv & em == Vec{N,U}(0)) & (iv & sm != Vec{N,U}(0))
end
@inline function Base.signbit{N,T<:FloatingTypes}(v1::Vec{N,T})
    U = uint_type(T)
    sm = Vec{N,U}(sign_mask(T))
    iv = reinterpret(Vec{N,U}, v1)
    iv & sm != Vec{N,U}(0)
end

#=
@generated function Base.ifelse{N,T}(x1::Bool, v2::Vec{N,T}, v3::Vec{N,T})
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
    append!(instrs, array2vector("%arg3", N, typ, "%2", "%arg3arr"))
    push!(instrs, "%res = select i1 %0, $vtyp %arg2, $vtyp %arg3")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{Bool, NTuple{N,T}, NTuple{N,T}},
            x1, v2.elts, v3.elts))
    end
end
=#

@generated function Base.ifelse{N,T}(v1::Vec{N,Bool}, v2::Vec{N,T},
        v3::Vec{N,T})
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, btyp, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
    append!(instrs, array2vector("%arg3", N, typ, "%2", "%arg3arr"))
    push!(instrs, "%cond = trunc $vbtyp %arg1 to <$N x i1>")
    push!(instrs, "%res = select <$N x i1> %cond, $vtyp %arg2, $vtyp %arg3")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,Bool}, NTuple{N,T}, NTuple{N,T}},
            v1.elts, v2.elts, v3.elts))
    end
end

# Integer arithmetic functions

for op in (:~, :+, :-)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end
@inline Base. !{N}(v1::Vec{N,Bool}) = ~v1
@inline function Base.abs{N,T<:IntTypes}(v1::Vec{N,T})
    # s = -Vec{N,T}(signbit(v1))
    s = v1 >> Val{8*sizeof(T)}
    # Note: -v1 == ~v1 + 1
    (s $ v1) - s
end
@inline Base.abs{N,T<:UIntTypes}(v1::Vec{N,T}) = v1
@inline Base.sign{N,T<:IntTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0), Vec{N,T}(0),
        ifelse(v1 < Vec{N,T}(0), Vec{N,T}(-1), Vec{N,T}(1)))
@inline Base.sign{N,T<:UIntTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0), Vec{N,T}(0), Vec{N,T}(1))
@inline Base.signbit{N,T<:IntTypes}(v1::Vec{N,T}) = v1 < Vec{N,T}(0)
@inline Base.signbit{N,T<:UIntTypes}(v1::Vec{N,T}) = Vec{N,Bool}(false)
# @inline Base.signbit{N,T<:IntTypes}(v1::Vec{N,T}) = v1 >> Val{8*sizeof(T)}

for op in (:&, :|, :$, :+, :-, :*, :div, :rem)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end
@inline Base.copysign{N,T<:IntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -abs(v1), abs(v1))
@inline Base.copysign{N,T<:UIntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) = v1
@inline Base.flipsign{N,T<:IntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -v1, v1)
@inline Base.flipsign{N,T<:UIntTypes}(v1::Vec{N,T}, v2::Vec{N,T}) = v1
@inline Base.max{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(v1>=v2, v1, v2)
@inline Base.min{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(v1>=v2, v2, v1)

@inline function Base.muladd{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T},
        v3::Vec{N,T})
    v1*v2+v3
end

for op in (:<<, :>>, :>>>)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes,I}(v1::Vec{N,T}, ::Type{Val{I}}) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, Val{I})
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, x2::Int) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, T(x2))
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, x2::Integer) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, T(x2))
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrapshift(Val{$(QuoteNode(op))}, v1, v2)
        @inline Base.$op{N,T<:IntegerTypes}(x1::T, v2::Vec{N,T}) =
            $op(Vec{N,T}(x1), v2)
    end
end

# Floating point arithmetic functions

for op in (
        :+, :-,
        :abs, :ceil, :cos, :exp, :exp2, :floor, :inv, :log, :log10, :log2,
        :round, :sin, :sqrt, :trunc)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end
@inline Base.exp10{N,T<:FloatingTypes}(v1::Vec{N,T}) = Vec{N,T}(10)^v1
@inline Base.sign{N,T<:FloatingTypes}(v1::Vec{N,T}) =
    ifelse(v1 == Vec{N,T}(0.0), Vec{N,T}(0.0), copysign(Vec{N,T}(1.0), v1))

for op in (:+, :-, :*, :/, :^, :copysign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end
@inline Base. ^{N,T<:FloatingTypes}(v1::Vec{N,T},x2::Integer) =
    llvmwrap(Val{:powi}, v1, Int(x2))
@inline Base.flipsign{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T}) =
    ifelse(signbit(v2), -v1, v1)

for op in (:fma, :muladd)
    @eval begin
        @inline function Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T},
                v3::Vec{N,T})
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, v3)
        end
    end
end

# Type promotion

# Promote scalars of all Integertypes to vectors of IntegerTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :&, :|, :$, :+, :-, :*, :copysign, :div, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N}(s1::Bool, v2::Vec{N,Bool}) =
            $op(Vec{N,Bool}(s1), v2)
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes) =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline Base.ifelse{N,T<:IntegerTypes}(c::Vec{N,Bool}, s1::IntegerTypes,
        v2::Vec{N,T}) =
    ifelse(c, Vec{N,T}(s1), v2)
@inline Base.ifelse{N,T<:IntegerTypes}(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::IntegerTypes) =
    ifelse(c, v1, Vec{N,T}(s2))

for op in (:muladd,)
    @eval begin
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T},
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes,
                v3::Vec{N,T}) =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, s2::IntegerTypes,
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, v2::Vec{N,T},
                s3::IntegerTypes) =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:IntegerTypes}(s1::IntegerTypes, v2::Vec{N,T},
                s3::IntegerTypes) =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:IntegerTypes}(v1::Vec{N,T}, s2::IntegerTypes,
                s3::IntegerTypes) =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Promote scalars of all ScalarTypes to vectors of FloatingTypes, leaving the
# vector type unchanged

for op in (
        :(==), :(!=), :(<), :(<=), :(>), :(>=),
        :+, :-, :*, :/, :^, :copysign, :flipsign, :max, :min, :rem)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes) =
            $op(v1, Vec{N,T}(s2))
    end
end
@inline Base.ifelse{N,T<:FloatingTypes}(c::Vec{N,Bool}, s1::ScalarTypes,
        v2::Vec{N,T}) =
    ifelse(c, Vec{N,T}(s1), v2)
@inline Base.ifelse{N,T<:FloatingTypes}(c::Vec{N,Bool}, v1::Vec{N,T},
        s2::ScalarTypes) =
    ifelse(c, v1, Vec{N,T}(s2))

for op in (:fma, :muladd)
    @eval begin
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T},
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes,
                v3::Vec{N,T}) =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, s2::ScalarTypes,
                v3::Vec{N,T}) =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, v2::Vec{N,T},
                s3::ScalarTypes) =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:FloatingTypes}(s1::ScalarTypes, v2::Vec{N,T},
                s3::ScalarTypes) =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op{N,T<:FloatingTypes}(v1::Vec{N,T}, s2::ScalarTypes,
                s3::ScalarTypes) =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

# Reduction operations

function getneutral{T}(op::Symbol, ::Type{T})
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

# We cannot pass in the neutral element via Val{}; if we try, Julia refused to
# inline this function, which is then disastrous for performance
@generated function llvmwrapreduce{Op,N,T}(::Type{Val{Op}}, v::Vec{N,T})
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    typ = llvmtype(T)
    decls = []
    instrs = []
    n = N
    append!(instrs, array2vector("%vec_$n", N, typ, "%0", "%argarr"))
    nold,n = n,nextpow2(n)
    if n > nold
        append!(instrs,
            extendvector("%vec_$nold", nold, typ, n, n-nold, llvmconst(T,z),
                "%vec_$n"))
    end
    while n > 1
        nold,n = n, div(n, 2)
        atyp = "[$n x $typ]"
        vtyp = "<$n x $typ>"
        ins = llvmins(Val{Op}, n, T)
        append!(instrs, subvector("%vec_$nold", nold, typ, "%vec_$(n)_1", n, 0))
        append!(instrs, subvector("%vec_$nold", nold, typ, "%vec_$(n)_2", n, n))
        if ins[1] == '@'
            push!(decls, "declare $vtyp $ins($vtyp, $vtyp)")
            push!(instrs,
                "%vec_$n = " *
                    "call $vtyp $ins($vtyp %vec_$(n)_1, $vtyp %vec_$(n)_2)")
        else
            push!(instrs, "%vec_$n = $ins $vtyp %vec_$(n)_1, %vec_$(n)_2")
        end
    end
    push!(instrs, "%res = extractelement <$n x $typ> %vec_$n, i32 0")
    push!(instrs, "ret $typ %res")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            T, Tuple{NTuple{N,T}}, v.elts)
    end
end

@inline Base.all{N,T<:IntegerTypes}(v::Vec{N,T}) = llvmwrapreduce(Val{:&}, v)
@inline Base.any{N,T<:IntegerTypes}(v::Vec{N,T}) = llvmwrapreduce(Val{:|}, v)
@inline Base.maximum{N,T<:FloatingTypes}(v::Vec{N,T}) =
    llvmwrapreduce(Val{:max}, v)
@inline Base.minimum{N,T<:FloatingTypes}(v::Vec{N,T}) =
    llvmwrapreduce(Val{:min}, v)
@inline Base.prod{N,T}(v::Vec{N,T}) = llvmwrapreduce(Val{:*}, v)
@inline Base.sum{N,T}(v::Vec{N,T}) = llvmwrapreduce(Val{:+}, v)

@generated function Base.reduce{Op,N,T}(::Type{Val{Op}}, v::Vec{N,T})
    @assert isa(Op, Symbol)
    z = getneutral(Op, T)
    stmts = []
    n = N
    push!(stmts, :($(symbol(:v,n)) = v))
    nold,n = n,nextpow2(n)
    if n > nold
        push!(stmts,
            :($(symbol(:v,n)) = Vec{$n,T}($(Expr(:tuple,
                [:($(symbol(:v,nold)).elts[$i]) for i in 1:nold]...,
                [z for i in nold+1:n]...)))))
    end
    while n > 1
        nold,n = n, div(n, 2)
        push!(stmts,
            :($(symbol(:v,n,"lo")) = Vec{$n,T}($(Expr(:tuple,
                [:($(symbol(:v,nold)).elts[$i]) for i in 1:n]...)))))
        push!(stmts,
            :($(symbol(:v,n,"hi")) = Vec{$n,T}($(Expr(:tuple,
                [:($(symbol(:v,nold)).elts[$i]) for i in n+1:nold]...)))))
        push!(stmts,
            :($(symbol(:v,n)) =
                $Op($(symbol(:v,n,"lo")), $(symbol(:v,n,"hi")))))
    end
    push!(stmts, :(v1[1]))
    Expr(:block, Expr(:meta, :inline), stmts...)
end

@inline Base.maximum{N,T<:IntegerTypes}(v::Vec{N,T}) = reduce(Val{:max}, v)
@inline Base.minimum{N,T<:IntegerTypes}(v::Vec{N,T}) = reduce(Val{:min}, v)

# Load and store functions

export vload, vloada
@generated function vload{N,T,Aligned}(::Type{Vec{N,T}}, ptr::Ptr{T},
        ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = ""
    if align > 0 flags *= ", align $align" end
    push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    push!(instrs, "%res = load $vtyp, $vtyp* %ptr$flags")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{Ptr{T}}, ptr))
    end
end

@inline vloada{N,T}(::Type{Vec{N,T}}, ptr::Ptr{T}) =
    vload(Vec{N,T}, ptr, Val{true})

@inline function vload{N,T,Aligned}(::Type{Vec{N,T}}, arr::Vector{T},
        i::Integer, ::Type{Val{Aligned}} = Val{false})
    # @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), Val{Aligned})
end
@inline vloada{N,T}(::Type{Vec{N,T}}, arr::Vector{T}, i::Integer) =
    vload(Vec{N,T}, arr, i, Val{true})

@generated function vload{N,T,Aligned}(::Type{Vec{N,T}}, ptr::Ptr{T},
        mask::Vec{N,Bool}, ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    abtyp = "[$N x $btyp]"
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    append!(instrs, array2vector("%maskb", N, btyp, "%1", "%arg2arr"))
    push!(instrs, "%mask = trunc $vbtyp %maskb to <$N x i1>")
    push!(decls,
        "declare $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp*, i32, " *
            "<$N x i1>, $vtyp)")
    push!(instrs,
        "%res = call $vtyp @llvm.masked.load.$(suffix(N,T))($vtyp* %ptr, " *
            "i32 $align, <$N x i1> %mask, $vtyp $(llvmconst(N, T, 0)))")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{Ptr{T}, NTuple{N,Bool}}, ptr, mask.elts))
    end
end

@inline vloada{N,T}(::Type{Vec{N,T}}, ptr::Ptr{T}, mask::Vec{N,Bool}) =
    vload(Vec{N,T}, ptr, mask, Val{true})

@inline function vload{N,T,Aligned}(::Type{Vec{N,T}}, arr::Vector{T},
        i::Integer, mask::Vec{N,Bool}, ::Type{Val{Aligned}} = Val{false})
    # @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vload(Vec{N,T}, pointer(arr, i), mask, Val{Aligned})
end
@inline vloada{N,T}(::Type{Vec{N,T}}, arr::Vector{T}, i::Integer,
        mask::Vec{N,Bool}) =
    vload(Vec{N,T}, arr, i, mask, Val{true})

export vstore, vstorea
@generated function vstore{N,T,Aligned}(v::Vec{N,T}, ptr::Ptr{T},
        ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = ""
    if align > 0 flags *= ", align $align" end
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    push!(instrs, "store $vtyp %arg1, $vtyp* %ptr$flags")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Void, Tuple{NTuple{N,T}, Ptr{T}}, v.elts, ptr)
    end
end

@inline vstorea{N,T}(v::Vec{N,T}, ptr::Ptr{T}) = vstore(v, ptr, Val{true})

@inline function vstore{N,T,Aligned}(v::Vec{N,T}, arr::Vector{T}, i::Integer,
        ::Type{Val{Aligned}} = Val{false})
    @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), Val{Aligned})
end
@inline vstorea{N,T}(v::Vec{N,T}, arr::Vector{T}, i::Integer) =
    vstore(v, arr, i, Val{true})

@generated function vstore{N,T,Aligned}(v::Vec{N,T}, ptr::Ptr{T},
        mask::Vec{N,Bool}, ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    btyp = llvmtype(Bool)
    abtyp = "[$N x $btyp]"
    vbtyp = "<$N x $btyp>"
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    append!(instrs, array2vector("%maskb", N, btyp, "%2", "%arg3arr"))
    push!(instrs, "%mask = trunc $vbtyp %maskb to <$N x i1>")
    push!(decls,
        "declare void @llvm.masked.store.$(suffix(N,T))($vtyp, $vtyp*, i32, " *
            "<$N x i1>)")
    push!(instrs,
        "call void @llvm.masked.store.$(suffix(N,T))($vtyp %arg1, " *
            "$vtyp* %ptr, i32 $align, <$N x i1> %mask)")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Void, Tuple{NTuple{N,T}, Ptr{T}, NTuple{N,Bool}}, v.elts, ptr,
                mask.elts)
    end
end

@inline vstorea{N,T}(v::Vec{N,T}, ptr::Ptr{T}, mask::Vec{N,Bool}) =
    vstore(v, ptr, mask, Val{true})

@inline function vstore{N,T,Aligned}(v::Vec{N,T}, arr::Vector{T}, i::Integer,
        mask::Vec{N,Bool}, ::Type{Val{Aligned}} = Val{false})
    # @boundscheck 1 <= i <= length(arr) - (N-1) || throw(BoundsError())
    vstore(v, pointer(arr, i), mask, Val{Aligned})
end
@inline function vstorea{N,T}(v::Vec{N,T}, arr::Vector{T}, i::Integer,
        mask::Vec{N,Bool})
    vstore(v, arr, i, mask, Val{true})
end

end
