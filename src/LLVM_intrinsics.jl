# LLVM operations and intrinsics
module Intrinsics

# Note, that in the functions below, some care needs to be taken when passing
# Julia Bools to LLVM. Julia passes Bools as LLVM i8 but expect them to only
# have the last bit as non-zero. Failure to comply with this can give weird errors
# like false !== false where the first false is the result of some computation.

# Note, no difference is made between Julia usigned integers and signed integers
# when passed to LLVM. It is up to the caller to make sure that the correct
# intrinsic is called (e.g uitofp vs sitofp).

# TODO: fastmath flags

import ..SIMD: SIMD, VE, LVec, FloatingTypes
# Inlcude Bool in IntegerTypes
const IntegerTypes = Union{SIMD.IntegerTypes, Bool}

const d = Dict{DataType, String}(
    Bool         => "i8",
    Int8         => "i8",
    Int16        => "i16",
    Int32        => "i32",
    Int64        => "i64",
    Int128       => "i128",

    UInt8        => "i8",
    UInt16       => "i16",
    UInt32       => "i32",
    UInt64       => "i64",
    UInt128      => "i128",

    #Float16     => "half",
    Float32      => "float",
    Float64      => "double",
)
# Add the Ptr translations
foreach(x -> (d[Ptr{x}] = d[Int]), collect(keys(d)))

# LT = LLVM Type (scalar and vectors), we keep type names intentionally short
# to make the signatures smaller
const LT{T} = Union{LVec{<:Any, T}, T}

suffix(N::Integer, ::Type{Ptr{T}}) where {T} = "v$(N)p0$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
suffix(N::Integer, ::Type{T}) where {T}      = "v$(N)$(T<:IntegerTypes   ? "i" : "f")$(8*sizeof(T))"
suffix(::Type{T}) where {T}                  = "$(T<:IntegerTypes        ? "i" : "f")$(8*sizeof(T))"

llvm_name(llvmf, N, T)                           = string("llvm", ".", llvmf, ".", suffix(N, T))
llvm_name(llvmf, ::Type{LVec{N, T}}) where {N,T} = string("llvm", ".", llvmf, ".", suffix(N, T))
llvm_name(llvmf, ::Type{T}) where {T}            = string("llvm", ".", llvmf, ".", suffix(T))

llvm_type(::Type{T}) where {T}            = d[T]
llvm_type(::Type{LVec{N, T}}) where {N,T} = "< $N x $(d[T])>"


####################
# Unary operators  #
####################

const UNARY_INTRINSICS_FLOAT = [
    :sqrt
    :sin
    :cos
    :exp
    :trunc
    :exp2
    :log
    :log10
    :log2
    :fabs
    :floor
    :ceil
    :rint
    :nearbyint
    :round
]

const UNARY_INTRINSICS_INT = [
    :bitreverse
    :bswap
    :ctpop
    :ctlz
    :cttz
    :fshl
    :fshr
]
for (fs, c) in zip([UNARY_INTRINSICS_FLOAT, UNARY_INTRINSICS_INT],
                   [FloatingTypes,          IntegerTypes])
    for f in fs
        @eval begin
            @generated function $(f)(x::T) where T<:LT{<:$c}
                ff = llvm_name($(QuoteNode(f)), T)
                return :(
                    $(Expr(:meta, :inline));
                    ccall($ff, llvmcall, T, (T,), x)
                )
            end
        end
    end
end

# fneg (not an intrinsic so cannot use `ccall)
@generated function fneg(x::T) where T<:LT{<:FloatingTypes}
    s = """
    %2 = fneg $(llvm_type(T)) %0
    ret $(llvm_type(T)) %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T, Tuple{T}, x)
    )
end

#####################
# Binary operators  #
#####################

const BINARY_OPS_FLOAT = [
    :fadd
    :fsub
    :fmul
    :fdiv
    :frem
]

const BINARY_OPS_INT = [
    :add
    :sub
    :mul
    :sdiv
    :udiv
    :srem
    :urem
    :shl
    :ashr
    :lshr
    :and
    :or
    :xor
]

for (fs, c) in zip([BINARY_OPS_FLOAT, BINARY_OPS_INT],
                   [FloatingTypes, IntegerTypes])
    for f in fs
        @eval @generated function $f(x::T, y::T) where T<:LT{<:$c}
            ff = $(QuoteNode(f))
            s = """
            %3 = $ff $(llvm_type(T)) %0, %1
            ret $(llvm_type(T)) %3
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, T, Tuple{T, T}, x, y)
            )
        end
    end
end

const BINARY_INTRINSICS_FLOAT = [
    :minnum
    :maxnum
    :minimum
    :maximum
    :copysign
    :pow
    :floor
    :ceil
    :trunc
    :rint
    :nearbyint
    :round
]

for f in BINARY_INTRINSICS_FLOAT
    @eval @generated function $(f)(x::T, y::T) where T<:LT{<:FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), T,)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, T, (T, T), x, y)
        )
    end
end

# pow, powi
for (f, c) in [(:pow, FloatingTypes), (:powi, IntegerTypes)]
    @eval @generated function $(f)(x::T, y::T2) where {T <: LT{<:FloatingTypes}, T2 <: $c}
        ff = llvm_name($(QuoteNode(f)), T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, T, (T, T2), x, y)
        )
    end
end

# Comparisons
const CMP_FLAGS_FLOAT = [
    :false
    :oeq
    :ogt
    :oge
    :olt
    :ole
    :one
    :ord
    :ueq
    :ugt
    :uge
    :ult
    :ule
    :une
    :uno
    :true
]

const CMP_FLAGS_INT = [
    :eq
    :ne
    :sgt
    :sge
    :slt
    :sle
    :ugt
    :uge
    :ult
    :ule
]

for (f, c, flags) in zip(["fcmp",          "icmp"],
                         [FloatingTypes,   IntegerTypes],
                         [CMP_FLAGS_FLOAT, CMP_FLAGS_INT])
    for flag in flags
        ftot = Symbol(string(f, "_", flag))
        @eval @generated function $ftot(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: $c}
            fflag = $(QuoteNode(flag))
            ff = $(QuoteNode(f))
            s = """
            %res = $ff $(fflag) <$(N) x $(d[T])> %0, %1
            %resb = zext <$(N) x i1> %res to <$(N) x i8>
            ret <$(N) x i8> %resb
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, Bool}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
            )
        end
    end
end


#####################
# Ternary operators #
#####################

@generated function select(cond::LVec{N, Bool}, x::LVec{N, T}, y::LVec{N, T}) where {N, T}
    s = """
    %cond = trunc <$(N) x i8> %0 to <$(N) x i1>
    %res = select <$N x i1> %cond, <$N x $(d[T])> %1, <$N x $(d[T])> %2
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, Bool}, LVec{N, T}, LVec{N, T}}, cond, x, y)
    )
end

const MULADD_INTRINSICS = [
    :fmuladd,
    :fma,
]

for f in MULADD_INTRINSICS
    @eval @generated function $(f)(a::LVec{N, T}, b::LVec{N, T}, c::LVec{N, T}) where {N, T<:FloatingTypes}
        ff = llvm_name($(QuoteNode(f)), N, T)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, LVec{N, T}, (LVec{N, T}, LVec{N, T}, LVec{N, T}), a, b, c)
        )
    end
end


################
# Load / store #
################

# These alignment numbers feels a bit dubious
n_align(align, N, T) = align ? N * sizeof(T) : sizeof(T)
temporal_str(temporal) = temporal ? ", !nontemporal !{i32 1}" : ""

@generated function load(x::Type{LVec{N, T}}, ptr::Ptr{T},
                         ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    s = """
    %ptr = inttoptr $(d[Int]) %0 to <$N x $(d[T])>*
    %res = load <$N x $(d[T])>, <$N x $(d[T])>* %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{Ptr{T}}, ptr)
    )
end

@generated function maskedload(ptr::Ptr{T}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    # TODO: Allow setting the passthru
    decl = "declare <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))(<$N x $(d[T])>*, i32, <$N x i1>, <$N x $(d[T])>)"
    s = """
    %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
    %ptr = inttoptr $(d[Int]) %0 to <$N x $(d[T])>*
    %res = call <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))(<$N x $(d[T])>* %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), LVec{N, T}, Tuple{Ptr{T}, LVec{N,Bool}}, ptr, mask)
    )
end

@generated function store(x::LVec{N, T}, ptr::Ptr{T},
                          ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    s = """
    %ptr = inttoptr $(d[Int]) %1 to <$N x $(d[T])>*
    store <$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret void
    """
    return :(

        $(Expr(:meta, :inline));
        Base.llvmcall($s, Cvoid, Tuple{LVec{N, T}, Ptr{T}}, x, ptr)
    )
end

@generated function maskedstore(x::LVec{N, T}, ptr::Ptr{T}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    # TODO: Allow setting the passthru
    decl = "declare <$N x $(d[T])> @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(d[T])>*, i32, <$N x i1>)"
    s = """
    %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
    %ptr = inttoptr $(d[Int]) %1 to <$N x $(d[T])>*
    %res = call <$N x $(d[T])> @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
    ret void
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), Cvoid, Tuple{LVec{N, T}, Ptr{T}, LVec{N,Bool}}, x, ptr, mask)
    )
end


####################
# Gather / Scatter #
####################

@generated function maskedgather(ptrs::LVec{N,Ptr{T}},
                                 mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, Al}
    # TODO: Allow setting the passthru
    decl = "declare <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*>, i32, <$N x i1>, <$N x $(d[T])>)"
    s = """
    %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
    %ptrs = inttoptr <$N x $(d[Int])> %0 to <$N x $(d[T])*>
    %res = call <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), LVec{N, T}, Tuple{LVec{N, Ptr{T}}, LVec{N, Bool}}, ptrs, mask)
    )
end

@generated function maskedscatter(x::LVec{N, T}, ptrs::LVec{N, Ptr{T}},
                                  mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, Al}

    decl = "declare <$N x $(d[T])> @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(d[T])*>, i32, <$N x i1>)"
    s = """
    %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
    %ptrs = inttoptr <$N x $(d[Int])> %1 to <$N x $(d[T])*>
    call <$N x $(d[T])> @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(d[T])*> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
    ret void
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($decl, $s), Cvoid, Tuple{LVec{N, T}, LVec{N, Ptr{T}}, LVec{N, Bool}}, x, ptrs, mask)
    )
end


######################
# LVector Operations #
######################

@generated function extractelement(x::LVec{N, T}, i::I) where {N, T, I <: IntegerTypes}
    s = """
    %3 = extractelement <$N x $(d[T])> %0, $(d[I]) %1
    ret $(d[T]) %3
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T, Tuple{LVec{N, T}, $i}, x, i)
    )
end

@generated function insertelement(x::LVec{N, T}, v::T, i::IntegerTypes) where {N, T}
    s = """
    %4 = insertelement <$N x $(d[T])> %0, $(d[T]) %1, $(d[i]) %2
    ret <$N x $(d[T])> %4
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LVec{N, T}, T, typeof(i)}, x, v, i)
    )
end

_shuffle_vec(I) = join((string("i32 ", i == :undef ? "undef" : Int32(i::Integer)) for i in I), ", ")
@generated function shufflevector(x::LVec{N, T}, y::LVec{N, T}, ::Val{I}) where {N, T, I}
    shfl = _shuffle_vec(I)
    M = length(I)
    s = """
    %res = shufflevector <$N x $(d[T])> %0, <$N x $(d[T])> %1, <$M x i32> <$shfl>
    ret <$M x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{$M, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
    )
end

@generated function shufflevector(x::LVec{N, T}, ::Val{I}) where {N, T, I}
    shfl = _shuffle_vec(I)
    M = length(I)
    s = """
    %res = shufflevector <$(N) x $(d[T])> %0, <$N x $(d[T])> undef, <$M x i32> <$shfl>
    ret <$M x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{$M, T}, Tuple{LVec{N, T}}, x)
    )
end

@generated function constantvector(v::T, y::Type{LVec{N, T}}) where {N, T}
    s = """
    %2 = insertelement <$N x $(d[T])> undef, $(d[T]) %0, i32 0
    %res = shufflevector <$N x $(d[T])> %2, <$N x $(d[T])> undef, <$N x i32> zeroinitializer
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{T}, v)
    )
end

#########################
# Conversion Operations #
#########################

const CAST_SIZE_CHANGE_FLOAT = [
    (:fptrunc, >)
    (:fpext, <)
]

const CAST_SIZE_CHANGE_INT = [
    (:trunc, >)
    (:zext, <)
    (:sext, <)
]

for (fs, c) in zip([CAST_SIZE_CHANGE_FLOAT, CAST_SIZE_CHANGE_INT],
                   [FloatingTypes,          IntegerTypes])
    for (f, criteria) in fs
        @eval @generated function $f(::Type{LVec{N, T2}}, x::LVec{N, T1}) where {N, T1 <: $c, T2 <: $c}
            sT1, sT2 = sizeof(T1) * 8, sizeof(T2) * 8
            # Not changing size is not allowed
            @assert $criteria(sT1, sT2) "size of conversion type ($T2: $sT2) must be $($criteria) than the element type ($T1: $sT1)"
            ff = $(QuoteNode(f))
            s = """
            %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
            ret <$(N) x $(d[T2])> %2
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, T1}}, x)
            )
        end
    end
end

const CONVERSION_FLOAT_TO_INT = [
    :fptoui,
    :fptosi
]

const CONVERSION_INT_TO_FLOAT = [
    :uitofp,
    :sitofp
]

for (fs, (from, to)) in zip([CONVERSION_FLOAT_TO_INT,       CONVERSION_INT_TO_FLOAT],
                           [(FloatingTypes, IntegerTypes), (IntegerTypes, FloatingTypes)])
    for f in fs
        @eval @generated function $f(::Type{LVec{N, T2}}, x::LVec{N, T1}) where {N, T1 <: $from, T2 <: $to}
            ff = $(QuoteNode(f))
            s = """
            %2 = $ff <$(N) x $(d[T1])> %0 to <$(N) x $(d[T2])>
            ret <$(N) x $(d[T2])> %2
            """
            return :(
                $(Expr(:meta, :inline));
                Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, T1}}, x)
            )
        end
    end
end


###########
# Bitcast #
###########

@generated function bitcast(::Type{T1}, x::T2) where {T1<:LT, T2<:LT}
    sT1, sT2 = sizeof(T1), sizeof(T2)
    @assert sT1 == sT2 "size of conversion type ($T1: $sT1) must be equal to the vector type ($T2: $sT2)"
    s = """
    %2 = bitcast $(llvm_type(T2)) %0 to $(llvm_type(T1))
    ret $(llvm_type(T1)) %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T1, Tuple{T2}, x)
    )
end

##################################
# Horizontal reductions (LLVM 9) #
##################################

const HORZ_REDUCTION_OPS_FLOAT = [
    :fmax
    :fmin
]

const HORZ_REDUCTION_OPS_INT = [
    :and
    :or
    :mul
    :add
    :smax
    :umax
    :smin
    :umin
]

for (fs, c) in zip([HORZ_REDUCTION_OPS_FLOAT, HORZ_REDUCTION_OPS_INT],
                   [FloatingTypes,            IntegerTypes])
    for f in fs
        f_red = Symbol("reduce_", f)
        @eval @generated function $f_red(x::LVec{N, T}) where {N,T<:$c}
            ff = llvm_name(string("experimental.vector.reduce.", $(QuoteNode(f))), N, T)
            decl = "declare $(d[T]) @$ff(<$N x $(d[T])>)"
            s2 = """
            %res = call $(d[T]) @$ff(<$N x $(d[T])> %0)
            ret $(d[T]) %res
            """
            return quote
                $(Expr(:meta, :inline));
                Base.llvmcall($(decl, s2), T, Tuple{LVec{N, T},}, x)
            end
        end
    end
end

# The fadd and fmul reductions take an initializer
horz_reduction_version = Base.libllvm_version < v"9" ? "" : "v2."
for (f, neutral) in [(:fadd, "0.0"), (:fmul, "1.0")]
    f_red = Symbol("reduce_", f)
    @eval @generated function $f_red(x::LVec{N, T}) where {N,T<:FloatingTypes}
        ff = llvm_name(string("experimental.vector.reduce.$horz_reduction_version", $(QuoteNode(f))), N, T)
        decl = "declare $(d[T]) @$ff($(d[T]), <$N x $(d[T])>)"
        s2 = """
        %res = call $(d[T]) @$ff($(d[T]) $($neutral), <$N x $(d[T])> %0)
        ret $(d[T]) %res
        """
        return quote
            $(Expr(:meta, :inline));
            Base.llvmcall($(decl, s2), T, Tuple{LVec{N, T},}, x)
        end
    end
end

end
