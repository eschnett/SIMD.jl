# LLVM operations and intrinsics
module Intrinsics

# Note, that in the functions below, some care needs to be taken when passing
# Julia Bools to LLVM. Julia passes Bools as LLVM i8 but expect them to only
# have the last bit as non-zero. Failure to comply with this can give weird errors
# like false !== false where the first false is the result of some computation.

# Note, no difference is made between Julia usigned integers and signed integers
# when passed to LLVM. It is up to the caller to make sure that the correct
# intrinsic is called (e.g uitofp vs sitofp).

import ..SIMD: SIMD, VE, LVec, FloatingTypes
using Core: LLVMPtr
# Include Bool in IntegerTypes
const IntegerTypes = Union{SIMD.IntegerTypes, Bool}

const d = Dict{Type, String}(
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

    Float16      => "half",
    Float32      => "float",
    Float64      => "double",
)
# Add the Ptr translations
# Julia <=1.11 (LLVM <=16) passes `Ptr{T}` as `i64`, Julia >=1.12 (LLVM >=17) passes them as `ptr`.
# Use `argtoptr` e.g. as `%ptr = $argtoptr $(d[Ptr]) %0 to <$N x $(d[T])>*`
@static if VERSION >= v"1.12-DEV"
    const argtoptr = "bitcast"
    d[Ptr] = "ptr"
else
    const argtoptr = "inttoptr"
    d[Ptr] = d[Int]
end

# LT = LLVM Type (scalar and vectors), we keep type names intentionally short
# to make the signatures smaller
const LT{T} = Union{LVec{<:Any, T}, T}

suffix(N::Integer, ::Type{Ptr{T}}) where {T} = "v$(N)p0$(T<:IntegerTypes ? "i" : "f")$(8*sizeof(T))"
suffix(N::Integer, ::Type{T}) where {T}      = "v$(N)$(T<:IntegerTypes   ? "i" : "f")$(8*sizeof(T))"
suffix(::Type{T}) where {T}                  = "$(T<:IntegerTypes        ? "i" : "f")$(8*sizeof(T))"

dotit(f) = replace(string(f), "_" => ".")
llvm_name(llvmf, N, T)                           = string("llvm", ".", dotit(llvmf), ".", suffix(N, T))
llvm_name(llvmf, ::Type{LVec{N, T}}) where {N,T} = string("llvm", ".", dotit(llvmf), ".", suffix(N, T))
llvm_name(llvmf, ::Type{T}) where {T}            = string("llvm", ".", dotit(llvmf), ".", suffix(T))

llvm_type(::Type{T}) where {T}            = d[T]
llvm_type(::Type{LVec{N, T}}) where {N,T} = "<$N x $(d[T])>"

@static if VERSION >= v"1.12-DEV"
    llvm_ptr(T, AS) = "ptr addrspace($AS)"
else
    llvm_ptr(T, AS) = "$(llvm_type(T)) addrspace($AS)*"
end

############
# FastMath #
############

module FastMath
    const nnan     = 1 << 0
    const ninf     = 1 << 1
    const nsz      = 1 << 2
    const arcp     = 1 << 3
    const contract = 1 << 4
    const afn      = 1 << 5
    const reassoc  = 1 << 6
    const fast     = 1 << 7
end

struct FastMathFlags{T} end
Base.@pure FastMathFlags(T::Int) = FastMathFlags{T}()

function fp_str(::Type{FastMathFlags{T}}) where {T}
    flags = String[]
    (T & FastMath.nnan     != 0) && push!(flags, "nnan")
    (T & FastMath.ninf     != 0) && push!(flags, "ninf")
    (T & FastMath.nsz      != 0) && push!(flags, "nsz")
    (T & FastMath.arcp     != 0) && push!(flags, "arcp")
    (T & FastMath.contract != 0) && push!(flags, "contract")
    (T & FastMath.afn      != 0) && push!(flags, "afn")
    (T & FastMath.reassoc  != 0) && push!(flags, "reassoc")
    (T & FastMath.fast     != 0) && push!(flags, "fast")
    return join(flags, " ")
end
fp_str(::Type{Nothing}) = ""

const FPFlags{T} = Union{Nothing, FastMathFlags{T}}

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

const SHIFT_INTRINSICS_INT = [
    :fshl
    :fshr
]

for f in SHIFT_INTRINSICS_INT
    @eval begin
        @generated function $(f)(a::T, b::T, c::T) where T<:LT{<:IntegerTypes}
            ff = llvm_name($(QuoteNode(f)), T)
            return :(
                $(Expr(:meta, :inline));
                ccall($ff, llvmcall, T, (T,T,T), a, b, c)
            )
        end
    end
end

# ctlz/cttz: additional i1 flag argument, is_zero_undef
for f in [:ctlz, :cttz]
    @eval @generated function $(f)(x::T) where {T<:LT{<:IntegerTypes}}
        ff = llvm_name($(QuoteNode(f)), T,)
        typ = llvm_type(T)
        mod = """
            declare $typ @$(ff)($typ, i1)

            define $typ @entry($typ) #0 {
            top:
                %res = call $typ @$(ff)($typ %0, i1 0)
                ret $typ %res
            }

            attributes #0 = { alwaysinline }
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($(mod, "entry"), T, Tuple{T}, x)
        )
    end
end

# fneg (not an intrinsic so cannot use `ccall)
@generated function fneg(x::T, ::F=nothing) where {T<:LT{<:FloatingTypes}, F<:FPFlags}
    fpflags = fp_str(F)
    s = """
    %2 = fneg $fpflags $(llvm_type(T)) %0
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

for f in BINARY_OPS_FLOAT
    @eval @generated function $f(x::T, y::T, ::F=nothing) where {T<:LT{<:FloatingTypes}, F<:FPFlags}
        fpflags = fp_str(F)
        ff = $(QuoteNode(f))
        s = """
        %3 = $ff $fpflags $(llvm_type(T)) %0, %1
        ret $(llvm_type(T)) %3
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($s, T, Tuple{T, T}, x, y)
        )
    end
end

for f in BINARY_OPS_INT
    @eval @generated function $f(x::T, y::T, ::F=nothing) where {T<:LT{<:IntegerTypes}, F<:FPFlags}
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

const BINARY_INTRINSICS_INT = [
    :sadd_sat
    :uadd_sat
    :ssub_sat
    :usub_sat
]

for f in BINARY_INTRINSICS_FLOAT
    @eval @generated function $(f)(x::LVec{N, T}, y::LVec{N, T}, ::F=nothing) where {N, T<:FloatingTypes, F<:FPFlags}
        XT = llvm_type(LVec{N, T})
        ff = $f
        fpflags = fp_str(F)
        mod = """
            declare $XT @llvm.$ff.$(suffix(N, T))($XT, $XT)
            define $XT @entry($XT , $XT) #0 {
            top:
                %res = call $fpflags $XT @llvm.$ff.$(suffix(N, T))($XT %0, $XT %1)
                ret $XT %res
            }

            attributes #0 = { alwaysinline }
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
    end
end

for f in BINARY_INTRINSICS_INT
    @eval @generated function $(f)(x::T, y::T) where T<:LT{<:IntegerTypes}
        ff = llvm_name($(QuoteNode(f)), T,)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, T, (T, T), x, y)
        )
    end
end


# pow, powi
for (f, c) in [(:pow, FloatingTypes), (:powi, Union{Int32,UInt32})]
    @eval @generated function $(f)(x::T, y::T2) where {T <: LT{<:FloatingTypes}, T2 <: $c}
        ff = llvm_name($(QuoteNode(f)), T)  * "." * suffix(T2)
        return :(
            $(Expr(:meta, :inline));
            ccall($ff, llvmcall, T, (T, T2), x, y)
        )
    end
end

# Overflow
const OVERFLOW_INTRINSICS = [
    :sadd_with_overflow
    :uadd_with_overflow
    :ssub_with_overflow
    :usub_with_overflow
    :smul_with_overflow
    :umul_with_overflow
]

const SUPPORTS_VEC_OVERFLOW = Base.libllvm_version >= v"9"
for f in OVERFLOW_INTRINSICS
    @eval @generated function $f(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: IntegerTypes}
        if !SUPPORTS_VEC_OVERFLOW
            return :(error("LLVM version 9.0 or greater required (Julia 1.5 or greater)"))
        end
        ff = llvm_name($(QuoteNode(f)), N, T)
        if $(QuoteNode(f)) == :smul_with_overflow && Sys.ARCH == :i686 && T == Int64
            str = "this intrinsic ($ff) is broken on i686"
            return :(error($str))
        end

        # Julia passes Tuple{[U]Int8, Bool} as [2 x i8] so we need to special case that scenario
        ret_type = sizeof(T) == 1 ? "[2 x <$N x i8>]" : "{<$N x $(d[T])>, <$N x i8>}"

        mod = """
            declare {<$N x $(d[T])>, <$N x i1>} @$ff(<$N x $(d[T])>, <$N x $(d[T])>)

            define $ret_type @entry(<$N x $(d[T])>, <$N x $(d[T])>) #0 {
            top:
                %res = call {<$N x $(d[T])>, <$N x i1>} @$ff(<$N x $(d[T])> %0, <$N x $(d[T])> %1)
                %plus     = extractvalue {<$N x $(d[T])>, <$N x i1>} %res, 0
                %overflow = extractvalue {<$N x $(d[T])>, <$N x i1>} %res, 1
                %overflow_ext = zext <$(N) x i1> %overflow to <$(N) x i8>
                %new_tuple   = insertvalue $ret_type undef,      <$N x $(d[T])> %plus,         0
                %new_tuple_2 = insertvalue $ret_type %new_tuple, <$N x i8>      %overflow_ext, 1
                ret $ret_type %new_tuple_2
            }

            attributes #0 = { alwaysinline }
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall(($mod, "entry"), Tuple{LVec{N, T}, LVec{N, Bool}}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
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

for flag in CMP_FLAGS_FLOAT
    ftot = Symbol(string("fcmp_", flag))
    @eval @generated function $ftot(x::LVec{N, T}, y::LVec{N, T}, ::F=nothing) where {N, T <: FloatingTypes, F<:FPFlags}
        fpflags = fp_str(F)
        fflag = $(QuoteNode(flag))
        s = """
        %res = fcmp $(fpflags) $(fflag) <$(N) x $(d[T])> %0, %1
        %resb = zext <$(N) x i1> %res to <$(N) x i8>
        ret <$(N) x i8> %resb
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($s, LVec{N, Bool}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
    end
end

for flag in CMP_FLAGS_INT
    ftot = Symbol(string("icmp_", flag))
    @eval @generated function $ftot(x::LVec{N, T}, y::LVec{N, T}) where {N, T <: IntegerTypes}
        fflag = $(QuoteNode(flag))
        s = """
        %res = icmp $(fflag) <$(N) x $(d[T])> %0, %1
        %resb = zext <$(N) x i1> %res to <$(N) x i8>
        ret <$(N) x i8> %resb
        """
        return :(
            $(Expr(:meta, :inline));
            Base.llvmcall($s, LVec{N, Bool}, Tuple{LVec{N, T}, LVec{N, T}}, x, y)
        )
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
    %ptr = $argtoptr $(d[Ptr]) %0 to <$N x $(d[T])>*
    %res = load <$N x $(d[T])>, <$N x $(d[T])>* %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{Ptr{T}}, ptr)
    )
end

@generated function load(x::Type{LVec{N, T}}, ptr::LLVMPtr{T, AS},
                         ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, AS, Al, Te}
    s = """
    %ptr = bitcast $(llvm_ptr(UInt8, AS)) %0 to $(llvm_ptr(LVec{N, T}, AS))
    %res = load <$N x $(d[T])>, $(llvm_ptr(LVec{N, T}, AS)) %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret <$N x $(d[T])> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T}, Tuple{LLVMPtr{T, AS}}, ptr)
    )
end

@generated function maskedload(ptr::Ptr{T}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))(<$N x $(d[T])>*, i32, <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry($(d[Ptr]), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptr = $argtoptr $(d[Ptr]) %0 to <$N x $(d[T])>*
            %res = call <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))(<$N x $(d[T])>* %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{Ptr{T}, LVec{N,Bool}}, ptr, mask)
    )
end

@generated function maskedload(ptr::LLVMPtr{T, AS}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, AS, Al, Te}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))($(llvm_ptr(LVec{N, T}, AS)), i32, <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry($(llvm_ptr(UInt8, AS)), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptr = bitcast $(llvm_ptr(UInt8, AS)) %0 to $(llvm_ptr(LVec{N, T}, AS))
            %res = call <$N x $(d[T])> @llvm.masked.load.$(suffix(N, T))($(llvm_ptr(LVec{N, T}, AS)) %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{LLVMPtr{T, AS}, LVec{N,Bool}}, ptr, mask)
    )
end

@generated function store(x::LVec{N, T}, ptr::Ptr{T},
                          ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    s = """
    %ptr = $argtoptr $(d[Ptr]) %1 to <$N x $(d[T])>*
    store <$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret void
    """
    return :(

        $(Expr(:meta, :inline));
        Base.llvmcall($s, Cvoid, Tuple{LVec{N, T}, Ptr{T}}, x, ptr)
    )
end

@generated function store(x::LVec{N, T}, ptr::LLVMPtr{T, AS},
                          ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, AS, Al, Te}
    s = """
    %ptr = bitcast $(llvm_ptr(UInt8, AS)) %1 to $(llvm_ptr(LVec{N, T}, AS))
    store <$N x $(d[T])> %0, $(llvm_ptr(LVec{N, T}, AS)) %ptr, align $(n_align(Al, N, T)) $(temporal_str(Te))
    ret void
    """
    return :(

        $(Expr(:meta, :inline));
        Base.llvmcall($s, Cvoid, Tuple{LVec{N, T}, LLVMPtr{T, AS}}, x, ptr)
    )
end

@generated function maskedstore(x::LVec{N, T}, ptr::Ptr{T}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, Al, Te}
    # TODO: Allow setting the passthru
    mod = """
        declare void @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(d[T])>*, i32, <$N x i1>)

        define void @entry(<$N x $(d[T])>, $(d[Ptr]), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptr = $argtoptr $(d[Ptr]) %1 to <$N x $(d[T])>*
            call void @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(d[T])>* %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, Ptr{T}, LVec{N,Bool}}, x, ptr, mask)
    )
end

@generated function maskedstore(x::LVec{N, T}, ptr::LLVMPtr{T, AS}, mask::LVec{N,Bool},
                               ::Val{Al}=Val(false), ::Val{Te}=Val(false)) where {N, T, AS, Al, Te}
    # TODO: Allow setting the passthru
    mod = """
        declare void @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])>, $(llvm_ptr(LVec{N, T}, AS)), i32, <$N x i1>)

        define void @entry(<$N x $(d[T])>, $(llvm_ptr(UInt8, AS)), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptr = bitcast $(llvm_ptr(UInt8, AS)) %1 to $(llvm_ptr(LVec{N, T}, AS))
            call void @llvm.masked.store.$(suffix(N, T))(<$N x $(d[T])> %0, $(llvm_ptr(LVec{N, T}, AS)) %ptr, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, LLVMPtr{T, AS}, LVec{N,Bool}}, x, ptr, mask)
    )
end


@generated function maskedexpandload(ptr::Ptr{T}, mask::LVec{N,Bool}) where {N, T}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.expandload.$(suffix(N, T))($(d[T])*, <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry($(d[Ptr]), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptr = $argtoptr $(d[Ptr]) %0 to $(d[T])*
            %res = call <$N x $(d[T])> @llvm.masked.expandload.$(suffix(N, T))($(d[T])* %ptr, <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{Ptr{T}, LVec{N, Bool}}, ptr, mask)
    )
end

@generated function maskedexpandload(ptr::LLVMPtr{T, AS}, mask::LVec{N,Bool}) where {N, T, AS}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.expandload.$(suffix(N, T))($(llvm_ptr(T, AS)), <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry($(llvm_ptr(UInt8, AS)), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptr = bitcast $(llvm_ptr(UInt8, AS)) %0 to $(llvm_ptr(T, AS))
            %res = call <$N x $(d[T])> @llvm.masked.expandload.$(suffix(N, T))($(llvm_ptr(T, AS)) %ptr, <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{LLVMPtr{T, AS}, LVec{N, Bool}}, ptr, mask)
    )
end


@generated function maskedcompressstore(x::LVec{N, T}, ptr::Ptr{T},
                                        mask::LVec{N,Bool}) where {N, T}
    mod = """
        declare void @llvm.masked.compressstore.$(suffix(N, T))(<$N x $(d[T])>, $(d[T])*, <$N x i1>)

        define void @entry(<$N x $(d[T])>, $(d[Ptr]), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptr = $argtoptr $(d[Ptr]) %1 to $(d[T])*
            call void @llvm.masked.compressstore.$(suffix(N, T))(<$N x $(d[T])> %0, $(d[T])* %ptr, <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, Ptr{T}, LVec{N, Bool}}, x, ptr, mask)
    )
end

@generated function maskedcompressstore(x::LVec{N, T}, ptr::LLVMPtr{T, AS},
                                        mask::LVec{N,Bool}) where {N, T, AS}
    mod = """
        declare void @llvm.masked.compressstore.$(suffix(N, T))(<$N x $(d[T])>, $(llvm_ptr(T, AS)), <$N x i1>)

        define void @entry(<$N x $(d[T])>, $(llvm_ptr(UInt8, AS)), <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptr = bitcast $(llvm_ptr(UInt8, AS)) %1 to $(llvm_ptr(T, AS))
            call void @llvm.masked.compressstore.$(suffix(N, T))(<$N x $(d[T])> %0, $(llvm_ptr(T, AS)) %ptr, <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, LLVMPtr{T, AS}, LVec{N, Bool}}, x, ptr, mask)
    )
end


####################
# Gather / Scatter #
####################

@generated function maskedgather(ptrs::LVec{N,Ptr{T}},
                                 mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, Al}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*>, i32, <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry(<$N x $(d[Ptr])>, <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptrs = $argtoptr <$N x $(d[Ptr])> %0 to <$N x $(d[T])*>
            %res = call <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(d[T])*> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{LVec{N, Ptr{T}}, LVec{N, Bool}}, ptrs, mask)
    )
end

@generated function maskedgather(ptrs::LVec{N,LLVMPtr{T, AS}},
                                 mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, AS, Al}
    # TODO: Allow setting the passthru
    mod = """
        declare <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(llvm_ptr(T, AS))>, i32, <$N x i1>, <$N x $(d[T])>)

        define <$N x $(d[T])> @entry(<$N x $(llvm_ptr(UInt8, AS))>, <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %1 to <$(N) x i1>
            %ptrs = bitcast <$N x $(llvm_ptr(UInt8, AS))> %0 to <$N x $(llvm_ptr(T, AS))>
            %res = call <$N x $(d[T])> @llvm.masked.gather.$(suffix(N, T))(<$N x $(llvm_ptr(T, AS))> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask, <$N x $(d[T])> zeroinitializer)
            ret <$N x $(d[T])> %res
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), LVec{N, T}, Tuple{LVec{N, LLVMPtr{T, AS}}, LVec{N, Bool}}, ptrs, mask)
    )
end


@generated function maskedscatter(x::LVec{N, T}, ptrs::LVec{N, Ptr{T}},
                                  mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, Al}
    mod = """
        declare void @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(d[T])*>, i32, <$N x i1>)

        define void @entry(<$N x $(d[T])>, <$N x $(d[Ptr])>, <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptrs = $argtoptr <$N x $(d[Ptr])> %1 to <$N x $(d[T])*>
            call void @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(d[T])*> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, LVec{N, Ptr{T}}, LVec{N, Bool}}, x, ptrs, mask)
    )
end

@generated function maskedscatter(x::LVec{N, T}, ptrs::LVec{N, LLVMPtr{T, AS}},
                                  mask::LVec{N,Bool}, ::Val{Al}=Val(false)) where {N, T, AS, Al}
    mod = """
        declare void @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])>, <$N x $(llvm_ptr(T, AS))>, i32, <$N x i1>)

        define void @entry(<$N x $(d[T])>, <$N x $(llvm_ptr(UInt8, AS))>, <$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %2 to <$(N) x i1>
            %ptrs = bitcast <$N x $(llvm_ptr(UInt8, AS))> %1 to <$N x $(llvm_ptr(T, AS))>
            call void @llvm.masked.scatter.$(suffix(N, T))(<$N x $(d[T])> %0, <$N x $(llvm_ptr(T, AS))> %ptrs, i32 $(n_align(Al, N, T)), <$N x i1> %mask)
            ret void
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Cvoid, Tuple{LVec{N, T}, LVec{N, LLVMPtr{T, AS}}, LVec{N, Bool}}, x, ptrs, mask)
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
            if !$criteria(sT1, sT2)
                str = "size of conversion type ($T2: $sT2) must be $($criteria) than the element type ($T1: $sT1)"
                return :(throw(ArgumentError($str)))
            end
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

@generated function inttoptr(::Type{LVec{N, Ptr{T2}}}, x::LVec{N, T1}) where {N, T1 <: IntegerTypes, T2 <: Union{IntegerTypes, FloatingTypes}}
    convert = VERSION >= v"1.12-DEV" ? "inttoptr" : "bitcast"
    s = """
    %2 = $convert <$(N) x $(d[T1])> %0 to <$(N) x $(d[Ptr])>
    ret <$(N) x $(d[Ptr])> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, Ptr{T2}}, Tuple{LVec{N, T1}}, x)
    )
end

@generated function inttoptr(::Type{LVec{N, LLVMPtr{T2, AS}}}, x::LVec{N, T1}) where {N, T1 <: IntegerTypes, T2 <: Union{IntegerTypes, FloatingTypes}, AS}
    s = """
    %2 = inttoptr <$(N) x $(d[T1])> %0 to <$(N) x $(llvm_ptr(UInt8, AS))>
    ret <$(N) x $(llvm_ptr(UInt8, AS))> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, LLVMPtr{T2, AS}}, Tuple{LVec{N, T1}}, x)
    )
end

@generated function ptrtoint(::Type{LVec{N, T2}}, x::LVec{N, Ptr{T1}}) where {N, T1 <: Union{IntegerTypes, FloatingTypes}, T2 <: IntegerTypes}
    convert = VERSION >= v"1.12-DEV" ? "ptrtoint" : "bitcast"
    s = """
    %2 = $convert <$(N) x $(d[Ptr])> %0 to <$(N) x $(d[T2])>
    ret <$(N) x $(d[T2])> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, Ptr{T1}}}, x)
    )
end

@generated function ptrtoint(::Type{LVec{N, T2}}, x::LVec{N, LLVMPtr{T1, AS}}) where {N, T1 <: Union{IntegerTypes, FloatingTypes}, T2 <: IntegerTypes, AS}
    s = """
    %2 = ptrtoint <$(N) x $(llvm_ptr(UInt8, AS))> %0 to <$(N) x $(d[T2])>
    ret <$(N) x $(d[T2])> %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, T2}, Tuple{LVec{N, LLVMPtr{T1, AS}}}, x)
    )
end

###########
# Bitcast #
###########

@generated function bitcast(::Type{T1}, x::T2) where {T1<:LT, T2<:LT}
    sT1, sT2 = sizeof(T1) * 8, sizeof(T2) * 8
    if sT1 != sT2
        return :(throw(ArgumentError(("size of conversion type ($($T1): $($sT1)) must be equal to the vector type ($($T2): $($sT2))"))))
    end
    s = """
    %2 = bitcast $(llvm_type(T2)) %0 to $(llvm_type(T1))
    ret $(llvm_type(T1)) %2
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, T1, Tuple{T2}, x)
    )
end

###########
# Bitmask #
###########

@generated function bitmask(x::LVec{N, Bool}) where {N}
    if N > 128 # Julia doesn't export anything larger than UInt128
        return :(throw(ArgumentError(("vector length $(N) must be <= 128"))))
    end
    P = nextpow(2, max(N, 8))
    T = Symbol("UInt", P)
    if N == P
        s = """
        %mask = trunc <$(N) x i8> %0 to <$(N) x i1>
        %maski = bitcast <$(N) x i1> %mask to i$(N)
        ret i$(N) %maski
        """
    else
        s = """
        %mask = trunc <$(N) x i8> %0 to <$(N) x i1>
        %maski = bitcast <$(N) x i1> %mask to i$(N)
        %maskizext = zext i$(N) %maski to i$(P)
        ret i$(P) %maskizext
        """
    end
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, $T, Tuple{LVec{N, Bool}}, x)
    )
end

##################################
# Horizontal reductions (LLVM 9) #
##################################

const SUPPORTS_FMAXIMUM_FMINIMUM = Base.libllvm_version >= v"18"

const HORZ_REDUCTION_OPS_FLOAT = [
    SUPPORTS_FMAXIMUM_FMINIMUM ? :fmaximum : :fmax
    SUPPORTS_FMAXIMUM_FMINIMUM ? :fminimum : :fmin
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

const horizontal_reduction_prefix = Base.libllvm_version < v"12" ? "experimental.vector.reduce." : "vector.reduce."
for (fs, c) in zip([HORZ_REDUCTION_OPS_FLOAT, HORZ_REDUCTION_OPS_INT],
                   [FloatingTypes,            IntegerTypes])
    for f in fs
        f_red = Symbol("reduce_", f)
        @eval @generated function $f_red(x::LVec{N, T}) where {N,T<:$c}
            ff = llvm_name(string(horizontal_reduction_prefix, $(QuoteNode(f))), N, T)
            mod = """
                declare $(d[T]) @$ff(<$N x $(d[T])>)

                define $(d[T]) @entry(<$N x $(d[T])>) #0 {
                top:
                    %res = call $(d[T]) @$ff(<$N x $(d[T])> %0)
                    ret $(d[T]) %res
                }

                attributes #0 = { alwaysinline }
            """
            return quote
                $(Expr(:meta, :inline));
                Base.llvmcall($(mod, "entry"), T, Tuple{LVec{N, T},}, x)
            end
        end
    end
end

# The fadd and fmul reductions take an initializer
const horz_reduction_version = (v"9" < Base.libllvm_version < v"12") ? "v2." : ""
const horz_experimental = Base.libllvm_version < v"12" ? "experimental." : ""
const horizontal_reduction_2arg_prefix =  "$(horz_experimental)vector.reduce.$horz_reduction_version"
for (f, neutral) in [(:fadd, "0.0"), (:fmul, "1.0")]
    f_red = Symbol("reduce_", f)
    @eval @generated function $f_red(x::LVec{N, T}, ::F=nothing) where {N,T<:FloatingTypes, F<:FPFlags}
        fpflags = fp_str(F)
        ff = llvm_name(string(horizontal_reduction_2arg_prefix, $(QuoteNode(f))), N, T)
        mod = """
            declare $(d[T]) @$ff($(d[T]), <$N x $(d[T])>)

            define $(d[T]) @entry(<$N x $(d[T])>) #0 {
            top:
                %res = call $(fpflags) $(d[T]) @$ff($(d[T]) $($neutral), <$N x $(d[T])> %0)
                ret $(d[T]) %res
            }

            attributes #0 = { alwaysinline }
        """
        return quote
            $(Expr(:meta, :inline));
            Base.llvmcall($(mod, "entry"), T, Tuple{LVec{N, T},}, x)
        end
    end
end

# See: https://llvm.org/docs/LangRef.html#id839
@generated function reduce_add(x::LVec{N, Bool}) where {N}
    native_bit_width = sizeof(Int) * 8
    if N < native_bit_width
        ret = """
        %res = zext i$(N) %maskipopcnt to i$(native_bit_width)
        ret i$(native_bit_width) %res
        """
    elseif N == native_bit_width
        ret = "ret i$(native_bit_width) %maskipopcnt"
    else
        ret = """
        %res = trunc i$(N) %maskipopcnt to i$(native_bit_width)
        ret i$(native_bit_width) %res
        """
    end
    mod = """
        declare i$(N) @llvm.ctpop.i$(N)(i$(N))

        define i$(native_bit_width) @entry(<$(N) x i8>) #0 {
        top:
            %mask = trunc <$(N) x i8> %0 to <$(N) x i1>
            %maski = bitcast <$(N) x i1> %mask to i$(N)
            %maskipopcnt = call i$(N) @llvm.ctpop.i$(N)(i$(N) %maski)
            $(ret)
        }

        attributes #0 = { alwaysinline }
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall(($mod, "entry"), Int, Tuple{LVec{N, Bool}}, x)
    )
end

###################################
# add_ptr (through getelementptr) #
###################################

@generated function add_ptr(ptr::LLVMPtr{T, AS}, i::LVec{N, I}) where {T, AS, N, I <: IntegerTypes}
    s = """
    %res = getelementptr i8, $(llvm_ptr(UInt8, AS)) %0, <$(N) x $(d[I])> %1
    ret <$(N) x $(llvm_ptr(UInt8, AS))> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, LLVMPtr{T, AS}}, Tuple{LLVMPtr{T, AS}, LVec{N, I}}, ptr, i)
    )
end

@generated function add_ptr(ptr::LVec{N, LLVMPtr{T, AS}}, i::I) where {T, AS, N, I <: IntegerTypes}
    s = """
    %res = getelementptr i8, <$(N) x $(llvm_ptr(UInt8, AS))> %0, $(d[I]) %1
    ret <$(N) x $(llvm_ptr(UInt8, AS))> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, LLVMPtr{T, AS}}, Tuple{LVec{N, LLVMPtr{T, AS}}, I}, ptr, i)
    )
end

@generated function add_ptr(ptr::LVec{N, LLVMPtr{T, AS}}, i::LVec{N, I}) where {T, AS, N, I <: IntegerTypes}
    s = """
    %res = getelementptr i8, <$(N) x $(llvm_ptr(UInt8, AS))> %0, <$(N) x $(d[I])> %1
    ret <$(N) x $(llvm_ptr(UInt8, AS))> %res
    """
    return :(
        $(Expr(:meta, :inline));
        Base.llvmcall($s, LVec{N, LLVMPtr{T, AS}}, Tuple{LVec{N, LLVMPtr{T, AS}}, LVec{N, I}}, ptr, i)
    )
end

end
