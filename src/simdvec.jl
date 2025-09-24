struct Vec{N, T<:VecTypes}
    data::LVec{N, T}
end

# Constructors
@inline Vec(v::NTuple{N, T}) where {N, T<:VecTypes} = Vec(VE.(v))
@inline Vec(v::Vararg{T, N}) where {N, T<:VecTypes} = Vec(v)
@inline Vec(v::Vec) = v
# Numbers defines this and it is needed in power_by_squaring...
Base.copy(v::Vec) = v

# No throwing versions of convert
@inline _unsafe_convert(::Type{T}, v) where {T <: IntegerTypes} = v % T
@inline _unsafe_convert(::Type{T}, v) where {T <: VecTypes} = convert(T, v)
@inline constantvector(v::T1, ::Type{Vec{N, T2}}) where {N, T1, T2} =
    Vec(Intrinsics.constantvector(_unsafe_convert(T2, v), Intrinsics.LVec{N, T2}))

@inline Vec{N, T}(v::Vec{N, T}) where {N, T<:VecTypes} = v
@inline Vec{N, T}(v::Vec{N, T}) where {N, T<:FloatingTypes} = v
@inline Vec{N, T1}(v::T2) where {N, T1<:VecTypes, T2<:VecTypes} = constantvector(v, Vec{N, T1})
@inline Vec{N, T1}(v::Vec{N, T2}) where {N, T1<:Union{IntegerTypes, Ptr}, T2<:Union{IntegerTypes, Ptr}} =
    convert(Vec{N, T1}, v)

@inline Base.convert(::Type{Vec{N,T}}, v::Vec{N,T}) where {N,T} = v
@inline function Base.convert(::Type{Vec{N, T1}}, v::Vec{N, T2}) where {T1, T2, N}
    if T1 <: Ptr || T1 <: LLVMPtr
        return Vec(Intrinsics.inttoptr(Intrinsics.LVec{N, T1}, v.data))
    elseif T1 <: BIntegerTypes
        if T2 <: Ptr || T2 <: LLVMPtr
            return Vec(Intrinsics.ptrtoint(Intrinsics.LVec{N, T1}, v.data))
        elseif T2 <: BIntegerTypes
            if sizeof(T1) < sizeof(T2)
                return Vec(Intrinsics.trunc(Intrinsics.LVec{N, T1}, v.data))
            elseif sizeof(T1) == sizeof(T2)
                return Vec(Intrinsics.bitcast(Intrinsics.LVec{N, T1}, v.data))
            else
                if T2 <: Union{UIntTypes, Bool}
                    return Vec(Intrinsics.zext(Intrinsics.LVec{N, T1}, v.data))
                else
                    return Vec(Intrinsics.sext(Intrinsics.LVec{N, T1}, v.data))
                end
            end
        elseif T2 <: FloatingTypes
            if T1 <: Union{UIntTypes, Bool}
                return Vec(Intrinsics.fptoui(Intrinsics.LVec{N, T1}, v.data))
            elseif T1 <: IntTypes
                return Vec(Intrinsics.fptosi(Intrinsics.LVec{N, T1}, v.data))
            end
        end
    elseif T1 <: FloatingTypes
        if T2 <: Union{UIntTypes, Bool}
            return Vec(Intrinsics.uitofp(Intrinsics.LVec{N, T1}, v.data))
        elseif T2 <: IntTypes
            return Vec(Intrinsics.sitofp(Intrinsics.LVec{N, T1}, v.data))
        elseif T2 <: FloatingTypes
            if sizeof(T1) < sizeof(T2)
                return Vec(Intrinsics.fptrunc(Intrinsics.LVec{N, T1}, v.data))
            else
                return Vec(Intrinsics.fpext(Intrinsics.LVec{N, T1}, v.data))
            end
        end
    end
    _unreachable()
end
@noinline _unreachable() = error("unreachable")

Base.Tuple(v::Vec) = map(i -> i.value, v.data)
Base.NTuple{N, T}(v::Vec{N}) where {T, N} = map(i -> convert(T, i.value), v.data)

Base.eltype(::Type{Vec{N,T}}) where {N,T} = T
Base.ndims( ::Type{Vec{N,T}}) where {N,T} = 1
Base.length(::Type{Vec{N,T}}) where {N,T} = N
Base.size(  ::Type{Vec{N,T}}) where {N,T} = (N,)
Base.size(  ::Type{Vec{N,T}}, n::Integer) where {N,T} = n > N ? 1 : (N,)[n]

Base.eltype(V::Vec) = eltype(typeof(V))
Base.ndims(V::Vec) = ndims(typeof(V))
Base.length(V::Vec) = length(typeof(V))
Base.size(V::Vec) = size(typeof(V))
Base.size(V::Vec, n::Integer) = size(typeof(V), n)

if VERSION <= v"1.4.0-rc1.0"
    function Base.show(io::IO, v::Vec{N,T}) where {N,T}
        print(io, "<$N x $T>[")
        join(io, [x.value for x in v.data], ", ")
        print(io, "]")
    end
else
    # This crashes on pre 1.4-rc2
    function Base.show(io::IO, v::Vec{N,T}) where {N,T}
        io = IOContext(io, :typeinfo => eltype(v))
        print(io, "<$N x $T>[")
        join(io, [sprint(show, x.value; context=io) for x in v.data], ", ")
        print(io, "]")
    end
end

@inline Base.checkbounds(v::Vec, i::IntegerTypes) =
(i < 1 || i > length(v.data)) && Base.throw_boundserror(v, i)

@propagate_inbounds function Base.getindex(v::Vec, i::IntegerTypes)
    @boundscheck checkbounds(v, i)
    return Intrinsics.extractelement(v.data, i-1)
end

@propagate_inbounds function Base.setindex(v::Vec{N,T}, x, i::IntegerTypes) where {N,T}
    @boundscheck checkbounds(v, i)
    Vec(Intrinsics.insertelement(v.data, _unsafe_convert(T, x), i-1))
end

Base.zero(::Type{Vec{N,T}}) where {N, T} = Vec{N,T}(zero(T))
Base.zero(::Vec{N,T}) where {N, T} = zero(Vec{N, T})
Base.one(::Type{Vec{N,T}}) where {N, T} = Vec{N, T}(one(T))
Base.one(::Vec{N,T}) where {N, T} = one(Vec{N, T})

Base.reinterpret(::Type{Vec{N, T}}, v::Vec) where {T, N} = Vec(Intrinsics.bitcast(Intrinsics.LVec{N, T}, v.data))
Base.reinterpret(::Type{Vec{N, T}}, v::ScalarTypes) where {T, N} = Vec(Intrinsics.bitcast(Intrinsics.LVec{N, T}, v))
Base.reinterpret(::Type{T}, v::Vec) where {T} = Intrinsics.bitcast(T, v.data)

bitmask(v::Vec{<:Any,Bool}) = Intrinsics.bitmask(v.data)

const FASTMATH = Intrinsics.FastMathFlags(Intrinsics.FastMath.fast)

###################
# Unary operators #
###################

const UNARY_OPS = [
    (:sqrt           , FloatingTypes , Intrinsics.sqrt)       ,
    (:sin            , FloatingTypes , Intrinsics.sin)        ,
    (:trunc          , FloatingTypes , Intrinsics.trunc)      ,
    (:cos            , FloatingTypes , Intrinsics.cos)        ,
    (:exp            , FloatingTypes , Intrinsics.exp)        ,
    (:exp2           , FloatingTypes , Intrinsics.exp2)       ,
    (:log            , FloatingTypes , Intrinsics.log)        ,
    (:log10          , FloatingTypes , Intrinsics.log10)      ,
    (:log2           , FloatingTypes , Intrinsics.log2)       ,
    (:abs            , FloatingTypes , Intrinsics.fabs)       ,
    (:floor          , FloatingTypes , Intrinsics.floor)      ,
    (:ceil           , FloatingTypes , Intrinsics.ceil)       ,
    # (:rint         , FloatingTypes , Intrinsics)            ,
    # (:nearbyint    , FloatingTypes , Intrinsics)            ,
    (:round          , FloatingTypes , Intrinsics.round)      ,

    (:bswap          , IntegerTypes  , Intrinsics.bswap)      ,
    (:count_ones     , IntegerTypes  , Intrinsics.ctpop)      ,
    (:leading_zeros  , IntegerTypes  , Intrinsics.ctlz)       ,
    (:trailing_zeros , IntegerTypes  , Intrinsics.cttz)       ,
]

if isdefined(Base, :bitreverse)
    push!(UNARY_OPS,
        (:bitreverse   , IntegerTypes  , Intrinsics.bitreverse)
    )
end

for (op, constraint, llvmop) in UNARY_OPS
    @eval @inline (Base.$op)(x::Vec{<:Any, <:$constraint}) =
        Vec($(llvmop)(x.data))
end

# Fix up bswap (see <https://github.com/eschnett/SIMD.jl/issues/122>):
# bswap for 1-byte types is a no-op
Base.bswap(x::Vec{<:Any, <:Union{Int8,UInt8}}) = x

Base.:+(v::Vec{<:Any, <:ScalarTypes}) = v
Base.:-(v::Vec{<:Any, <:IntegerTypes}) = zero(v) - v
Base.:-(v::Vec{<:Any, <:FloatingTypes}) = Vec(Intrinsics.fneg(v.data))
Base.FastMath.sub_fast(v::Vec{<:Any, <:FloatingTypes}) = Vec(Intrinsics.fneg(v.data, FASTMATH))
Base.:~(v::Vec{N, T}) where {N, T<:IntegerTypes} = Vec(Intrinsics.xor(v.data, Vec{N, T}(-1).data))
Base.:~(v::Vec{N, Bool}) where {N} = Vec(Intrinsics.xor(v.data, Vec{N, Bool}(true).data))
Base.abs(v::Vec{N, T}) where {N, T} = Vec(vifelse(v < zero(T), -v, v))
Base.abs2(v::Vec{N, T}) where {N, T} = v*v
Base.:!(v1::Vec{N,Bool}) where {N} = ~v1
Base.inv(v::Vec{N, T}) where {N, T<:FloatingTypes} = one(T) / v

_unsigned(::Type{Float16}) = UInt16
_unsigned(::Type{Float32}) = UInt32
_unsigned(::Type{Float64}) = UInt64
function Base.issubnormal(x::Vec{N, T}) where {N, T<:FloatingTypes}
    y = reinterpret(Vec{N, _unsigned(T)}, x)
    (y & Base.exponent_mask(T) == 0) & (y & Base.significand_mask(T) != 0)
end

@inline Base.signbit(x::Vec{N, <:IntegerTypes}) where {N} = x < 0

@inline Base.leading_ones(x::Vec{<:Any, <:IntegerTypes})  = leading_zeros(~(x))
@inline Base.trailing_ones(x::Vec{<:Any, <:IntegerTypes}) = trailing_zeros(~(x))
@inline Base.count_zeros(x::Vec{<:Any, <:IntegerTypes}) = count_ones(~(x))

@inline Base.isnan(v::Vec{<:Any, <:FloatingTypes}) = v != v
@inline Base.isfinite(v::Vec{<:Any, <:FloatingTypes}) = v - v == zero(v)
@inline Base.isinf(v::Vec{<:Any, <:FloatingTypes}) = !isnan(v) & !isfinite(v)
@inline Base.sign(v1::Vec{N,T}) where {N,T} =
    vifelse(v1 == zero(Vec{N,T}), zero(Vec{N,T}),
            vifelse(v1 < zero(Vec{N,T}), -one(Vec{N,T}), one(Vec{N,T})))

@inline Base.isnan(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N,Bool})
@inline Base.isfinite(v::Vec{N, <:IntegerTypes}) where {N} = one(Vec{N, Bool})
@inline Base.isinf(v::Vec{N, <:IntegerTypes}) where {N} = zero(Vec{N, Bool})


####################
# Binary operators #
####################

const BINARY_OPS = [
    (:(Base.:+)        , IntegerTypes  , Intrinsics.add)
    (:(Base.:-)        , IntegerTypes  , Intrinsics.sub)
    (:(Base.:*)        , IntegerTypes  , Intrinsics.mul)
    (:(Base.div)       , UIntTypes     , Intrinsics.udiv)
    (:(Base.div)       , IntTypes      , Intrinsics.sdiv)
    (:(Base.rem)       , UIntTypes     , Intrinsics.urem)
    (:(Base.rem)       , IntTypes      , Intrinsics.srem)

    (:(add_saturate) , IntTypes  , Intrinsics.sadd_sat)
    (:(add_saturate) , UIntTypes , Intrinsics.uadd_sat)
    (:(sub_saturate) , IntTypes  , Intrinsics.ssub_sat)
    (:(sub_saturate) , UIntTypes , Intrinsics.usub_sat)

    (:(Base.:+)        , FloatingTypes , Intrinsics.fadd)
    (:(Base.:-)        , FloatingTypes , Intrinsics.fsub)
    (:(Base.:*)        , FloatingTypes , Intrinsics.fmul)
    (:(Base.:^)        , FloatingTypes , Intrinsics.pow)
    (:(Base.:/)        , FloatingTypes , Intrinsics.fdiv)
    (:(Base.rem)       , FloatingTypes , Intrinsics.frem)
    (:(Base.min)       , FloatingTypes , Intrinsics.minnum)
    (:(Base.max)       , FloatingTypes , Intrinsics.maxnum)
    (:(Base.copysign)  , FloatingTypes , Intrinsics.copysign)
    (:(Base.:~)        , BIntegerTypes , Intrinsics.xor)
    (:(Base.:&)        , BIntegerTypes , Intrinsics.and)
    (:(Base.:|)        , BIntegerTypes , Intrinsics.or)
    (:(Base.:⊻)        , BIntegerTypes , Intrinsics.xor)

    (:(Base.:(==))   , BIntegerTypes  , Intrinsics.icmp_eq)
    (:(Base.:!=)     , BIntegerTypes  , Intrinsics.icmp_ne)
    (:(Base.:>)      , BIntTypes      , Intrinsics.icmp_sgt)
    (:(Base.:>=)     , BIntTypes      , Intrinsics.icmp_sge)
    (:(Base.:<)      , BIntTypes      , Intrinsics.icmp_slt)
    (:(Base.:<=)     , BIntTypes      , Intrinsics.icmp_sle)
    (:(Base.:>)      , UIntTypes      , Intrinsics.icmp_ugt)
    (:(Base.:>=)     , UIntTypes      , Intrinsics.icmp_uge)
    (:(Base.:<)      , UIntTypes      , Intrinsics.icmp_ult)
    (:(Base.:<=)     , UIntTypes      , Intrinsics.icmp_ule)

    (:(Base.:(==))   , FloatingTypes , Intrinsics.fcmp_oeq)
    (:(Base.:!=)     , FloatingTypes , Intrinsics.fcmp_une)
    (:(Base.:>)      , FloatingTypes , Intrinsics.fcmp_ogt)
    (:(Base.:>=)     , FloatingTypes , Intrinsics.fcmp_oge)
    (:(Base.:<)      , FloatingTypes , Intrinsics.fcmp_olt)
    (:(Base.:<=)     , FloatingTypes , Intrinsics.fcmp_ole)
]

function get_fastmath_function(op)
    if op isa Expr && op.head == Symbol(".") && op.args[1] == :Base &&
        op.args[2].value in keys(Base.FastMath.fast_op)
        return :(Base.FastMath.$(Base.FastMath.fast_op[op.args[2].value]))
    end
    return nothing
end

for (op, constraint, llvmop) in BINARY_OPS
    @eval @inline function $op(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
        Vec($(llvmop)(x.data, y.data))
    end

    # Add a fast math version if applicable
    if (fast_op = get_fastmath_function(op)) !== nothing
        @eval @inline function $(fast_op)(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
            Vec($(llvmop)(x.data, y.data, FASTMATH))
        end
    end
end

# overflow
const OVERFLOW_INTRINSICS = [
    (:(Base.Checked.add_with_overflow) , IntTypes  , Intrinsics.sadd_with_overflow)
    (:(Base.Checked.add_with_overflow) , UIntTypes , Intrinsics.uadd_with_overflow)
    (:(Base.Checked.sub_with_overflow) , IntTypes  , Intrinsics.ssub_with_overflow)
    (:(Base.Checked.sub_with_overflow) , UIntTypes , Intrinsics.usub_with_overflow)
    (:(Base.Checked.mul_with_overflow) , IntTypes  , Intrinsics.smul_with_overflow)
    (:(Base.Checked.mul_with_overflow) , UIntTypes , Intrinsics.umul_with_overflow)
]
for (op, constraint, llvmop) in OVERFLOW_INTRINSICS
    @eval @inline function $op(x::Vec{N, T}, y::Vec{N, T}) where {N, T <: $constraint}
        val, overflows = $(llvmop)(x.data, y.data)
        return Vec(val), Vec(overflows)
    end
end

# max min
@inline Base.max(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    Vec(vifelse(v1 >= v2, v1, v2))
@inline Base.min(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntegerTypes} =
    Vec(vifelse(v1 >= v2, v2, v1))

# Pow
@inline Base.:^(x::Vec{N,T}, y::IntegerTypes) where {N,T<:FloatingTypes} =
    Vec(Intrinsics.powi(x.data, y))
# Do what Base does for HWNumber:
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{0}) = one(typeof(x))
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{1}) = x
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{2}) = x*x
@inline Base.literal_pow(::typeof(^), x::Vec, ::Val{3}) = x*x*x

# Sign
@inline Base.flipsign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T} =
    vifelse(signbit(v2), -v1, v1)
@inline Base.copysign(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T<:IntTypes} =
    vifelse(signbit(v2), -abs(v1), abs(v1))
_signed(::Type{Float16}) = Int16
_signed(::Type{Float32}) = Int32
_signed(::Type{Float64}) = Int64
@inline Base.signbit(x::Vec{N, T}) where {N, T <:FloatingTypes} =
    signbit(reinterpret(Vec{N, _signed(T)}, x))

# Pointer arithmetic
# Cast pointer to Int and back
@inline Base.:+(x::Vec{N,Ptr{T}}, y::Vec{N,<:IntegerTypes}) where {N,T} = convert(Vec{N,Ptr{T}}, convert(Vec{N,UInt}, x) + y)
@inline Base.:-(x::Vec{N,Ptr{T}}, y::Vec{N,<:IntegerTypes}) where {N,T} = convert(Vec{N,Ptr{T}}, convert(Vec{N,UInt}, x) - y)
@inline Base.:+(x::Ptr{T}, y::Vec{N,<:IntegerTypes}) where {N,T} = convert(Vec{N,Ptr{T}}, convert(UInt, x) + y)
@inline Base.:-(x::Ptr{T}, y::Vec{N,<:IntegerTypes}) where {N,T} = convert(Vec{N,Ptr{T}}, convert(UInt, x) - y)
@inline Base.:+(x::Vec{N,Ptr{T}}, y::IntegerTypes) where {N,T} = convert(Vec{N,Ptr{T}}, convert(Vec{N,UInt}, x) + y)
@inline Base.:-(x::Vec{N,Ptr{T}}, y::IntegerTypes) where {N,T} = convert(Vec{N,Ptr{T}}, convert(Vec{N,UInt}, x) - y)

# use gep
@inline Base.:+(x::Vec{N,LLVMPtr{T, AS}}, y::Vec{N,<:IntegerTypes}) where {N,T,AS} = Vec(Intrinsics.add_ptr(x.data, y.data))
@inline Base.:-(x::Vec{N,LLVMPtr{T, AS}}, y::Vec{N,<:IntegerTypes}) where {N,T,AS} = Vec(Intrinsics.add_ptr(x.data, (-y).data))
@inline Base.:+(x::LLVMPtr{T, AS}, y::Vec{N,<:IntegerTypes}) where {N,T,AS} = Vec(Intrinsics.add_ptr(x, y.data))
@inline Base.:-(x::LLVMPtr{T, AS}, y::Vec{N,<:IntegerTypes}) where {N,T,AS} = Vec(Intrinsics.add_ptr(x, (-y).data))
@inline Base.:+(x::Vec{N,LLVMPtr{T, AS}}, y::IntegerTypes) where {N,T,AS} = Vec(Intrinsics.add_ptr(x.data, y))
@inline Base.:-(x::Vec{N,LLVMPtr{T, AS}}, y::IntegerTypes) where {N,T,AS} = Vec(Intrinsics.add_ptr(x.data, -y))

@inline Base.:+(y::Vec{N,<:IntegerTypes}, x::Vec{N,<:AnyPtr{T}}) where {N,T} = x + y
@inline Base.:+(y::Vec{N,<:IntegerTypes}, x::AnyPtr{T}) where {N,T} = x + y
@inline Base.:+(y::IntegerTypes, x::Vec{N,<:AnyPtr{T}}) where {N,T} = x + y

@inline Base.:-(x::Vec{N,<:AnyPtr{T}}, y::Vec{N,<:AnyPtr{T}}) where {N,T} = convert(Vec{N,Int}, x) - convert(Vec{N,Int}, y)
@inline Base.:-(x::AnyPtr{T}, y::Vec{N,<:AnyPtr{T}}) where {N,T} = UInt(x) % Int - convert(Vec{N,Int}, y)
@inline Base.:-(x::Vec{N,<:AnyPtr{T}}, y::AnyPtr{T}) where {N,T} = convert(Vec{N,Int}, x) - UInt(y) % Int

# Bitshifts
# See https://github.com/JuliaLang/julia/blob/7426625b5c07b0d93110293246089a259a0a677d/src/intrinsics.cpp#L1179-L1196
# Shifting with a value larger than the number of bits in the type is undefined behavior
# so set to zero in those cases.
@inline function shl_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y >= sizeof(T1) * 8,
        zero(Vec{N, T1}),
        Vec(Intrinsics.shl(x.data, convert(Vec{N,T1}, y).data)))
end

@inline function lshr_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y >= sizeof(T1) * 8,
        zero(Vec{N, T1}),
        Vec(Intrinsics.lshr(x.data, convert(Vec{N,T1}, y).data)))
end

@inline function ashr_int(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes}
    vifelse(y >= sizeof(T1) * 8,
            Vec(Intrinsics.ashr(x.data, Vec{N,T1}(sizeof(T1)*8-1).data)),
            Vec(Intrinsics.ashr(x.data, Vec{N,T1}(y).data)))
end

# See https://github.com/JuliaLang/julia/blob/a211abcdfacc05cb93c15774a59ce8961c16dac4/base/int.jl#L422-L435
@inline Base.:>>(x::Vec{N, <:IntTypes}, y::Vec{N, <:UIntTypes}) where {N} =
    ashr_int(x, y)
@inline Base.:>>(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:UIntTypes, T2<:UIntTypes} =
    lshr_int(x, y)
@inline Base.:<<(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:UIntTypes} =
    shl_int(x, y)
@inline Base.:>>>(x::Vec{N, T1}, y::Vec{N, T2}) where {N, T1<:IntegerTypes, T2<:UIntTypes} =
    lshr_int(x, y)

@inline unsigned(v::Vec{<:Any, <:UIntTypes}) = v
@inline unsigned(v::Vec{N, Int32}) where {N} = convert(Vec{N, UInt32}, v)
@inline unsigned(v::Vec{N, Int64}) where {N} = convert(Vec{N, UInt64}, v)

@inline Base.:>>(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x >> unsigned(y), x << unsigned(-y))
@inline Base.:<<(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x << unsigned(y), x >> unsigned(-y))
@inline Base.:>>>(x::Vec{N, T1}, y::Vec{N, Int}) where {N, T1<:IntegerTypes} =
    vifelse(0 <= y, x >>> unsigned(y), x << unsigned(-y))

for v in (:<<, :>>, :>>>)
    @eval begin
        @inline Base.$v(x::Vec{N,T}, y::ScalarTypes) where {N, T} = $v(x, Vec{N,T}(y))
        @inline Base.$v(x::Vec{N,T}, y::T2) where {N, T<:IntegerTypes, T2<:UIntTypes} = $v(x, Vec{N,T2}(y))
        @inline Base.$v(x::ScalarTypes, y::Vec{N,T}) where {N, T} = $v(Vec{N,T}(x), y)
        @inline Base.$v(x::Vec{N,T1}, y::Vec{N,T2}) where {N, T1<:IntegerTypes, T2<:IntegerTypes} =
            $v(x, convert(Vec{N, Int}, y))
    end
end


# Vectorize binary functions
for (op, constraint) in [BINARY_OPS;
        (:(Base.flipsign) , ScalarTypes)
        (:(Base.copysign) , ScalarTypes)
        (:(Base.signbit)  , ScalarTypes)
        (:(Base.min)      , IntegerTypes)
        (:(Base.max)      , IntegerTypes)
        (:(Base.:<<)      , IntegerTypes)
        (:(Base.:>>)      , IntegerTypes)
        (:(Base.:>>>)     , IntegerTypes)
        (:(Base.Checked.add_with_overflow) , IntTypes)
        (:(Base.Checked.add_with_overflow) , UIntTypes)
        (:(Base.Checked.sub_with_overflow) , IntTypes)
        (:(Base.Checked.sub_with_overflow) , UIntTypes)
        (:(Base.Checked.mul_with_overflow) , IntTypes)
        (:(Base.Checked.mul_with_overflow) , UIntTypes)
    ]
    ops = [op]
    if (fast_op = get_fastmath_function(op)) !== nothing
        push!(ops, fast_op)
    end
    for op in ops
        @eval @inline function $op(x::T2, y::Vec{N, T}) where {N, T2<:ScalarTypes, T <: $constraint}
            $op(Vec{N, T}(x), y)
        end
        @eval @inline function $op(x::Vec{N, T}, y::T2) where {N, T2 <:ScalarTypes, T <: $constraint}
            $op(x, Vec{N, T}(y))
        end
    end
end

#####################
# Ternary operators #
#####################

@inline vifelse(v::Bool, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} = ifelse(v, v1, v2)
@inline vifelse(v::Bool, v1::Vec{N, T}, v2::ScalarTypes) where {N, T} = ifelse(v, v1, Vec{N,T}(v2))
@inline vifelse(v::Bool, v1::ScalarTypes, v2::Vec{N, T}) where {N, T} = ifelse(v, Vec{N,T}(v1), v2)

@inline vifelse(v::Bool, v1::T, v2::T) where {T} = ifelse(v, v1, v2)
@inline vifelse(v::Vec{N, Bool}, v1::Vec{N, T}, v2::Vec{N, T}) where {N, T} =
    Vec(Intrinsics.select(v.data, v1.data, v2.data))
@inline vifelse(v::Vec{N, Bool}, v1::T2, v2::Vec{N, T}) where {N, T, T2 <:ScalarTypes} = vifelse(v, Vec{N, T}(v1), v2)
@inline vifelse(v::Vec{N, Bool}, v1::Vec{N, T}, v2::T2) where {N, T, T2 <:ScalarTypes} = vifelse(v, v1, Vec{N, T}(v2))

# fma, muladd and vectorization of these
for (op, llvmop) in [(:fma, Intrinsics.fma), (:muladd, Intrinsics.fmuladd)]
    @eval begin
        @inline Base.$op(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N,T<:FloatingTypes} =
            Vec($llvmop(a.data, b.data, c.data))
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T}, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, v3)
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), v3)
        @inline Base.$op(s1::ScalarTypes, s2::ScalarTypes, v3::Vec{N,T}) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), Vec{N,T}(s2), v3)
        @inline Base.$op(v1::Vec{N,T}, v2::Vec{N,T}, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, v2, Vec{N,T}(s3))
        @inline Base.$op(s1::ScalarTypes, v2::Vec{N,T}, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(Vec{N,T}(s1), v2, Vec{N,T}(s3))
        @inline Base.$op(v1::Vec{N,T}, s2::ScalarTypes, s3::ScalarTypes) where {N,T<:FloatingTypes} =
            $op(v1, Vec{N,T}(s2), Vec{N,T}(s3))
    end
end

if isdefined(Base, :bitrotate)
    @inline Base.bitrotate(x::Vec, k::Vec) = Vec(Intrinsics.fshl(x.data, x.data, k.data))
    @inline Base.bitrotate(x::Vec{N, T}, k::Integer) where {N, T} = bitrotate(x, Vec{N, T}(k))
end


##############
# Reductions #
##############
const HORZ_REDUCTION_OPS = [
    (&   , Union{IntegerTypes, Bool}  , Intrinsics.reduce_and)
    (|   , Union{IntegerTypes, Bool}  , Intrinsics.reduce_or)
    (max , IntTypes      , Intrinsics.reduce_smax)
    (max , UIntTypes     , Intrinsics.reduce_umax)
    (max , FloatingTypes , Intrinsics.SUPPORTS_FMAXIMUM_FMINIMUM ? Intrinsics.reduce_fmaximum : Intrinsics.reduce_fmax)
    (min , IntTypes      , Intrinsics.reduce_smin)
    (min , UIntTypes     , Intrinsics.reduce_umin)
    (min , FloatingTypes , Intrinsics.SUPPORTS_FMAXIMUM_FMINIMUM ? Intrinsics.reduce_fminimum : Intrinsics.reduce_fmin)
    (+   , Union{IntegerTypes, Bool}  , Intrinsics.reduce_add)
    (*   , IntegerTypes  , Intrinsics.reduce_mul)
    (+   , FloatingTypes , Intrinsics.reduce_fadd)
    (*   , FloatingTypes , Intrinsics.reduce_fmul)
]

for (op, constraint, llvmop) in HORZ_REDUCTION_OPS
    @eval @inline Base.reduce(::typeof($op), x::Vec{<:Any, <:$constraint}) =
        $(llvmop)(x.data)
end
Base.reduce(F::Any, v::Vec) = error("reduction not defined for SIMD.Vec on $F")

@inline Base.all(v::Vec{<:Any,Bool}) = reduce(&, v)
@inline Base.any(v::Vec{<:Any,Bool}) = reduce(|, v)
@inline Base.maximum(v::Vec) = reduce(max, v)
@inline Base.minimum(v::Vec) = reduce(min, v)
@inline Base.prod(v::Vec) = reduce(*, v)
@inline Base.sum(v::Vec) = reduce(+, v)
@inline Base.prod(v::Vec{<:Any, <:FloatingTypes}) = Intrinsics.reduce_fmul(v.data, Intrinsics.FastMathFlags(Intrinsics.FastMath.reassoc))
@inline Base.sum(v::Vec{<:Any, <:FloatingTypes}) = Intrinsics.reduce_fadd(v.data, Intrinsics.FastMathFlags(Intrinsics.FastMath.reassoc))

############
# Shuffles #
############

@inline function shufflevector(x::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(Intrinsics.shufflevector(x.data, Val(I)))
end
@inline function shufflevector(x::Vec{N, T}, y::Vec{N, T}, ::Val{I}) where {N, T, I}
    Vec(Intrinsics.shufflevector(x.data, y.data, Val(I)))
end

############################
# Complex Number functions #
############################

@inline Base.real(v::Vec{N, T}) where {N, T<:ScalarTypes} = v
@inline Base.conj(v::Vec{N, T}) where {N, T<:ScalarTypes} = v
@inline Base.angle(v::Vec{N, T}) where {N, T <: FloatingTypes} = vifelse(signbit(v), Vec{N, T}(T(pi)), zero(v))
@inline Base.imag(v::Vec{N, T}) where {N, T <: ScalarTypes} = zero(v)
