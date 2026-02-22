using Base: Slice, ScalarIndex
using Core: LLVMPtr

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
ContiguousArray{T,N} = Union{DenseArray{T,N}, ContiguousSubArray{T,N},
    Base.ReinterpretArray{T,N,T2,A} where {A <: Union{DenseArray{T2,N},
                                                      ContiguousSubArray{T2,N}}} where {T2}}

"""
    FastContiguousArray{T,N}

This is the type of arrays that `pointer(A, i)` works.
"""
FastContiguousArray{T,N} = Union{DenseArray{T,N}, Base.FastContiguousSubArray{T,N},
    Base.ReinterpretArray{T,N,T2,A} where {A <: Union{DenseArray{T2,N},
                                                      Base.FastContiguousSubArray{T2,N}}} where {T2}}
# https://github.com/eschnett/SIMD.jl/pull/40#discussion_r254131184
# https://github.com/JuliaArrays/MappedArrays.jl/pull/24#issuecomment-460568978

# vload
@propagate_inbounds function vload(::Type{Vec{N, T}}, ptr::AnyPtr{T}, mask::Union{Nothing, Vec{N, Bool}}=nothing,
                       ::Val{Aligned}=Val(false), ::Val{Nontemporal}=Val(false)) where {N, T, Aligned, Nontemporal}
    if mask === nothing
        Vec(Intrinsics.load(Intrinsics.LVec{N, T}, ptr, Val(Aligned), Val(Nontemporal)))
    else
        Vec(Intrinsics.maskedload(ptr, mask.data, Val(Aligned), Val(Nontemporal)))
    end
end

@propagate_inbounds function vload(::Type{Vec{N, T}}, a::FastContiguousArray{T,1}, i::Integer, mask=nothing,
                       ::Val{Aligned}=Val(false), ::Val{Nontemporal}=Val(false)) where {N, T, Aligned, Nontemporal}
    @boundscheck checkbounds(a, i:(i+N-1))
    GC.@preserve a begin
        ptr = pointer(a, i)
        vload(Vec{N, T}, ptr, mask, Val(Aligned), Val(Nontemporal))
    end
end
@propagate_inbounds vloada(::Type{Vec{N, T}}, ptr::AnyPtr{T}, mask=nothing) where {N, T} = vload(Vec{N, T}, ptr, mask, Val(true))
@propagate_inbounds vloadnt(::Type{Vec{N, T}}, ptr::AnyPtr{T}, mask=nothing) where {N, T} = vload(Vec{N, T}, ptr, mask, Val(true), Val(true))
@propagate_inbounds vloada(::Type{Vec{N, T}}, a::FastContiguousArray{T,1}, i::Integer, mask=nothing) where {N, T} = vload(Vec{N, T}, a, i, mask, Val(true))
@propagate_inbounds vloadnt(::Type{Vec{N, T}}, a::FastContiguousArray{T,1}, i::Integer, mask=nothing) where {N, T} = vload(Vec{N, T}, a, i, mask, Val(true), Val(true))

# vstore
@propagate_inbounds function vstore(x::Vec{N, T}, ptr::AnyPtr{T}, mask::Union{Nothing, Vec{N, Bool}}=nothing,
                       ::Val{Aligned}=Val(false), ::Val{Nontemporal}=Val(false)) where {N, T, Aligned, Nontemporal}
    if mask === nothing
        Intrinsics.store(x.data, ptr, Val(Aligned), Val(Nontemporal))
    else
        Intrinsics.maskedstore(x.data, ptr, mask.data, Val(Aligned), Val(Nontemporal))
    end
end
@propagate_inbounds function vstore(x::Vec{N, T}, a::FastContiguousArray{T,1}, i::Integer, mask=nothing,
               ::Val{Aligned}=Val(false), ::Val{Nontemporal}=Val(false)) where {N, T, Aligned, Nontemporal}
    @boundscheck checkbounds(a, i:(i+N-1))
    GC.@preserve a begin
        ptr = pointer(a, i)
        vstore(x, ptr, mask, Val(Aligned), Val(Nontemporal))
    end
    return a
end
@propagate_inbounds vstorea(x::Vec, ptr::AnyPtr, mask=nothing) = vstore(x, ptr, mask, Val(true))
@propagate_inbounds vstorent(x::Vec, ptr::AnyPtr, mask=nothing) = vstore(x, ptr, mask, Val(true), Val(true))
@propagate_inbounds vstorea(x::Vec, a, i, mask=nothing) = vstore(x, a, i, mask, Val(true))
@propagate_inbounds vstorent(x::Vec, a, i, mask=nothing) = vstore(x, a, i, mask, Val(true), Val(true))

@inline vloadx(ptr::AnyPtr, mask::Vec{<:Any, Bool}) =
    Vec(Intrinsics.maskedexpandload(ptr, mask.data))

@propagate_inbounds function vloadx(a::FastContiguousArray{T,1},
                                    i::Integer, mask::Vec{N, Bool}) where {N, T}
    @boundscheck checkbounds(a, i:i + N - 1)
    return GC.@preserve a begin
        ptr = pointer(a, i)
        vloadx(ptr, mask)
    end
end

@inline vstorec(x::Vec{N, T}, ptr::AnyPtr{T}, mask::Vec{N, Bool}) where {N, T} =
    Intrinsics.maskedcompressstore(x.data, ptr, mask.data)

@propagate_inbounds function vstorec(x::Vec{N, T}, a::FastContiguousArray{T,1},
                                     i::Integer, mask::Vec{N, Bool}) where {N, T}
    @boundscheck checkbounds(a, i:i + N - 1)
    GC.@preserve a begin
        ptr = pointer(a, i)
        vstorec(x, ptr, mask)
    end
    return a
end

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

@inline function _get_vec_pointers(a, idx::Vec{N, <:Integer}) where {N}
    ptrs = pointer(a) + (idx - 1) * sizeof(eltype(a))
end

# Have to be careful with optional arguments and @boundscheck,
# see https://github.com/JuliaLang/julia/issues/30411,
# therefore use @propagate_inbounds
@inline vgather(ptrs::Vec{N,<:AnyPtr{T}},
                 mask::Vec{N,Bool}=one(Vec{N,Bool}),
                 ::Val{Aligned}=Val(false)) where {N, T<:BScalarTypes, Aligned} =
    return Vec(Intrinsics.maskedgather(ptrs.data, mask.data))
@inline vgather(ptrs::Vec{<:Any,<:AnyPtr{<:BScalarTypes}}, ::Nothing, aligned::Val = Val(false)) =
    vgather(ptrs, one(Vec{length(ptrs),Bool}), aligned)
@propagate_inbounds function vgather(a::FastContiguousArray{T,1}, idx::Vec{N, <:Integer},
                                     mask::Vec{N,Bool}=one(Vec{N,Bool}),
                                     ::Val{Aligned}=Val(false)) where {N, T<:BScalarTypes, Aligned}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds idx[i])
    end
    GC.@preserve a begin
        ptrs = _get_vec_pointers(a, idx)
        return vgather(ptrs, mask, Val(Aligned))
    end
end
@propagate_inbounds vgathera(a, idx, mask) = vgather(a, idx, mask, Val(true))
@propagate_inbounds vgathera(a, idx::Vec{N}) where {N} = vgather(a, idx, one(Vec{N,Bool}), Val(true))
@propagate_inbounds vgathera(ptrs::Vec{N,<:AnyPtr{T}}, mask::Vec{N,Bool}) where {N,T} = vgather(ptrs, mask, Val(true))
@propagate_inbounds vgathera(ptrs::Vec{N,<:AnyPtr{T}}) where {N,T} = vgather(ptrs, one(Vec{N,Bool}), Val(true))

@propagate_inbounds Base.getindex(a::FastContiguousArray{T,1}, idx::Vec{N,<:Integer}) where {N,T} =
    vgather(a, idx)


@propagate_inbounds vscatter(x::Vec{N,T}, ptrs::Vec{N,<:AnyPtr{T}},
                             mask::Vec{N,Bool}, ::Val{Aligned}=Val(false)) where {N, T<:BScalarTypes, Aligned} =
    Intrinsics.maskedscatter(x.data, ptrs.data, mask.data)
@inline vscatter(x::Vec{N,T}, ptrs::Vec{N,<:AnyPtr{T}},
                 ::Nothing, aligned::Val=Val(false)) where {N, T<:BScalarTypes} =
    vscatter(x, ptrs, one(Vec{N, Bool}), aligned)
@propagate_inbounds function vscatter(x::Vec{N,T}, a::FastContiguousArray{T,1}, idx::Vec{N, <:Integer},
                                      mask::Vec{N,Bool}=one(Vec{N, Bool}),
                                      ::Val{Aligned}=Val(false)) where {N, T<:BScalarTypes, Aligned}
    @boundscheck for i in 1:N
        checkbounds(a, @inbounds idx[i])
    end
    GC.@preserve a begin
        ptrs = _get_vec_pointers(a, idx)
        vscatter(x, ptrs, mask, Val(Aligned))
    end
    return
end
@propagate_inbounds vscattera(x, a, idx, mask) = vscatter(x, a, idx, mask, Val(true))
@propagate_inbounds vscattera(x, a, idx::Vec{N}) where {N}  = vscatter(x, a, idx, one(Vec{N,Bool}), Val(true))
@propagate_inbounds vscattera(x::Vec{N,T}, ptrs::Vec{N,<:AnyPtr{T}}, mask::Vec{N,Bool}) where {N,T} = vscatter(x, ptrs, mask, Val(true))
@propagate_inbounds vscattera(x::Vec{N,T}, ptrs::Vec{N,<:AnyPtr{T}}) where {N,T} = vscatter(x, ptrs, one(Vec{N,Bool}), Val(true))

@propagate_inbounds Base.setindex!(a::FastContiguousArray{T,1}, v::Vec{N,T}, idx::Vec{N,<:Integer}) where {N, T} =
    vscatter(v, a, idx)


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

@inline _checkarity(::AbstractArray{<:Any,N}, ::Vararg{Any,N}) where {N} =
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

_checkarity(::AbstractArray{<:Any,N}, ::Vararg{Any,M}) where {N,M} =
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
Base.@propagate_inbounds _pointer(arr::Union{Array, Base.ReinterpretArray}, i, I) =
    pointer(arr, LinearIndices(arr)[i, I...])
Base.@propagate_inbounds _pointer(arr::Base.FastContiguousSubArray, i, I) =
    pointer(arr, (i, I...))
Base.@propagate_inbounds _pointer(arr::Base.FastContiguousSubArray, i, I::Tuple{}) =
    pointer(arr, i)
# must be separate methods to resolve ambiguity
Base.@propagate_inbounds _pointer(arr::Base.ReinterpretArray, i, I::Tuple{}) =
    pointer(arr, i)
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
    return vgather(ptrs, mask)
end

Base.@propagate_inbounds function Base.setindex!(
        arr::ContiguousArray{T}, v::Vec{N,T}, idx::Vec{N,<:Integer},
        args::Vararg{Union{Integer,Vec{N,Bool}}}) where {N,T}
    I, mask = _preprocessindices(arr, idx, args)
    ptrs = _pointer(arr, 1, I) - sizeof(T) + sizeof(T) * idx
    vscatter(v, ptrs, mask)
    return arr
end
