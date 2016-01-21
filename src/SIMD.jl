module SIMD

# Define an abstract type to parameterize the generated vector types
abstract AbstractVec{N,T}

Base.length{N,T}(::AbstractVec{N,T}) = N
Base.eltype{N,T}(::AbstractVec{N,T}) = T

# Create a vector type
function mkvectype{T}(typename::Symbol, N::Integer, ::Type{T})
    implname = symbol(typename, "_impl")
    @eval begin
        nbits = 8*$N*sizeof($T)
        bitstype nbits $implname
        immutable $typename <: AbstractVec{$N,$T}
            elts::$implname
            $typename(elts::$implname) = new(elts)
        end
        Base.length(::Type{$typename}) = $N
        Base.eltype(::Type{$typename}) = $T
    end
end

# Return a vector type, creating it if necessary
export Vec
@generated function Vec{N,T}(::Type{Val{N}}, ::Type{T})
    typename = symbol("Vec", N, "x", T)
    mkvectype(typename, N, T)
    typename
end
Vec{T}(N::Integer, ::Type{T}) = Vec(Val{N}, T)

# Convert Julia types to LLVM types
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

# Type-dependent LLVM intrinsics
llvmins{T<:Integer}(::Type{Val{:+}}, ::Type{T}) = "add"
llvmins{T<:Integer}(::Type{Val{:-}}, ::Type{T}) = "sub"
llvmins{T<:Integer}(::Type{Val{:*}}, ::Type{T}) = "mul"
llvmins{T<:Signed}(::Type{Val{:div}}, ::Type{T}) = "sdiv"
llvmins{T<:Signed}(::Type{Val{:rem}}, ::Type{T}) = "srem"
llvmins{T<:Unsigned}(::Type{Val{:div}}, ::Type{T}) = "udiv"
llvmins{T<:Unsigned}(::Type{Val{:rem}}, ::Type{T}) = "urem"

llvmins{T<:AbstractFloat}(::Type{Val{:+}}, ::Type{T}) = "fadd"
llvmins{T<:AbstractFloat}(::Type{Val{:-}}, ::Type{T}) = "fsub"
llvmins{T<:AbstractFloat}(::Type{Val{:*}}, ::Type{T}) = "fmul"
llvmins{T<:AbstractFloat}(::Type{Val{:/}}, ::Type{T}) = "fdiv"
llvmins{T<:AbstractFloat}(::Type{Val{:rem}}, ::Type{T}) = "frem"

llvmins{T<:AbstractFloat}(::Type{Val{:^}}, ::Type{T}) = "@llvm.pow"
llvmins{T<:AbstractFloat}(::Type{Val{:abs}}, ::Type{T}) = "@llvm.fabs"
llvmins{T<:AbstractFloat}(::Type{Val{:muladd}}, ::Type{T}) = "@llvm.fmuladd"
llvmins{T<:AbstractFloat}(::Type{Val{:sqrt}}, ::Type{T}) = "@llvm.fsqrt"

# Function wrappers
@generated function llvmwrap{Op,N,T}(::Type{Val{Op}}, v1::AbstractVec{N,T})
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ins = llvmins(Val{Op}, T)
    flags = fastflags(T)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    if ins[1] == '@'
        suffix = ".v$(N)f$(8*sizeof(T))"
        push!(decls, "declare <$N x $typ> $ins$suffix(<$N x $typ>)")
        push!(instrs, "%res = call <$N x $typ> $ins$suffix(<$N x $typ> %arg1)")
    else
        push!(instrs, "%res = $ins $flags <$N x $typ> zeroinitializer, %arg1")
    end
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$etyp}, v1.elts))
    end
end

@generated function llvmwrap{Op,N,T}(::Type{Val{Op}},
        v1::AbstractVec{N,T}, v2::AbstractVec{N,T})
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ins = llvmins(Val{Op}, T)
    flags = fastflags(T)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs, "%arg2 = bitcast $jtyp %1 to <$N x $typ>")
    if ins[1] == '@'
        suffix = ".v$(N)f$(8*sizeof(T))"
        push!(decls,
            "declare <$N x $typ> $ins$suffix(<$N x $typ>, <$N x $typ>)")
        push!(instrs,
            "%res = call <$N x $typ> $ins$suffix(<$N x $typ> %arg1, " *
                "<$N x $typ> %arg2)")
    else
        push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    end
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$etyp, $etyp}, v1.elts, v2.elts))
    end
end

@generated function llvmwrap{Op,N,T}(::Type{Val{Op}},
        v1::AbstractVec{N,T}, v2::AbstractVec{N,T}, v3::AbstractVec{N,T})
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ins = llvmins(Val{Op}, T)
    flags = fastflags(T)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs, "%arg2 = bitcast $jtyp %1 to <$N x $typ>")
    push!(instrs, "%arg3 = bitcast $jtyp %2 to <$N x $typ>")
    if ins[1] == '@'
        suffix = ".v$(N)f$(8*sizeof(T))"
        push!(decls,
            "declare <$N x $typ> $ins$suffix(<$N x $typ>, <$N x $typ>, " *
                "<$N x $typ>)")
        push!(instrs,
            "%res = call <$N x $typ> $ins$suffix(<$N x $typ> %arg1, " *
                "<$N x $typ> %arg2, <$N x $typ> %arg3)")
    else
        push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2, %arg3")
    end
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$etyp, $etyp, $etyp}, v1.elts, v2.elts, v3.elts))
    end
end

# Arithmetic functions

for op in (:+, :-, :abs, :sqrt)
    @eval begin
        Base.$op{N,T}(v1::AbstractVec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end

for op in (:+, :-, :*, :/, :div, :rem, :^)
    @eval begin
        Base.$op{N,T}(v1::AbstractVec{N,T}, v2::AbstractVec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end

for op in (:muladd,)
    @eval begin
        Base.$op{N,T}(v1::AbstractVec{N,T}, v2::AbstractVec{N,T},
                v3::AbstractVec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, v3)
    end
end

# Load and store functions

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

@generated function Base.convert{V<:AbstractVec}(::Type{V}, x::Number)
    N = length(V)
    T = eltype(V)
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    decls = []
    instrs = []
    append!(instrs, scalar2vector("%res", N, typ, "%0"))
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = Vec(N,T)
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$T}, $T(x)))
    end
end

@generated function Base.convert{V<:AbstractVec}(::Type{V}, xs::Tuple)
    N = length(V)
    T = eltype(V)
    @assert nfields(xs) == N
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    decls = []
    instrs = []
    append!(instrs, array2vector("%res", N, typ, "%0", "%resarr"))
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = Vec(N,T)
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{NTuple{$N,$T}}, NTuple{$N,$T}(xs)))
    end
end

@generated function Base.convert{N,T}(::Type{NTuple{N,T}}, v1::AbstractVec{N,T})
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    append!(instrs, vector2array("%res", N, typ, "%arg1"))
    push!(instrs, "ret [$N x $typ] %res")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{$N,$T}, Tuple{$etyp}, v1.elts)
    end
end

export setindex
@generated function setindex{N,T,I}(v1::AbstractVec{N,T}, ::Type{Val{I}}, x)
    @assert isa(I, Integer)
    @assert 1 <= I <= N
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs,
        "%res = insertelement <$N x $typ> %arg1, $typ %1, $ityp $(I-1)")
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$etyp, T}, v1.elts, T(x)))
    end
end

export setindex
@generated function setindex{N,T}(v1::AbstractVec{N,T}, i::Integer, x)
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs, "%res = insertelement <$N x $typ> %arg1, $typ %2, $ityp %1")
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        #TODO @boundscheck @assert 1 <= i <= N
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{$etyp, Int, T}, v1.elts, Int(i)-1, T(x)))
    end
end

@generated function Base.getindex{N,T,I}(v1::AbstractVec{N,T}, ::Type{Val{I}})
    @assert isa(I, Integer)
    @assert 1 <= I <= N
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs, "%res = extractelement <$N x $typ> %arg1, $ityp $(I-1)")
    push!(instrs, "ret $typ %res")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            T, Tuple{$etyp}, v1.elts)
    end
end

@generated function Base.getindex{N,T}(v1::AbstractVec{N,T}, i::Integer)
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    ityp = llvmtype(Int)
    decls = []
    instrs = []
    push!(instrs, "%arg1 = bitcast $jtyp %0 to <$N x $typ>")
    push!(instrs, "%res = extractelement <$N x $typ> %arg1, $ityp %1")
    push!(instrs, "ret $typ %res")
    vtyp = v1
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        #TODO @boundscheck @assert 1 <= i <= N
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            T, Tuple{$etyp, Int}, v1.elts, Int(i)-1)
    end
end

export vload, vloada
@generated function vload{V<:AbstractVec, T, Aligned}(::Type{V}, ptr::Ptr{T},
        ::Type{Val{Aligned}} = Val{false})
    @assert isa(Aligned, Bool)
    N = length(V)
    @assert T === eltype(V)
    jtyp = "i$(8*N*sizeof(T))"
    typ = llvmtype(T)
    decls = []
    instrs = []
    if Aligned
        align = N * sizeof(T)
    else
        align = sizeof(T)   # This is overly optimistic
    end
    flags = ", align $align"
    push!(instrs, "%ptr = bitcast $typ* %0 to <$N x $typ>*")
    push!(instrs, "%res = load <$N x $typ>, <$N x $typ>* %ptr$flags")
    push!(instrs, "%resbits = bitcast <$N x $typ> %res to $jtyp")
    push!(instrs, "ret $jtyp %resbits")
    # push!(instrs, "%ptr = bitcast $typ* %0 to $jtyp*")
    # push!(instrs, "%res = load $jtyp, $jtyp* %ptr$flags")
    # push!(instrs, "ret $jtyp %res")
    vtyp = Vec(N,T)
    etyp = fieldtype(vtyp, 1)
    quote
        # $(Expr(:meta, :inline))
        $vtyp(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            $etyp, Tuple{Ptr{T}}, ptr))
    end
end

vloada{V<:AbstractVec}(::Type{V}, ptr::Ptr) = vload(V, ptr, Val{true})

@inline function vload{V<:AbstractVec, Aligned}(::Type{V}, arr::Vector,
        i::Integer, ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck @assert 1 <= i < length(arr) - length(V)
    vload(V, pointer(arr, i), Val{Aligned})
end
vloada{V<:AbstractVec}(::Type{V}, arr::Vector, i::Integer) =
    vload(V, arr, i, Val{true})

end
