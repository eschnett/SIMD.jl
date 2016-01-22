module SIMD

# The Julia SIMD vector type

export Vec
immutable Vec{N,T}
    elts::NTuple{N,T}
    Vec(elts::NTuple{N,T}) = new(elts)
end

# Type properties

Base.length{N,T}(::Type{Vec{N,T}}) = N
Base.eltype{N,T}(::Type{Vec{N,T}}) = T
Base.length{N,T}(::Vec{N,T}) = N
Base.eltype{N,T}(::Vec{N,T}) = T

# Type conversion

@generated function Base.convert{N,T}(::Type{Vec{N,T}}, x::Number)
    quote
        # :(Expr(:meta, :inline))
        let y = T(x)
            Vec{N,T}($(Expr(:tuple, [:x for i in 1:N]...)))
        end
    end
end

@generated function Base.convert{N,T}(::Type{Vec{N,T}}, xs::Tuple)
    @assert nfields(x) == N
    quote
        # :(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple, [:(T(xs[$i])) for i in 1:N])))
    end
end

@generated function Base.convert{N,T}(::Type{Vec{N,T}}, xs::Tuple)
    @assert nfields(xs) == N
    quote
        # :(Expr(:meta, :inline))
        Vec{N,T}($(Expr(:tuple, [:(T(xs[$i])) for i in 1:N]...)))
    end
end

@generated function Base.convert{R<:Tuple,N,T}(::Type{R}, v::Vec{N,T})
    @assert nfields(R) == N
    quote
        # :(Expr(:meta, :inline))
        R($(Expr(:tuple, [:($(fieldtype(R, i))(v.elts[$i])) for i in 1:N]...)))
    end
end

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

# Type-dependent LLVM intrinsics
llvmins{T<:Integer}(::Type{Val{:+}}, N, ::Type{T}) = "add"
llvmins{T<:Integer}(::Type{Val{:-}}, N, ::Type{T}) = "sub"
llvmins{T<:Integer}(::Type{Val{:*}}, N, ::Type{T}) = "mul"
llvmins{T<:Signed}(::Type{Val{:div}}, N, ::Type{T}) = "sdiv"
llvmins{T<:Signed}(::Type{Val{:rem}}, N, ::Type{T}) = "srem"
llvmins{T<:Unsigned}(::Type{Val{:div}}, N, ::Type{T}) = "udiv"
llvmins{T<:Unsigned}(::Type{Val{:rem}}, N, ::Type{T}) = "urem"
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

llvmins{T<:AbstractFloat}(::Type{Val{:+}}, N, ::Type{T}) = "fadd"
llvmins{T<:AbstractFloat}(::Type{Val{:-}}, N, ::Type{T}) = "fsub"
llvmins{T<:AbstractFloat}(::Type{Val{:*}}, N, ::Type{T}) = "fmul"
llvmins{T<:AbstractFloat}(::Type{Val{:/}}, N, ::Type{T}) = "fdiv"
llvmins{T<:AbstractFloat}(::Type{Val{:rem}}, N, ::Type{T}) = "frem"
llvmins{T<:AbstractFloat}(::Type{Val{:(==)}}, N, ::Type{T}) = "fcmp oeq"
llvmins{T<:AbstractFloat}(::Type{Val{:(!=)}}, N, ::Type{T}) = "fcmp une"
llvmins{T<:AbstractFloat}(::Type{Val{:(>)}}, N, ::Type{T}) = "fcmp ogt"
llvmins{T<:AbstractFloat}(::Type{Val{:(>=)}}, N, ::Type{T}) = "fcmp oge"
llvmins{T<:AbstractFloat}(::Type{Val{:(<)}}, N, ::Type{T}) = "fcmp olt"
llvmins{T<:AbstractFloat}(::Type{Val{:(<=)}}, N, ::Type{T}) = "fcmp ole"

llvmins{T<:AbstractFloat}(::Type{Val{:ifelse}}, N, ::Type{T}) = "select"

llvmins{T<:AbstractFloat}(::Type{Val{:^}}, N, ::Type{T}) =
    "@llvm.pow.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:abs}}, N, ::Type{T}) =
    "@llvm.fabs.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:muladd}}, N, ::Type{T}) =
    "@llvm.fmuladd.$(suffix(N,T))"
llvmins{T<:AbstractFloat}(::Type{Val{:sqrt}}, N, ::Type{T}) =
    "@llvm.sqrt.$(suffix(N,T))"

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
@generated function setindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}, x::Number)
    @assert isa(I, Integer)
    #TODO @boundscheck @assert 1 <= I <= N
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    decls = []
    instrs = []
    push!(instrs, "%resarr = insertvalue $atyp %0, $typ %1, $(I-1)")
    push!(instrs, "ret $atyp %resarr")
    quote
        # $(Expr(:meta, :inline))
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
        # $(Expr(:meta, :inline))
        let j = Int(i)
            #TODO @boundscheck @assert 1 <= j <= N
            Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
                NTuple{N,T}, Tuple{NTuple{N,T}, Int, T}, v.elts, j-1, T(x)))
        end
    end
end

Base.getindex{N,T,I}(v::Vec{N,T}, ::Type{Val{I}}) = v.elts[I]
Base.getindex{N,T}(v::Vec{N,T}, i::Integer) = v.elts[i]

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
    ins = llvmins(Val{Op}, N, R)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1)")
    else
        push!(instrs, "%res = $ins $vtypr zeroinitializer, %arg1")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        # $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}}, v1.elts))
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
    ins = llvmins(Val{Op}, N, R)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ1, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ2, "%1", "%arg2arr"))
    if ins[1] == '@'
        push!(decls, "declare $vtypr $ins($vtyp1, $vtyp2)")
        push!(instrs, "%res = call $vtypr $ins($vtyp1 %arg1, $vtyp2 %arg2)")
    else
        push!(instrs, "%res = $ins $vtypr %arg1, %arg2")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        # $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}, NTuple{N,T2}}, v1.elts, v2.elts))
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
    ins = llvmins(Val{Op}, N, R)
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
        push!(instrs, "%res = $ins $vtypr %arg1, %arg2, %arg3")
    end
    append!(instrs, vector2array("%resarr", N, typr, "%res"))
    push!(instrs, "ret $atypr %resarr")
    quote
        # $(Expr(:meta, :inline))
        Vec{N,R}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,R}, Tuple{NTuple{N,T1}, NTuple{N,T2}, NTuple{N,T3}},
            v1.elts, v2.elts, v3.elts))
    end
end

# Arithmetic functions

for op in (:+, :-, :abs, :sqrt)
    @eval begin
        Base.$op{N,T}(v1::Vec{N,T}) = llvmwrap(Val{$(QuoteNode(op))}, v1)
    end
end

for op in (:+, :-, :*, :/, :div, :rem, :^)
    @eval begin
        Base.$op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2)
    end
end

for op in (:muladd,)
    @eval begin
        Base.$op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}, v3::Vec{N,T}) =
            llvmwrap(Val{$(QuoteNode(op))}, v1, v2, v3)
    end
end

Base.ifelse{N,T}(v1::Vec{N,Bool}, v2::Vec{N,T}, v3::Vec{N,T}) =
    llvmwrap(Val{:ifelse}, v1, v2, v3, T)

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
    flags = ", align $align"
    push!(instrs, "%ptr = bitcast $typ* %0 to $vtyp*")
    push!(instrs, "%res = load $vtyp, $vtyp* %ptr$flags")
    append!(instrs, vector2array("%resarr", N, typ, "%res"))
    push!(instrs, "ret $atyp %resarr")
    quote
        # $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{Ptr{T}}, ptr))
    end
end

vloada{N,T}(::Type{Vec{N,T}}, ptr::Ptr{T}) = vload(Vec{N,T}, ptr, Val{true})

@inline function vload{N,T,Aligned}(::Type{Vec{N,T}}, arr::Vector{T},
        i::Integer, ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck @assert 1 <= i < length(arr) - N
    vload(Vec{N,T}, pointer(arr, i), Val{Aligned})
end
vloada{N,T}(::Type{Vec{N,T}}, arr::Vector{T}, i::Integer) =
    vload(Vec{N,T}, arr, i, Val{true})

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
    flags = ", align $align"
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    push!(instrs, "%ptr = bitcast $typ* %1 to $vtyp*")
    push!(instrs, "store $vtyp %arg1, $vtyp* %ptr$flags")
    push!(instrs, "ret void")
    quote
        # $(Expr(:meta, :inline))
        Void(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Void, Tuple{NTuple{N,T}, Ptr{T}}, v.elts, ptr))
    end
end

vstorea{N,T}(v::Vec{N,T}, ptr::Ptr{T}) = vstore(v, ptr, Val{true})

@inline function vstore{N,T,Aligned}(v::Vec{N,T}, arr::Vector{T}, i::Integer,
        ::Type{Val{Aligned}} = Val{false})
    #TODO @boundscheck @assert 1 <= i < length(arr) - N
    vstore(v, pointer(arr, i), Val{Aligned})
end
vstorea{N,T}(v::Vec{N,T}, arr::Vector{T}, i::Integer) =
    vstore(v, arr, i, Val{true})

# Conditionals

@generated function llvmwrapcond{Op,N,T}(::Type{Val{Op}}, v1::Vec{N,T},
        v2::Vec{N,T})
    @assert isa(Op, Symbol)
    btyp = llvmtype(Bool)
    vbtyp = "<$N x $btyp>"
    abtyp = "[$N x $btyp]"
    typ = llvmtype(T)
    atyp = "[$N x $typ]"
    vtyp = "<$N x $typ>"
    ins = llvmins(Val{Op}, N, T)
    decls = []
    instrs = []
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1arr"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2arr"))
    push!(instrs, "%cond = $ins $vtyp %arg1, %arg2")
    push!(instrs, "%res = zext <$N x i1> %cond to $vbtyp")
    append!(instrs, vector2array("%resarr", N, btyp, "%res"))
    push!(instrs, "ret $abtyp %resarr")
    quote
        # $(Expr(:meta, :inline))
        Vec{N,Bool}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,Bool}, Tuple{NTuple{N,T}, NTuple{N,T}},
            v1.elts, v2.elts))
    end
end

for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=))
    @eval begin
        Base.$op{N,T}(v1::Vec{N,T}, v2::Vec{N,T}) =
            llvmwrapcond(Val{$(QuoteNode(op))}, v1, v2)
    end
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
        # $(Expr(:meta, :inline))
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
        # $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            NTuple{N,T}, Tuple{NTuple{N,Bool}, NTuple{N,T}, NTuple{N,T}},
            v1.elts, v2.elts, v3.elts))
    end
end

end
