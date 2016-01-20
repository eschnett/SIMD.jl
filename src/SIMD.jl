module SIMD

export Vec
immutable Vec{N,T} <: DenseArray{T,1}
    elts::NTuple{N,T}
end

import Base: show
function show{N,T}(io::IO, v::Vec{N,T})
    print(io, "{")
    for i in 1:N
        i!=1 && print(io, ",")
        show(io, v.elts[i])
    end
    print(io, "}")
end

export vload, vloada
import Base: +, -, *, /, %
import Base: abs, sqrt
import Base: muladd

#=
+{N,T}(v::Vec{N,T}) = Vec{N,T}(ntuple(i->+v.elts[i], N))
-{N,T}(v::Vec{N,T}) = Vec{N,T}(ntuple(i->-v.elts[i], N))
=#

#=
for op in (:+, :-)
    @eval $op{N,T}(v::Vec{N,T}) = Vec{N,T}(ntuple(i->$op(v.elts[i]), N))
end
=#

#=
@generated function -{N,T}(v::Vec{N,T})
    args = []
    for i in 1:N
        push!(args, :(-(v.elts[$i])))
    end
    Expr(:tuple, args...)
end
=#

#=
function +{N,T}(v::Vec{N,T})
    Vec{N,T}(Base.llvmcall(
        """
        %2 = extractvalue [4 x double] %0, 0
        %3 = extractvalue [4 x double] %0, 1
        %4 = extractvalue [4 x double] %0, 2
        %5 = extractvalue [4 x double] %0, 3
        %6 = fadd fast double 0.0, %2
        %7 = fadd fast double 0.0, %3
        %8 = fadd fast double 0.0, %4
        %9 = fadd fast double 0.0, %5
        %10 = insertvalue [4 x double] undef, double %6, 0
        %11 = insertvalue [4 x double] %10, double %7, 1
        %12 = insertvalue [4 x double] %11, double %8, 2
        %13 = insertvalue [4 x double] %12, double %9, 3
        ret [4 x double] %13
        """,
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts))
end

function -{N,T}(v::Vec{N,T})
    Vec{N,T}(Base.llvmcall(
        """
        %2 = extractvalue [4 x double] %0, 0
        %3 = extractvalue [4 x double] %0, 1
        %4 = extractvalue [4 x double] %0, 2
        %5 = extractvalue [4 x double] %0, 3
        %6 = fsub fast double 0.0, %2
        %7 = fsub fast double 0.0, %3
        %8 = fsub fast double 0.0, %4
        %9 = fsub fast double 0.0, %5
        %10 = insertvalue [4 x double] undef, double %6, 0
        %11 = insertvalue [4 x double] %10, double %7, 1
        %12 = insertvalue [4 x double] %11, double %8, 2
        %13 = insertvalue [4 x double] %12, double %9, 3
        ret [4 x double] %13
        """,
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts))
end
=#

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

fastflags{T<:Signed}(::Type{T}) = "nsw"
fastflags{T<:Unsigned}(::Type{T}) = "nuw"
fastflags{T<:AbstractFloat}(::Type{T}) = "fast"

addins{T<:Integer}(::Type{T}) = "add"
addins{T<:AbstractFloat}(::Type{T}) = "fadd"

subins{T<:Integer}(::Type{T}) = "sub"
subins{T<:AbstractFloat}(::Type{T}) = "fsub"

mulins{T<:Integer}(::Type{T}) = "mul"
mulins{T<:AbstractFloat}(::Type{T}) = "fmul"

divins{T<:Signed}(::Type{T}) = "sdiv"
divins{T<:Unsigned}(::Type{T}) = "udiv"
divins{T<:AbstractFloat}(::Type{T}) = "fdiv"

remins{T<:Signed}(::Type{T}) = "srem"
remins{T<:Unsigned}(::Type{T}) = "urem"
remins{T<:AbstractFloat}(::Type{T}) = "frem"



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



@generated function vload{N,T}(::Type{Vec{N,T}}, arr::Vector{T}, i::Integer)
    instrs = []
    ins = "load"
    bytes = sizeof(T)   # This is overly optimistic
    flags = ", align $bytes"
    typ = llvmtype(T)
    push!(instrs, "%ptr = bitcast $typ* %0 to <$N x $typ>*")
    push!(instrs, "%res = $ins <$N x $typ>, <$N x $typ>* %ptr$flags")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    quote
        $(Expr(:meta, :inline))
        Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
            NTuple{N,T}, Tuple{Ptr{T}}, pointer(arr, i)))
    end
end

@generated function vloada{N,T}(::Type{Vec{N,T}}, arr::Vector{T}, i::Integer)
    instrs = []
    ins = "load"
    bytes = N * sizeof(T)
    flags = ", align $bytes"
    typ = llvmtype(T)
    push!(instrs, "%ptr = bitcast $typ* %0 to <$N x $typ>*")
    push!(instrs, "%res = $ins <$N x $typ>, <$N x $typ>* %ptr$flags")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{Ptr{T}}, pointer(arr, i))))
end

@generated function +{N,T}(v::Vec{N,T})
    instrs = []
    ins = addins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> zeroinitializer, %arg1")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts)))
end

@generated function -{N,T}(v::Vec{N,T})
    instrs = []
    ins = subins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> zeroinitializer, %arg1")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts)))
end

@generated function abs{N,T<:AbstractFloat}(v::Vec{N,T})
    decls = []
    instrs = []
    bits = 8*sizeof(T)
    ins = "@llvm.fabs.v$(N)f$bits"
    typ = llvmtype(T)
    push!(decls, "declare <$N x $typ> $ins(<$N x $typ>)")
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    push!(instrs, "%res = call <$N x $typ> $ins(<$N x $typ> %arg1)")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts)))
end

@generated function sqrt{N,T<:AbstractFloat}(v::Vec{N,T})
    decls = []
    instrs = []
    bits = 8*sizeof(T)
    ins = "@llvm.sqrt.v$(N)f$bits"
    typ = llvmtype(T)
    push!(decls, "declare <$N x $typ> $ins(<$N x $typ>)")
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    push!(instrs, "%res = call <$N x $typ> $ins(<$N x $typ> %arg1)")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
        NTuple{N,T}, Tuple{NTuple{N,T}}, v.elts)))
end

@generated function +{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
    instrs = []
    ins = addins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts)))
end

@generated function -{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
    instrs = []
    ins = subins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts)))
end

@generated function *{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
    instrs = []
    ins = mulins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts)))
end

@generated function /{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
    instrs = []
    ins = divins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts)))
end

@generated function %{N,T}(v1::Vec{N,T}, v2::Vec{N,T})
    instrs = []
    ins = remins(T)
    flags = fastflags(T)
    typ = llvmtype(T)
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    push!(instrs, "%res = $ins $flags <$N x $typ> %arg1, %arg2")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($(join(instrs, "\n")),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}}, v1.elts, v2.elts)))
end

function muladd{N,T<:Integer}(v1::Vec{N,T}, v2::Vec{N,T}, v3::Vec{N,T})
    v1*v2+v3
end

@generated function muladd{N,T<:AbstractFloat}(v1::Vec{N,T}, v2::Vec{N,T},
        v3::Vec{N,T})
    decls = []
    instrs = []
    bits = 8*sizeof(T)
    ins = "@llvm.fmuladd.v$(N)f$bits"
    typ = llvmtype(T)
    push!(decls,
        "declare <$N x $typ> $ins(<$N x $typ>, <$N x $typ>, <$N x $typ>)")
    append!(instrs, array2vector("%arg1", N, typ, "%0", "%arg1_array"))
    append!(instrs, array2vector("%arg2", N, typ, "%1", "%arg2_array"))
    append!(instrs, array2vector("%arg3", N, typ, "%2", "%arg3_array"))
    push!(instrs,
        "%res = call <$N x $typ> $ins(<$N x $typ> %arg1, " *
            "<$N x $typ> %arg2, <$N x $typ> %arg3)")
    append!(instrs, vector2array("%res_array", N, typ, "%res"))
    push!(instrs, "ret [$N x $typ] %res_array")
    :(Vec{N,T}(Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
        NTuple{N,T}, Tuple{NTuple{N,T}, NTuple{N,T}, NTuple{N,T}},
        v1.elts, v2.elts, v3.elts)))
end

end
