using SIMD
using Base.Test

typealias Float64x4 Vec(4,Float64)
arr = Array{Float64x4}()
code_native(+, (Float64x4,))
code_native(-, (Float64x4,))

code_native(+, (Float64x4, Float64x4))
code_native(-, (Float64x4, Float64x4))

code_native(muladd, (Float64x4, Float64x4, Float64x4))

f(x,y,z) = x+y-z
code_native(f, (Float64x4, Float64x4, Float64x4))

Float64x4((1,2,3,4)) |> info
@code_native Float64x4((1,2,3,4))
@code_native Float64x4((1.0,2.0,3.0,4.0))
# convert(Float64x4, (1,2,3,4))

@code_native Float64x4(1.0)

@code_native NTuple{4,Float64}(Float64x4(1.0))

@code_native setindex(Float64x4(0), 1, 1)

@code_native setindex(Float64x4(0), Val{1}, 1)
@code_native setindex(Float64x4(0), Val{2}, 1)
@code_native setindex(Float64x4(0), Val{3}, 1)
@code_native setindex(Float64x4(0), Val{4}, 1)

@code_native Float64x4((1,2,3,4))[1]

@code_native Float64x4((1,2,3,4))[Val{1}]
@code_native Float64x4((1,2,3,4))[Val{2}]
@code_native Float64x4((1,2,3,4))[Val{3}]
@code_native Float64x4((1,2,3,4))[Val{4}]

@code_native vload(Float64x4, Ptr{Float64}(0))
@code_native vloada(Float64x4, Ptr{Float64}(0))
@code_native vload(Float64x4, Vector{Float64}(0), 12)
@code_native vloada(Float64x4, Vector{Float64}(0), 12)

#=
info("vadd")

function vadd!{V<:SIMD.AbstractVec, T}(::Type{V}, xs::Vector{T}, ys::Vector{T})
    N = length(V)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        vstore(xs, xv, i)
    end
    sum(sv.elts)
end

@code_native vsum(Float64x4, Vector{Float64}(0))
=#

info("vsum")

function vsum{V<:SIMD.AbstractVec, T}(::Type{V}, xs::Vector{T})
    N = length(V)
    @assert V !== SIMD.AbstractVec{N,T}
    sv = V(0)
    @inbounds for i in 1:N:length(xs)
        xv = vload(V, xs, i)
        sv += xv
    end
    # sv[Val{1}] + sv[Val{2}] + sv[Val{3}] + sv[Val{4}]
    # sv[1] + sv[2] + sv[3] + sv[4]
    s = T(0)
    for i in 1:N
        s += sv[i]
    end
    s
end

info(vsum(Float64x4, [1.0,2.0,3.0,4.0]))
@code_native vsum(Float64x4, [1.0,2.0,3.0,4.0])
