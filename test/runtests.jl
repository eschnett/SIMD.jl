using SIMD
using Base.Test

@code_native Vec{4,Float64}((1.0,2.0,3.0,4.0))
@code_native Vec{4,Float64}((1,2,3,4))
@code_native Vec{4,Float64}(0)

v = Vec{4,Float64}(0)
@code_native NTuple{4,Float64}(v)

@code_native setindex(Vec{4,Float64}(0), Val{1}, 1)
@code_native setindex(Vec{4,Float64}(0), 1, 1)

@code_native Vec{4,Float64}(0)[Val{1}]
@code_native Vec{4,Float64}(0)[1]

@code_native sqrt(Vec{4,Float64}(1))
@code_native Vec{4,Float64}(1) / Vec{4,Float64}(2)
@code_native muladd(Vec{4,Float64}(1), Vec{4,Float64}(2), Vec{4,Float64}(3))

arr = Float64[1:10;]
@code_native vload(Vec{4,Float64}, pointer(arr, 1))
@code_native vload(Vec{4,Float64}, arr, 1)

@code_native vstore(Vec{4,Float64}(0), pointer(arr, 1))
@code_native vstore(Vec{4,Float64}(0), arr, 1)

info("vadd!")

function vadd!{N,T}(::Type{Vec{N,T}}, xs::Vector{T}, ys::Vector{T})
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        zv = xv + yv
        vstore(zv, xs, i)
    end
end

@code_native vadd!(Vec{4,Float64}, arr, arr)

info("vsum")

function vsum{N,T}(::Type{Vec{N,T}}, xs::Vector{T})
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
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

@code_native vsum(Vec{4,Float64}, arr)

@code_native Vec{4,Bool}(false)
@code_native Vec{4,Int8}(false)
@code_native Vec{4,Int64}(false)

@code_native Vec{4,Float64}(1) == Vec{4,Float64}(0)
@code_native ifelse(Vec{4,Bool}(false), Vec{4,Float64}(1), Vec{4,Float64}(0))

f(x,y,a,b) = ifelse(x==y,a,b)
code_native(f, (Vec{4,Float64}, Vec{4,Float64}, Vec{4,Float64}, Vec{4,Float64}))
code_native(f, (Float64, Float64, Vec{4,Float64}, Vec{4,Float64}))
