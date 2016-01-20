using SIMD
using Base.Test

function vsum{N,T}(::Type{Vec{N,T}}, xs::Vector{T})
    sv = Vec{N,T}((0,0,0,0))
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        sv += xv
    end
    sum(sv.elts)
end

typealias Float64x4 Vec{4,Float64}

info(vsum(Float64x4, [1.0, 2.0, 3.0, 4.0]))
@code_native vsum(Float64x4, [1.0, 2.0, 3.0, 4.0])
