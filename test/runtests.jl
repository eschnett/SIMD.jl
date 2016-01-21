using SIMD
using Base.Test

#=
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
=#

#=
typealias Float64x4 Vec{4,Float64}

code_llvm(+, (Float64x4,))
code_native(+, (Float64x4,))
=#


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
