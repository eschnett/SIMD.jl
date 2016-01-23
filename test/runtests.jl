using SIMD
using Base.Test

info("Basic definitions")

# The vector we are testing. Ideally, we should be able to use any vector size
# anywhere, but LLVM codegen bugs prevent us from doing so -- thus we make this
# a parameter.
const nbytes = 16   # 32

const L8 = nbytes÷4
const L4 = nbytes÷8

typealias V8I32 Vec{L8,Int32}
typealias V4F64 Vec{L4,Float64}

info("Type properties")

@test length(V8I32) == L8
@test length(V4F64) == L4
@test eltype(V8I32) === Int32
@test eltype(V4F64) === Float64

info("Type conversion")

const v8i32 = ntuple(i->Int32(ifelse(isodd(i), i, -i)), L8)
const v4f64 = ntuple(i->Float64(ifelse(isodd(i), i, -i)), L4)

@test V8I32(v8i32).elts === v8i32
@test V4F64(v4f64).elts === v4f64

@test V8I32(9).elts === ntuple(i->Int32(9), L8)
@test V4F64(9).elts === ntuple(i->9.0, L4)
@test V8I32(ntuple(i->Float32(v8i32[i]), L8)).elts === v8i32
@test V4F64(ntuple(i->Int64(v4f64[i]), L4)).elts === v4f64

@test NTuple{L8,Int32}(V8I32(v8i32)) === v8i32
@test NTuple{L4,Float64}(V4F64(v4f64)) === v4f64

info("Element-wise access")

for i in 1:L8
    @test setindex(V8I32(v8i32), Val{i}, 9.0).elts ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)
    @test setindex(V8I32(v8i32), i, 9.0).elts ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)

    @test V8I32(v8i32)[Val{i}] === v8i32[i]
    @test V8I32(v8i32)[i] === v8i32[i]
end
@test_throws BoundsError setindex(V8I32(v8i32), Val{0}, 0)
@test_throws BoundsError setindex(V8I32(v8i32), Val{L8+1}, 0)
@test_throws BoundsError setindex(V8I32(v8i32), 0, 0)
@test_throws BoundsError setindex(V8I32(v8i32), L8+1, 0)
@test_throws BoundsError V8I32(v8i32)[Val{0}]
@test_throws BoundsError V8I32(v8i32)[Val{L8+1}]
@test_throws BoundsError V8I32(v8i32)[0]
@test_throws BoundsError V8I32(v8i32)[L8+1]

for i in 1:L4
    @test setindex(V4F64(v4f64), Val{i}, 9).elts ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)
    @test setindex(V4F64(v4f64), i, 9).elts ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)

    @test V4F64(v4f64)[Val{i}] === v4f64[i]
    @test V4F64(v4f64)[i] === v4f64[i]
end

info("Arithmetic functions and conditionals")

const v8i32b = map(x->Int32(x+1), v8i32)
const v8i32c = map(x->Int32(x*2), v8i32)
for op in (~, +, -, abs)
    @test op(V8I32(v8i32)).elts === map(op, v8i32)
end
for op in (+, -, *, ÷, %, <<, >>, >>>)
    @test op(V8I32(v8i32), V8I32(v8i32b)).elts === map(op, v8i32, v8i32b)
end
for op in (muladd, (x,y,z)->ifelse(x==abs(x),y,z))
    @test op(V8I32(v8i32), V8I32(v8i32b), V8I32(v8i32c)).elts ===
        map(op, v8i32, v8i32b, v8i32c)
end

const v4f64b = map(x->Float64(x+1), v4f64)
const v4f64c = map(x->Float64(x*2), v4f64)
for op in (+, -, abs, sin, x->sqrt(abs(x)))
    @test op(V4F64(v4f64)).elts === map(op, v4f64)
end
for op in (+, -, *, /, %, ^, ==, !=, <, <=, >, >=)
    @test op(V4F64(v4f64), V4F64(v4f64b)).elts === map(op, v4f64, v4f64b)
end
@test ^(V4F64(v4f64), Vec{L4,Int64}(v4f64b)).elts === map(^, v4f64, v4f64b)
for op in (muladd, (x,y,z)->ifelse(x==abs(x),y,z))
    @test op(V4F64(v4f64), V4F64(v4f64b), V4F64(v4f64c)).elts ===
        map(op, v4f64, v4f64b, v4f64c)
end

info("Load and store functions")

const arri32 = Int32[i for i in 1:(2*L8)]
for i in 1:length(arri32)-(L8-1)
    @test vload(V8I32, arri32, i) === V8I32(ntuple(j->i+j-1, L8))
end
for i in 1:L8:length(arri32)-(L8-1)
    @test vloada(V8I32, arri32, i) === V8I32(ntuple(j->i+j-1, L8))
end
vstorea(V8I32(0), arri32, 1)
vstore(V8I32(1), arri32, 2)
for i in 1:length(arri32)
    @test arri32[i] == if i==1 0 elseif i<=(L8+1) 1 else i end
end

const arrf64 = Float64[i for i in 1:(4*L4)]
for i in 1:length(arrf64)-(L4-1)
    @test vload(V4F64, arrf64, i) === V4F64(ntuple(j->i+j-1, L4))
end
for i in 1:4:length(arrf64)-(L4-1)
    @test vloada(V4F64, arrf64, i) === V4F64(ntuple(j->i+j-1, L4))
end
vstorea(V4F64(0), arrf64, 1)
vstore(V4F64(1), arrf64, 2)
for i in 1:length(arrf64)
    @test arrf64[i] == if i==1 0 elseif i<=(L4+1) 1 else i end
end

info("Real-world examples")

function vadd!{N,T}(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}})
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        vstore(xv, xs, i)
    end
end

let xs = Float64[i for i in 1:(4*L4)];
    ys = Float64[1 for i in 1:(4*L4)]
    vadd!(xs, ys, V4F64)
    @test xs == Float64[i+1 for i in 1:(4*L4)]
    @code_native vadd!(xs, ys, V4F64)
end

function vsum{N,T}(xs::Vector{T}, ::Type{Vec{N,T}})
    @assert length(xs) % N == 0
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        sv += xv
    end
    s = T(0)
    for i in 1:N s+=sv[i] end
    s
end

let xs = Float64[i for i in 1:(4*L4)]
    s = vsum(xs, V4F64)
    @test s === (x->(x^2+x)/2)(Float64(4*L4))
    @code_native vsum(xs, V4F64)
end
