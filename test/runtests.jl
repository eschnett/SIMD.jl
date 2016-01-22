using SIMD
using Base.Test

typealias V8I32 Vec{8,Int32}
typealias V4F64 Vec{4,Float64}

# Type properties

@test length(V8I32) == 8
@test length(V4F64) == 4
@test eltype(V8I32) === Int32
@test eltype(V4F64) === Float64

# Type conversion

const v8i32 = ntuple(i->Int32(ifelse(isodd(i), i, -i)), 8)
const v4f64 = ntuple(i->Float64(ifelse(isodd(i), i, -i)), 4)

@test V8I32(v8i32).elts === v8i32
@test V4F64(v4f64).elts === v4f64

@test V8I32(9).elts === ntuple(i->Int32(9), 8)
@test V4F64(9).elts === ntuple(i->9.0, 4)
@test V8I32(ntuple(i->Float32(v8i32[i]), 8)).elts === v8i32
@test V4F64(ntuple(i->Int64(v4f64[i]), 4)).elts === v4f64

@test NTuple{8,Int32}(V8I32(v8i32)) === v8i32
@test NTuple{4,Float64}(V4F64(v4f64)) === v4f64

# Element-wise access

for i in 1:8
    @test setindex(V8I32(v8i32), Val{i}, 9.0).elts ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), 8)
    @test setindex(V8I32(v8i32), i, 9.0).elts ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), 8)

    @test V8I32(v8i32)[Val{i}] === v8i32[i]
    @test V8I32(v8i32)[i] === v8i32[i]
end
@test_throws BoundsError setindex(V8I32(v8i32), Val{0}, 0)
@test_throws BoundsError setindex(V8I32(v8i32), Val{9}, 0)
@test_throws BoundsError setindex(V8I32(v8i32), 0, 0)
@test_throws BoundsError setindex(V8I32(v8i32), 9, 0)
@test_throws BoundsError V8I32(v8i32)[Val{0}]
@test_throws BoundsError V8I32(v8i32)[Val{9}]
@test_throws BoundsError V8I32(v8i32)[0]
@test_throws BoundsError V8I32(v8i32)[9]

for i in 1:4
    @test setindex(V4F64(v4f64), Val{i}, 9).elts ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), 4)
    @test setindex(V4F64(v4f64), i, 9).elts ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), 4)

    @test V4F64(v4f64)[Val{i}] === v4f64[i]
    @test V4F64(v4f64)[i] === v4f64[i]
end

# Arithmetic functions and conditionals

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
@test ^(V4F64(v4f64), Vec{4,Int64}(v4f64b)).elts === map(^, v4f64, v4f64b)
for op in (muladd, (x,y,z)->ifelse(x==abs(x),y,z))
    @test op(V4F64(v4f64), V4F64(v4f64b), V4F64(v4f64c)).elts ===
        map(op, v4f64, v4f64b, v4f64c)
end
