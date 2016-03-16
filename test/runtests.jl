using SIMD
using Base.Test

macro showtest(expr)
    if length(expr.args) == 3
        lhs = expr.args[1]
        rhs = expr.args[3]
    else
        lhs = expr.args[1]
        rhs = nothing
    end
    esc(quote
        # @show $lhs
        # @show $rhs
        @test $expr
        # println()
    end)
end

info("Basic definitions")

# The vector we are testing. Ideally, we should be able to use any vector size
# anywhere, but LLVM codegen bugs prevent us from doing so -- thus we make this
# a parameter.
const nbytes = 32

const L8 = nbytes÷4
const L4 = nbytes÷8

typealias V8I32 Vec{L8,Int32}
typealias V4F64 Vec{L4,Float64}

info("Type properties")

@showtest eltype(V8I32) === Int32
@showtest eltype(V4F64) === Float64
@showtest length(V8I32) == L8
@showtest length(V4F64) == L4
@showtest ndims(V8I32) == 1
@showtest ndims(V4F64) == 1
@showtest size(V8I32,1) == L8
@showtest size(V4F64,1) == L4
@showtest size(V8I32) == (L8,)
@showtest size(V4F64) == (L4,)

info("Type conversion")

const v8i32 = ntuple(i->Int32(ifelse(isodd(i), i, -i)), L8)
const v4f64 = ntuple(i->Float64(ifelse(isodd(i), i, -i)), L4)

@showtest string(V8I32(v8i32)) == "Int32<" * string(v8i32)[2:end-1] * ">"
@showtest string(V4F64(v4f64)) == "Float64<" * string(v4f64)[2:end-1] * ">"

@showtest NTuple{L8,Int32}(V8I32(v8i32)) === v8i32
@showtest NTuple{L4,Float64}(V4F64(v4f64)) === v4f64
@showtest Tuple(V8I32(v8i32)) === v8i32
@showtest Tuple(V4F64(v4f64)) === v4f64

info("Element-wise access")

for i in 1:L8
    @showtest Tuple(setindex(V8I32(v8i32), Val{i}, 9.0)) ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)
    @showtest Tuple(setindex(V8I32(v8i32), i, 9.0)) ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)

    @showtest V8I32(v8i32)[Val{i}] === v8i32[i]
    @showtest V8I32(v8i32)[i] === v8i32[i]
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
    @showtest Tuple(setindex(V4F64(v4f64), Val{i}, 9)) ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)
    @showtest Tuple(setindex(V4F64(v4f64), i, 9)) ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)

    @showtest V4F64(v4f64)[Val{i}] === v4f64[i]
    @showtest V4F64(v4f64)[i] === v4f64[i]
end

let
    v0 = zero(Vec{4, Float64})
    v1 = one(Vec{4, Float64})
    @test sum(v0*v0) == 0.0
    @test sum(v1*v1) == 4.0
end

info("Integer arithmetic functions")

const v8i32b = map(x->Int32(x+1), v8i32)
const v8i32c = map(x->Int32(x*2), v8i32)

notbool(x) = !(x>=typeof(x)(0))
for op in (~, +, -, abs, notbool, sign, signbit)
    # @show op
    @showtest Tuple(op(V8I32(v8i32))) === map(op, v8i32)
end

for op in (
        +, -, *, ÷, %, ==, !=, <, <=, >, >=,
        copysign, div, flipsign, max, min, rem)
    # @show op
    @showtest Tuple(op(V8I32(v8i32), V8I32(v8i32b))) === map(op, v8i32, v8i32b)
end

ifelsebool(x,y,z) = ifelse(x>=typeof(x)(0),y,z)
for op in (ifelsebool, muladd)
    # @show op
    @showtest Tuple(op(V8I32(v8i32), V8I32(v8i32b), V8I32(v8i32c))) ===
        map(op, v8i32, v8i32b, v8i32c)
end

for op in (<<, >>, >>>)
    # @show op
    @showtest Tuple(op(V8I32(v8i32), Val{3})) === map(x->op(x,3), v8i32)
    @showtest Tuple(op(V8I32(v8i32), 3)) === map(x->op(x,3), v8i32)
    @showtest Tuple(op(V8I32(v8i32), V8I32(v8i32))) === map(op, v8i32, v8i32)
end

info("Floating point arithmetic functions")

const v4f64b = map(x->Float64(x+1), v4f64)
const v4f64c = map(x->Float64(x*2), v4f64)

logabs(x) = log(abs(x))
log10abs(x) = log10(abs(x))
log2abs(x) = log2(abs(x))
powi4(x) = x^4
sqrtabs(x) = sqrt(abs(x))
for op in (
        +, -,
        abs, ceil, inv, isfinite, isinf, isnan, issubnormal, floor, powi4,
        round, sign, signbit, sqrtabs, trunc)
    # @show op
    @showtest Tuple(op(V4F64(v4f64))) === map(op, v4f64)
end
function Base.isapprox(t1::Tuple,t2::Tuple)
    length(t1)==length(t2) &&
        all(Bool[isapprox(t1[i], t2[i]) for i in 1:length(t1)])
end
for op in (cos, exp, exp10, exp2, logabs, log10abs, log2abs, sin)
    # @show op
    rvec = Tuple(op(V4F64(v4f64)))
    rsca = map(op, v4f64)
    @showtest typeof(rvec) === typeof(rsca)
    @showtest isapprox(rvec, rsca)
end

@showtest isfinite(V4F64(0.0))[1]
@showtest isfinite(V4F64(-0.0))[1]
@showtest isfinite(V4F64(nextfloat(0.0)))[1]
@showtest isfinite(V4F64(-nextfloat(0.0)))[1]
@showtest isfinite(V4F64(1.0))[1]
@showtest isfinite(V4F64(-1.0))[1]
@showtest !isfinite(V4F64(Inf))[1]
@showtest !isfinite(V4F64(-Inf))[1]
@showtest !isfinite(V4F64(NaN))[1]

@showtest !isinf(V4F64(0.0))[1]
@showtest !isinf(V4F64(-0.0))[1]
@showtest !isinf(V4F64(nextfloat(0.0)))[1]
@showtest !isinf(V4F64(-nextfloat(0.0)))[1]
@showtest !isinf(V4F64(1.0))[1]
@showtest !isinf(V4F64(-1.0))[1]
@showtest isinf(V4F64(Inf))[1]
@showtest isinf(V4F64(-Inf))[1]
@showtest !isinf(V4F64(NaN))[1]

@showtest !isnan(V4F64(0.0))[1]
@showtest !isnan(V4F64(-0.0))[1]
@showtest !isnan(V4F64(nextfloat(0.0)))[1]
@showtest !isnan(V4F64(-nextfloat(0.0)))[1]
@showtest !isnan(V4F64(1.0))[1]
@showtest !isnan(V4F64(-1.0))[1]
@showtest !isnan(V4F64(Inf))[1]
@showtest !isnan(V4F64(-Inf))[1]
@showtest isnan(V4F64(NaN))[1]

@showtest !issubnormal(V4F64(0.0))[1]
@showtest !issubnormal(V4F64(-0.0))[1]
@showtest issubnormal(V4F64(nextfloat(0.0)))[1]
@showtest issubnormal(V4F64(-nextfloat(0.0)))[1]
@showtest !issubnormal(V4F64(1.0))[1]
@showtest !issubnormal(V4F64(-1.0))[1]
@showtest !issubnormal(V4F64(Inf))[1]
@showtest !issubnormal(V4F64(-Inf))[1]
@showtest !issubnormal(V4F64(NaN))[1]

@showtest !signbit(V4F64(0.0))[1]
@showtest signbit(V4F64(-0.0))[1]
@showtest !signbit(V4F64(nextfloat(0.0)))[1]
@showtest signbit(V4F64(-nextfloat(0.0)))[1]
@showtest !signbit(V4F64(1.0))[1]
@showtest signbit(V4F64(-1.0))[1]
@showtest !signbit(V4F64(Inf))[1]
@showtest signbit(V4F64(-Inf))[1]
@showtest !signbit(V4F64(NaN))[1]

for op in (
        +, -, *, /, %, ^, ==, !=, <, <=, >, >=,
        copysign, flipsign, max, min, rem)
    # @show op
    @showtest Tuple(op(V4F64(v4f64), V4F64(v4f64b))) === map(op, v4f64, v4f64b)
end

for op in (fma, ifelsebool, muladd)
    # @show op
    @showtest Tuple(op(V4F64(v4f64), V4F64(v4f64b), V4F64(v4f64c))) ===
        map(op, v4f64, v4f64b, v4f64c)
end

info("Type promotion")

for op in (
        ==, !=, <, <=, >, >=,
        &, |, $, +, -, *, copysign, div, flipsign, max, min, rem)
    # @show op
    @showtest op(42, V8I32(v8i32)) === op(V8I32(42), V8I32(v8i32))
    @showtest op(V8I32(v8i32), 42) === op(V8I32(v8i32), V8I32(42))
end
@showtest ifelse(signbit(V8I32(v8i32)), 42, V8I32(v8i32)) ===
    ifelse(signbit(V8I32(v8i32)), V8I32(42), V8I32(v8i32))
@showtest ifelse(signbit(V8I32(v8i32)), V8I32(v8i32), 42) ===
    ifelse(signbit(V8I32(v8i32)), V8I32(v8i32), V8I32(42))
for op in (muladd,)
    @showtest op(42, 42, V8I32(v8i32)) ===
        op(V8I32(42), V8I32(42), V8I32(v8i32))
    @showtest op(42, V8I32(v8i32), V8I32(v8i32)) ===
        op(V8I32(42), V8I32(v8i32), V8I32(v8i32))
    @showtest op(V8I32(v8i32), 42, V8I32(v8i32)) ===
        op(V8I32(v8i32), V8I32(42), V8I32(v8i32))
    @showtest op(V8I32(v8i32), V8I32(v8i32), 42) ===
        op(V8I32(v8i32), V8I32(v8i32), V8I32(42))
    @showtest op(42, V8I32(v8i32), 42) ===
        op(V8I32(42), V8I32(v8i32), V8I32(42))
    @showtest op(V8I32(v8i32), 42, 42) ===
        op(V8I32(v8i32), V8I32(42), V8I32(42))
end

for op in (
        ==, !=, <, <=, >, >=,
        +, -, *, /, ^, copysign, flipsign, max, min, rem)
    @showtest op(42, V4F64(v4f64)) === op(V4F64(42), V4F64(v4f64))
    @showtest op(V4F64(v4f64), 42) === op(V4F64(v4f64), V4F64(42))
end
@showtest ifelse(signbit(V4F64(v4f64)), 42, V4F64(v4f64)) ===
    ifelse(signbit(V4F64(v4f64)), V4F64(42), V4F64(v4f64))
@showtest ifelse(signbit(V4F64(v4f64)), V4F64(v4f64), 42) ===
    ifelse(signbit(V4F64(v4f64)), V4F64(v4f64), V4F64(42))
for op in (fma, muladd)
    @showtest op(42, 42, V4F64(v4f64)) ===
        op(V4F64(42), V4F64(42), V4F64(v4f64))
    @showtest op(42, V4F64(v4f64), V4F64(v4f64)) ===
        op(V4F64(42), V4F64(v4f64), V4F64(v4f64))
    @showtest op(V4F64(v4f64), 42, V4F64(v4f64)) ===
        op(V4F64(v4f64), V4F64(42), V4F64(v4f64))
    @showtest op(V4F64(v4f64), V4F64(v4f64), 42) ===
        op(V4F64(v4f64), V4F64(v4f64), V4F64(42))
    @showtest op(42, V4F64(v4f64), 42) ===
        op(V4F64(42), V4F64(v4f64), V4F64(42))
    @showtest op(V4F64(v4f64), 42, 42) ===
        op(V4F64(v4f64), V4F64(42), V4F64(42))
end

info("Reduction operations")

for op in (maximum, minimum, sum, prod)
    # @show op
    @showtest op(V8I32(v8i32)) === op(v8i32)
end
@showtest all(V8I32(v8i32)) == reduce(&, v8i32)
@showtest any(V8I32(v8i32)) == reduce(|, v8i32)

for op in (maximum, minimum, sum, prod)
    # @show op
    @showtest op(V4F64(v4f64)) === op(v4f64)
end

# TODO: This segfaults
# @showtest sum(Vec{3,Float64}(1)) === 3.0
@showtest prod(Vec{5,Float64}(2)) === 32.0

info("Load and store functions")

const arri32 = Int32[i for i in 1:(2*L8)]
for i in 1:length(arri32)-(L8-1)
    @showtest vload(V8I32, arri32, i) === V8I32(ntuple(j->i+j-1, L8))
end
for i in 1:L8:length(arri32)-(L8-1)
    @showtest vloada(V8I32, arri32, i) === V8I32(ntuple(j->i+j-1, L8))
end
vstorea(V8I32(0), arri32, 1)
vstore(V8I32(1), arri32, 2)
for i in 1:length(arri32)
    @showtest arri32[i] == if i==1 0 elseif i<=(L8+1) 1 else i end
end

const arrf64 = Float64[i for i in 1:(4*L4)]
for i in 1:length(arrf64)-(L4-1)
    @showtest vload(V4F64, arrf64, i) === V4F64(ntuple(j->i+j-1, L4))
end
for i in 1:4:length(arrf64)-(L4-1)
    @showtest vloada(V4F64, arrf64, i) === V4F64(ntuple(j->i+j-1, L4))
end
vstorea(V4F64(0), arrf64, 1)
vstore(V4F64(1), arrf64, 2)
for i in 1:length(arrf64)
    @showtest arrf64[i] == if i==1 0 elseif i<=(L4+1) 1 else i end
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

let xs = Float64[i for i in 1:(4*L4)],
    ys = Float64[1 for i in 1:(4*L4)]
    vadd!(xs, ys, V4F64)
    @showtest xs == Float64[i+1 for i in 1:(4*L4)]
    # @code_native vadd!(xs, ys, V4F64)
end

function vsum{N,T}(xs::Vector{T}, ::Type{Vec{N,T}})
    @assert length(xs) % N == 0
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        sv += xv
    end
    sum(sv)
end

let xs = Float64[i for i in 1:(4*L4)]
    s = vsum(xs, V4F64)
    @showtest s === (x->(x^2+x)/2)(Float64(4*L4))
    # @code_native vsum(xs, V4F64)
end

function vadd_masked!{N,T}(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}})
    @assert length(ys) == length(xs)
    limit = length(xs) - (N-1)
    vlimit = Vec{N,Int}(let l=length(xs); (l:l+N-1...) end)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        if i <= limit
            vstore(xv, xs, i)
        else
            mask = Vec{N,Int}(i) <= vlimit
            vstore(xv, xs, i, mask)
        end
    end
end

let xs = Float64[i for i in 1:13],
    ys = Float64[1 for i in 1:13]
    vadd_masked!(xs, ys, V4F64)
    @showtest xs == Float64[i+1 for i in 1:13]
    # @code_native vadd!(xs, ys, V4F64)
end

function vsum_masked{N,T}(xs::Vector{T}, ::Type{Vec{N,T}})
    vlimit = Vec{N,Int}(let l=length(xs); (l:l+N-1...) end)
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        mask = Vec{N,Int}(i) <= vlimit
        xv = vload(Vec{N,T}, xs, i, mask)
        sv += xv
    end
    sum(sv)
end

let xs = Float64[i for i in 1:13]
    s = vsum_masked(xs, V4F64)
    @showtest s === sum(xs)
    # @code_native vsum(xs, V4F64)
end
