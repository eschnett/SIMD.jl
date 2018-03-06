using SIMD
using Compat.Test
using Compat: @info

@info "Basic definitions"

# The vector we are testing. Ideally, we should be able to use any vector size
# anywhere, but LLVM codegen bugs prevent us from doing so -- thus we make this
# a parameter.
const nbytes = 32

const L8 = nbytes÷4
const L4 = nbytes÷8

const V8I32 = Vec{L8,Int32}
const V4F64 = Vec{L4,Float64}

@info "Type properties"

@test eltype(V8I32) === Int32
@test eltype(V4F64) === Float64
@test length(V8I32) == L8
@test length(V4F64) == L4
@test ndims(V8I32) == 1
@test ndims(V4F64) == 1
@test size(V8I32,1) == L8
@test size(V4F64,1) == L4
@test size(V8I32) == (L8,)
@test size(V4F64) == (L4,)

@info "Type conversion"

const v8i32 = ntuple(i->Int32(ifelse(isodd(i), i, -i)), L8)
const v4f64 = ntuple(i->Float64(ifelse(isodd(i), i, -i)), L4)

@test string(V8I32(v8i32)) == "Int32⟨" * string(v8i32)[2:end-1] * "⟩"
@test string(V4F64(v4f64)) == "Float64⟨" * string(v4f64)[2:end-1] * "⟩"

@test convert(V8I32, V8I32(v8i32)) === V8I32(v8i32)
@test convert(Vec{L8,Int64}, V8I32(v8i32)) ===
    Vec{L8, Int64}(convert(NTuple{L8,Int64}, v8i32))

@test NTuple{L8,Int32}(V8I32(v8i32)) === v8i32
@test NTuple{L4,Float64}(V4F64(v4f64)) === v4f64
@test Tuple(V8I32(v8i32)) === v8i32
@test Tuple(V4F64(v4f64)) === v4f64

@info "Element-wise access"

for i in 1:L8
    @test Tuple(setindex(V8I32(v8i32), 9.0, Val{i})) ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)
    @test Tuple(setindex(V8I32(v8i32), 9.0, i)) ===
        ntuple(j->Int32(ifelse(j==i, 9, v8i32[j])), L8)

    @test V8I32(v8i32)[Val{i}] === v8i32[i]
    @test V8I32(v8i32)[i] === v8i32[i]
end
@test_throws BoundsError setindex(V8I32(v8i32), 0, Val{0})
@test_throws BoundsError setindex(V8I32(v8i32), 0, Val{L8+1})
@test_throws BoundsError setindex(V8I32(v8i32), 0, 0)
@test_throws BoundsError setindex(V8I32(v8i32), 0, L8+1)
@test_throws BoundsError V8I32(v8i32)[Val{0}]
@test_throws BoundsError V8I32(v8i32)[Val{L8+1}]
@test_throws BoundsError V8I32(v8i32)[0]
@test_throws BoundsError V8I32(v8i32)[L8+1]

for i in 1:L4
    @test Tuple(setindex(V4F64(v4f64), 9, Val{i})) ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)
    @test Tuple(setindex(V4F64(v4f64), 9, i)) ===
        ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)

    @test V4F64(v4f64)[Val{i}] === v4f64[i]
    @test V4F64(v4f64)[i] === v4f64[i]
end

let
    v0 = zero(Vec{4, Float64})
    v1 = one(Vec{4, Float64})
    @test sum(v0*v0) == 0.0
    @test sum(v1*v1) == 4.0
end

@info "Integer arithmetic functions"

const v8i32b = map(x->Int32(x+1), v8i32)
const v8i32c = map(x->Int32(x*2), v8i32)

notbool(x) = !(x>=typeof(x)(0))
for op in (~, +, -, abs, notbool, sign, signbit)
    @test Tuple(op(V8I32(v8i32))) === map(op, v8i32)
end

for op in (
        +, -, *, ÷, %, ==, !=, <, <=, >, >=,
        copysign, div, flipsign, max, min, rem)
    @test Tuple(op(V8I32(v8i32), V8I32(v8i32b))) === map(op, v8i32, v8i32b)
end

ifelsebool(x,y,z) = ifelse(x>=typeof(x)(0),y,z)
for op in (ifelsebool, muladd)
    @test Tuple(op(V8I32(v8i32), V8I32(v8i32b), V8I32(v8i32c))) ===
        map(op, v8i32, v8i32b, v8i32c)
end

for op in (<<, >>, >>>)
    @test Tuple(op(V8I32(v8i32), Val{3})) === map(x->op(x,3), v8i32)
    @test Tuple(op(V8I32(v8i32), Val{-3})) === map(x->op(x,-3), v8i32)
    @test Tuple(op(V8I32(v8i32), 3)) === map(x->op(x,3), v8i32)
    @test Tuple(op(V8I32(v8i32), -3)) === map(x->op(x,-3), v8i32)
    @test Tuple(op(V8I32(v8i32), V8I32(v8i32))) === map(op, v8i32, v8i32)
end

@info "Floating point arithmetic functions"

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
    @test Tuple(op(V4F64(v4f64))) === map(op, v4f64)
end
function Base.isapprox(t1::Tuple,t2::Tuple)
    length(t1)==length(t2) &&
        all(Bool[isapprox(t1[i], t2[i]) for i in 1:length(t1)])
end
for op in (cos, exp, exp10, exp2, logabs, log10abs, log2abs, sin)
    rvec = Tuple(op(V4F64(v4f64)))
    rsca = map(op, v4f64)
    @test typeof(rvec) === typeof(rsca)
    @test isapprox(rvec, rsca)
end

@test isfinite(V4F64(0.0))[1]
@test isfinite(V4F64(-0.0))[1]
@test isfinite(V4F64(nextfloat(0.0)))[1]
@test isfinite(V4F64(-nextfloat(0.0)))[1]
@test isfinite(V4F64(1.0))[1]
@test isfinite(V4F64(-1.0))[1]
@test !isfinite(V4F64(Inf))[1]
@test !isfinite(V4F64(-Inf))[1]
@test !isfinite(V4F64(NaN))[1]

@test !isinf(V4F64(0.0))[1]
@test !isinf(V4F64(-0.0))[1]
@test !isinf(V4F64(nextfloat(0.0)))[1]
@test !isinf(V4F64(-nextfloat(0.0)))[1]
@test !isinf(V4F64(1.0))[1]
@test !isinf(V4F64(-1.0))[1]
@test isinf(V4F64(Inf))[1]
@test isinf(V4F64(-Inf))[1]
@test !isinf(V4F64(NaN))[1]

@test !isnan(V4F64(0.0))[1]
@test !isnan(V4F64(-0.0))[1]
@test !isnan(V4F64(nextfloat(0.0)))[1]
@test !isnan(V4F64(-nextfloat(0.0)))[1]
@test !isnan(V4F64(1.0))[1]
@test !isnan(V4F64(-1.0))[1]
@test !isnan(V4F64(Inf))[1]
@test !isnan(V4F64(-Inf))[1]
@test isnan(V4F64(NaN))[1]

@test !issubnormal(V4F64(0.0))[1]
@test !issubnormal(V4F64(-0.0))[1]
@test issubnormal(V4F64(nextfloat(0.0)))[1]
@test issubnormal(V4F64(-nextfloat(0.0)))[1]
@test !issubnormal(V4F64(1.0))[1]
@test !issubnormal(V4F64(-1.0))[1]
@test !issubnormal(V4F64(Inf))[1]
@test !issubnormal(V4F64(-Inf))[1]
@test !issubnormal(V4F64(NaN))[1]

@test !signbit(V4F64(0.0))[1]
@test signbit(V4F64(-0.0))[1]
@test !signbit(V4F64(nextfloat(0.0)))[1]
@test signbit(V4F64(-nextfloat(0.0)))[1]
@test !signbit(V4F64(1.0))[1]
@test signbit(V4F64(-1.0))[1]
@test !signbit(V4F64(Inf))[1]
@test signbit(V4F64(-Inf))[1]
@test !signbit(V4F64(NaN))[1]

for op in (
        +, -, *, /, %, ^, ==, !=, <, <=, >, >=,
        copysign, flipsign, max, min, rem)
    @test Tuple(op(V4F64(v4f64), V4F64(v4f64b))) === map(op, v4f64, v4f64b)
end

for op in (fma, ifelsebool, muladd)
    @test Tuple(op(V4F64(v4f64), V4F64(v4f64b), V4F64(v4f64c))) ===
        map(op, v4f64, v4f64b, v4f64c)
end

@info "Type promotion"

for op in (
        ==, !=, <, <=, >, >=,
        &, |, ⊻, +, -, *, copysign, div, flipsign, max, min, rem)
    @test op(42, V8I32(v8i32)) === op(V8I32(42), V8I32(v8i32))
    @test op(V8I32(v8i32), 42) === op(V8I32(v8i32), V8I32(42))
end
@test ifelse(signbit(V8I32(v8i32)), 42, V8I32(v8i32)) ===
    ifelse(signbit(V8I32(v8i32)), V8I32(42), V8I32(v8i32))
@test ifelse(signbit(V8I32(v8i32)), V8I32(v8i32), 42) ===
    ifelse(signbit(V8I32(v8i32)), V8I32(v8i32), V8I32(42))
for op in (muladd,)
    @test op(42, 42, V8I32(v8i32)) ===
        op(V8I32(42), V8I32(42), V8I32(v8i32))
    @test op(42, V8I32(v8i32), V8I32(v8i32)) ===
        op(V8I32(42), V8I32(v8i32), V8I32(v8i32))
    @test op(V8I32(v8i32), 42, V8I32(v8i32)) ===
        op(V8I32(v8i32), V8I32(42), V8I32(v8i32))
    @test op(V8I32(v8i32), V8I32(v8i32), 42) ===
        op(V8I32(v8i32), V8I32(v8i32), V8I32(42))
    @test op(42, V8I32(v8i32), 42) ===
        op(V8I32(42), V8I32(v8i32), V8I32(42))
    @test op(V8I32(v8i32), 42, 42) ===
        op(V8I32(v8i32), V8I32(42), V8I32(42))
end

for op in (
        ==, !=, <, <=, >, >=,
        +, -, *, /, ^, copysign, flipsign, max, min, rem)
    @test op(42, V4F64(v4f64)) === op(V4F64(42), V4F64(v4f64))
    @test op(V4F64(v4f64), 42) === op(V4F64(v4f64), V4F64(42))
end
@test ifelse(signbit(V4F64(v4f64)), 42, V4F64(v4f64)) ===
    ifelse(signbit(V4F64(v4f64)), V4F64(42), V4F64(v4f64))
@test ifelse(signbit(V4F64(v4f64)), V4F64(v4f64), 42) ===
    ifelse(signbit(V4F64(v4f64)), V4F64(v4f64), V4F64(42))
for op in (fma, muladd)
    @test op(42, 42, V4F64(v4f64)) ===
        op(V4F64(42), V4F64(42), V4F64(v4f64))
    @test op(42, V4F64(v4f64), V4F64(v4f64)) ===
        op(V4F64(42), V4F64(v4f64), V4F64(v4f64))
    @test op(V4F64(v4f64), 42, V4F64(v4f64)) ===
        op(V4F64(v4f64), V4F64(42), V4F64(v4f64))
    @test op(V4F64(v4f64), V4F64(v4f64), 42) ===
        op(V4F64(v4f64), V4F64(v4f64), V4F64(42))
    @test op(42, V4F64(v4f64), 42) ===
        op(V4F64(42), V4F64(v4f64), V4F64(42))
    @test op(V4F64(v4f64), 42, 42) ===
        op(V4F64(v4f64), V4F64(42), V4F64(42))
end

@info "Reduction operations"

for op in (maximum, minimum, sum, prod)
    @test op(V8I32(v8i32)) === op(v8i32)
end
@test all(V8I32(v8i32)) == reduce(&, v8i32)
@test any(V8I32(v8i32)) == reduce(|, v8i32)

for op in (maximum, minimum, sum, prod)
    @test op(V4F64(v4f64)) === op(v4f64)
end

@test sum(Vec{3,Float64}(1)) === 3.0
@test prod(Vec{5,Float64}(2)) === 32.0

@info "Load and store functions"

const arri32 = valloc(Int32, L8, 2*L8) do i i end
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

const arrf64 = valloc(Float64, L4, 4*L4) do i i end
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

@info "Real-world examples"

function vadd!(xs::AbstractArray{T,1}, ys::AbstractArray{T,1},
               ::Type{Vec{N,T}}) where {N,T}
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        yv = vload(Vec{N,T}, ys, i)
        xv += yv
        vstore(xv, xs, i)
    end
end

let xs = valloc(Float64, L4, 4*L4) do i i end,
    ys = valloc(Float64, L4, 4*L4) do i 1 end
    vadd!(xs, ys, V4F64)
    @test xs == Float64[i+1 for i in 1:(4*L4)]
    # @code_native vadd!(xs, ys, V4F64)
end

function vsum(xs::AbstractArray{T,1}, ::Type{Vec{N,T}}) where {N,T}
    @assert length(xs) % N == 0
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        xv = vload(Vec{N,T}, xs, i)
        sv += xv
    end
    sum(sv)
end

let xs = valloc(Float64, L4, 4*L4) do i i end
    s = vsum(xs, V4F64)
    @test s === (x->(x^2+x)/2)(Float64(4*L4))
    # @code_native vsum(xs, V4F64)
end

function vadd_masked!(xs::AbstractArray{T,1}, ys::AbstractArray{T,1},
                      ::Type{Vec{N,T}}) where {N, T}
    @assert length(ys) == length(xs)
    limit = length(xs) - (N-1)
    vlimit = Vec{N,Int}(let l=length(xs); (l:l+N-1...,) end)
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

let xs = valloc(Float64, 4, 13) do i i end,
    ys = valloc(Float64, 4, 13) do i 1 end
    vadd_masked!(xs, ys, V4F64)
    @test xs == Float64[i+1 for i in 1:13]
    # @code_native vadd!(xs, ys, V4F64)
end

function vsum_masked(xs::AbstractArray{T,1}, ::Type{Vec{N,T}}) where {N,T}
    vlimit = Vec{N,Int}(let l=length(xs); (l:l+N-1...,) end)
    sv = Vec{N,T}(0)
    @inbounds for i in 1:N:length(xs)
        mask = Vec{N,Int}(i) <= vlimit
        xv = vload(Vec{N,T}, xs, i, mask)
        sv += xv
    end
    sum(sv)
end

let xs = valloc(Float64, 4, 13) do i i end
    s = vsum_masked(xs, V4F64)
    @code_llvm vsum(xs, V4F64)
    @code_native vsum(xs, V4F64)
    @test s === sum(xs)
end

@info "Vector shuffles"

for T in (Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float32,Float64)
    a = Vec{4,T}((1,2,3,4))
    b = Vec{4,T}((5,6,7,8))
    @test shufflevector(a, b, Val{(2,3,4,5)}) === Vec{4,T}((3,4,5,6))
    @test shufflevector(a, b, Val{(1,7,5,5)}) === Vec{4,T}((2,8,6,6))
    @test shufflevector(a, b, Val{0:3}) === a
    @test shufflevector(a, b, Val{4:7}) === b
    @test shufflevector(a, Val{(1,0,2,3)}) === Vec{4,T}((2,1,3,4))
    @test shufflevector(a, b, Val{(0,1,4,5,2,3,6,7)}) === Vec{8,T}((1,2,5,6,3,4,7,8))
    @test shufflevector(shufflevector(a, b, Val{(6,:undef,0,:undef)}), Val{(0,2)}) === Vec{2,T}((7,1))
    @test isa(shufflevector(a, Val{(:undef,:undef,:undef,:undef)}), Vec{4,T})
    c = Vec{8,T}((1:8...,))
    d = Vec{8,T}((9:16...,))
    @test shufflevector(c, d, Val{(0,1,8,15)}) === Vec{4,T}((1,2,9,16))
    @test shufflevector(c, d, Val{1:2:15}) === Vec{8,T}((2:2:16...,))
end

let
    a = Vec{4,Bool}((true,false,true,false))
    b = Vec{4,Bool}((false,false,true,true))
    @test shufflevector(a, b, Val{(2,3,4,5)}) === Vec{4,Bool}((true,false,false,false))
end
