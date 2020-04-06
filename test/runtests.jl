using SIMD
using Test, InteractiveUtils

using Base: setindex

"""
    llvm_ir(f, args) :: String

Get LLVM IR of `f(args...)` as a string.
"""
llvm_ir(f, args) = sprint(code_llvm, f, Base.typesof(args...))

#@testset "SIMD" begin
    # The vector we are testing.
    global const nbytes = 32

    global const L8 = nbytes÷4
    global const L4 = nbytes÷8

    global const V8I32 = Vec{L8,Int32}
    global const V8I64 = Vec{L8,Int64}
    global const V4F64 = Vec{L4,Float64}

    global const v8i32 = ntuple(i->Int32(ifelse(isodd(i), i, -i)), L8)
    global const v8i64 = ntuple(i->Int64(ifelse(isodd(i), i, -i)), L8)
    global const v4f64 = ntuple(i->Float64(ifelse(isodd(i), i, -i)), L4)

    is_checking_bounds = Core.Compiler.inbounds_option() == :on

    @testset "Type properties" begin
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
    end

    @testset "Type conversion" begin
        @test string(V8I32(v8i32)) == "<8 x Int32>[" * string(v8i32)[2:end-1] * "]"
        @test string(V4F64(v4f64)) == "<4 x Float64>[" * string(v4f64)[2:end-1] * "]"

        @test convert(V8I32, V8I32(v8i32)) === V8I32(v8i32)
        @test convert(Vec{L8,Int64}, V8I32(v8i32)) ===
            Vec{L8, Int64}(convert(NTuple{L8,Int64}, v8i32))

        @test NTuple{L8,Int32}(V8I32(v8i32)) === Tuple(v8i32)
        @test NTuple{L4,Float64}(V4F64(v4f64)) === Tuple(v4f64)
        @test Tuple(V8I32(v8i32)) === Tuple(v8i32)
        @test Tuple(V4F64(v4f64)) === Tuple(v4f64)
    end

    @testset "Conversion and reinterpretation" begin
        v = V8I32(v8i32)
        V4I64 = reinterpret(Vec{4, Int64}, v)
        @test sum(count_ones(v)) == sum(count_ones(V4I64))
        @test sum(count_zeros(v)) == sum(count_zeros(V4I64))
        x = Int64(123456789)
        @test reinterpret(Int64, reinterpret(Vec{4, Int16}, x)) == x

        @test all(Tuple(convert(Vec{8, Float64}, v)) .== Tuple(v))
    end

    @testset "Element-wise access" begin

        for i in 1:L8
            @test V8I32(v8i32)[i] === v8i32[i]
        end

        @test_throws BoundsError setindex(V8I32(v8i32), 0, 0)
        @test_throws BoundsError setindex(V8I32(v8i32), 0, L8+1)
        @test_throws BoundsError V8I32(v8i32)[0]
        @test_throws BoundsError V8I32(v8i32)[L8+1]

        for i in 1:L4
            @test Tuple(setindex(V4F64(v4f64), 9, i)) ===
                ntuple(j->Float64(ifelse(j==i, 9.0, v4f64[j])), L4)

            @test V4F64(v4f64)[i] === v4f64[i]
        end

        let
            v0 = zero(Vec{4, Float64})
            v1 = one(Vec{4, Float64})
            @test sum(v0*v0) == 0.0
            @test sum(v1*v1) == 4.0
        end
    end

    @testset "Integer arithmetic functions" begin

        global const v8i32b = map(x->Int32(x+1), v8i32)
        global const v8i32c = map(x->Int32(x*2), v8i32)

        notbool(x) = !(x>=typeof(x)(0))
        for op in (~, +, -, abs, notbool, sign, signbit, count_ones, count_zeros,
                   leading_ones, leading_zeros, trailing_ones, trailing_zeros)
            @test Tuple(op(V8I32(v8i32))) == map(op, v8i32)
        end

        for op in (
                +, -, *, ÷, %, ==, !=, <, <=, >, >=,
                copysign, div, flipsign, max, min, rem)
            @test Tuple(op(V8I32(v8i32), V8I32(v8i32b))) === map(op, v8i32, v8i32b)
        end

        global vifelsebool(x,y,z) = vifelse(x>=typeof(x)(0),y,z)
        for op in (vifelsebool, muladd)
            @test Tuple(op(V8I32(v8i32), V8I32(v8i32b), V8I32(v8i32c))) ===
                map(op, v8i32, v8i32b, v8i32c)
        end

        for op in (<<, >>, >>>)
            for v in (V8I32(v8i32), V8I64(v8i64))
                for z in (3, UInt(3), Int32(10000), UInt8(4))
                    @test Tuple(op(v, z)) === map(x->op(x,z), Tuple(v))
                    @test Tuple(op(v, -z)) === map(x->op(x,-z), Tuple(v))
                    @test Tuple(op(v, v)) === map(op, Tuple(v), Tuple(v))
                end
            end
        end

        @test Tuple(V8I32(v8i32)^0) === v8i32.^0
        @test Tuple(V8I32(v8i32)^1) === v8i32.^1
        @test Tuple(V8I32(v8i32)^2) === v8i32.^2
        @test Tuple(V8I32(v8i32)^3) === v8i32.^3
    end

    @testset "saturation" begin
        v = Vec{4, UInt8}(UInt8.((150, 250, 125, 0)))
        @test SIMD.add_saturate(v, UInt8(50)) === Vec{4, UInt8}(UInt8.((200, 255, 175, 50)))
        @test SIMD.sub_saturate(v, UInt8(100)) === Vec{4, UInt8}(UInt8.((50, 150, 25, 0)))
        v = Vec{4, Int8}(Int8.((100, -100, 20, -20)))
        @test SIMD.add_saturate(v, Int8(50)) === Vec{4, Int8}(Int8.((127, -50, 70, 30)))
        @test SIMD.sub_saturate(v, Int8(50)) === Vec{4, Int8}(Int8.((50, -128, -30, -70)))
    end

    using Base.Checked: add_with_overflow, sub_with_overflow, mul_with_overflow
    if Base.libllvm_version >= v"9"
        @testset "overflow arithmetic" begin
            for f in (add_with_overflow, sub_with_overflow, mul_with_overflow)
                for T in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
                    t2 = div(typemax(T), T(2)) + one(T)
                    t1 = div(typemin(T), T(2)) - (T <: Unsigned ? zero(T) : one(T))
                    v = Vec(t2, t1, T(0), t2 - one(T))
                    if f == mul_with_overflow && Sys.ARCH == :i686 && T == Int64
                        @test_throws ErrorException f(v,v)
                        continue
                    end
                    @test Tuple(zip(Tuple.(f(v,v))...)) === map(f, Tuple(v), Tuple(v))
                end
            end
        end
    end

    @testset "Floating point arithmetic functions" begin

        global const v4f64b = map(x->Float64(x+1), v4f64)
        global const v4f64c = map(x->Float64(x*2), v4f64)

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
        for op in (cos, exp, exp2, logabs, log10abs, log2abs, sin)
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

        @test Tuple(V4F64(v4f64)^0) === v4f64.^0
        @test Tuple(V4F64(v4f64)^1) === v4f64.^1
        @test Tuple(V4F64(v4f64)^2) === v4f64.^2
        @test Tuple(V4F64(v4f64)^3) === v4f64.^3

        for op in (fma, vifelsebool, muladd)
            @test Tuple(op(V4F64(v4f64), V4F64(v4f64b), V4F64(v4f64c))) ===
                map(op, v4f64, v4f64b, v4f64c)
        end

        v = V4F64(v4f64)
        @test v^5 === v * v * v * v * v

        # Make sure our dispatching rule does not select floating point `pow`.
        # See: https://github.com/eschnett/SIMD.jl/pull/43
        ir = llvm_ir(^, (V4F64(v4f64), 2))
        @test occursin("@llvm.powi.v4f64", ir)
    end

    @testset "Type promotion" begin

        for op in (
                ==, !=, <, <=, >, >=,
                &, |, ⊻, +, -, *, copysign, div, flipsign, max, min, rem)
            @test op(42, V8I32(v8i32)) === op(V8I32(42), V8I32(v8i32))
            @test op(V8I32(v8i32), 42) === op(V8I32(v8i32), V8I32(42))
        end
        @test vifelse(signbit(V8I32(v8i32)), 42, V8I32(v8i32)) ===
            vifelse(signbit(V8I32(v8i32)), V8I32(42), V8I32(v8i32))
        @test vifelse(signbit(V8I32(v8i32)), V8I32(v8i32), 42) ===
            vifelse(signbit(V8I32(v8i32)), V8I32(v8i32), V8I32(42))
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
                +, -, *, /, copysign, flipsign, max, min, rem)
            @test op(42, V4F64(v4f64)) === op(V4F64(42), V4F64(v4f64))
            @test op(V4F64(v4f64), 42) === op(V4F64(v4f64), V4F64(42))
        end
        @test vifelse(signbit(V4F64(v4f64)), 42, V4F64(v4f64)) ===
            vifelse(signbit(V4F64(v4f64)), V4F64(42), V4F64(v4f64))
        @test vifelse(signbit(V4F64(v4f64)), V4F64(v4f64), 42) ===
            vifelse(signbit(V4F64(v4f64)), V4F64(v4f64), V4F64(42))
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
    end

    @testset "Reduction operations" begin

        for op in (maximum, minimum, sum, prod)
            @test op(V8I32(v8i32)) === op(v8i32)
        end
        t = Vec(true, true, true, true)
        tf = Vec(true, false, true, false)
        f = Vec(false, false, false, false)
        @test all(t) == reduce(&, t) == true
        @test all(tf) == reduce(&, tf) == false
        @test any(f) == reduce(|, f) == false
        @test any(tf) == reduce(|, tf) == true
        @test sum(t) == reduce(+, t) == 4
        @test sum(tf) == reduce(+, tf) == 2
        @test sum(f) == reduce(+, f) == 0

        b64 = rand(Bool, 64)
        tf64 = Vec(b64...)
        @test sum(tf64) == sum(b64)

        b65 = rand(Bool, 65)
        tf65 = Vec(b65...)
        @test sum(tf65) == sum(b65)

        for op in (maximum, minimum, sum, prod)
            @test op(V4F64(v4f64)) === op(v4f64)
        end

        @test sum(Vec{3,Float64}(1)) === 3.0
        @test prod(Vec{5,Float64}(2)) === 32.0
    end

    @testset "Load and store functions" begin

        global const arri32 = valloc(Int32, L8, 2*L8) do i i end
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

        global const arrf64 = valloc(Float64, L4, 4*L4) do i i end
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
    end

    @testset "fastmath" begin
        v = Vec(1.0,2.0,3.0,4.0)
        @test all(Tuple(@fastmath v+v) .≈ Tuple(v+v))
        @test all(Tuple(@fastmath v+1.0) .≈ Tuple(v+1.0))
        @test all(Tuple(@fastmath 1.0+v) .≈ Tuple(1.0+v))
        @test all(Tuple(@fastmath -v) .≈ Tuple(-v))
        f = v -> @fastmath v + v
        # Test that v+v is rewritten as v * 2.0 (change test if optimization changes)
        @test occursin(r"fmul fast <4 x double> %[0-9]*, <double 2\.000000e\+00", llvm_ir(f, (v,)))
    end

    @testset "Gather and scatter function" begin
        for (arr, VT) in [(arri32, V8I32), (arrf64, V4F64)]
            arr .= 1:length(arr)
            idxarr = floor.(Int, range(1, stop=length(arr), length=length(VT)))

            idx = Vec(Tuple(idxarr))
            @test vgather(arr, idx) === convert(VT, idx)
            @test vgathera(arr, idx) === convert(VT, idx)
            @test arr[idx] === convert(VT, idx)

            # Masked gather
            maskarr = zeros(Bool, length(VT))
            for i in 1:length(VT)
                maskarr[i] = true
                mask = Vec(Tuple(maskarr))
                @test vgather(arr, idx, mask) === VT(Tuple(idxarr .* maskarr))
                @test vgathera(arr, idx, mask) === VT(Tuple(idxarr .* maskarr))
                @test arr[idx, mask] === VT(Tuple(idxarr .* maskarr))
            end

            # Scatter
            varr = convert.(eltype(VT), idxarr .* 123)
            v = Vec(Tuple(varr))

            vscatter(v, fill!(arr, 0), idx)
            @test arr[idxarr] == varr
            vscattera(v, fill!(arr, 0), idx)
            @test arr[idxarr] == varr
            fill!(arr, 0)
            arr[idx] = v
            @test arr[idxarr] == varr

            # Masked scatter
            maskarr = zeros(Bool, length(VT))
            for i in 1:length(VT)
                maskarr[i] = true
                mask = Vec(Tuple(maskarr))
                vscatter(v, fill!(arr, 0), idx, mask)
                @test arr[idxarr] == varr .* maskarr
                vscattera(v, fill!(arr, 0), idx, mask)
                @test arr[idxarr] == varr .* maskarr
                fill!(arr, 0)
                arr[idx, mask] = v
                @test arr[idxarr] == varr .* maskarr
            end
        end
    end

    @testset "expandload" begin
        if Base.libllvm_version >= v"9" || Sys.CPU_NAME == "skylake"
            for arr in [arri32, arrf64]
                VT = Vec{4,eltype(arr)}
                arr .= 1:length(arr)
                @test vloadx(arr, 1, Vec((true, false, false, true))) === VT((1, 0, 0, 2))
                @test vloadx(arr, 1, Vec((true, false, true, false))) === VT((1, 0, 2, 0))
                @test vloadx(arr, 1, Vec((true, true, false, false))) === VT((1, 2, 0, 0))
                @test vloadx(arr, 1, Vec((true, false, false, false))) === VT((1, 0, 0, 0))
                @test vloadx(arr, 1, Vec((false, false, false, false))) === VT((0, 0, 0, 0))
            end
        else
            @info "Skipping tests for `expandload`" Base.libllvm_version Sys.CPU_NAME
        end
    end

    @testset "compressstore" begin
        if Base.libllvm_version >= v"9" || Sys.CPU_NAME == "skylake"
            for arr in [arri32, arrf64]
                VT = Vec{4,eltype(arr)}
                arr .= 1:length(arr)
                x = VT(Tuple(arr[1:4]))
                @test vstorec(x, zero(arr), 1, Vec((true, false, false, true))) ==
                    [[arr[1], arr[4]]; zero(arr)[3:end]]
                @test vstorec(x, zero(arr), 1, Vec((true, false, true, false))) ==
                    [[arr[1], arr[3]]; zero(arr)[3:end]]
                @test vstorec(x, zero(arr), 1, Vec((true, true, false, false))) ==
                    [[arr[1], arr[2]]; zero(arr)[3:end]]
                @test vstorec(x, zero(arr), 1, Vec((true, false, false, false))) ==
                    [[arr[1]]; zero(arr)[2:end]]
                @test vstorec(x, zero(arr), 1, Vec((false, false, false, false))) ==
                    zero(arr)
            end
        else
            @info "Skipping tests for `compressstore`" Base.libllvm_version Sys.CPU_NAME
        end
    end

    @testset "Index-based load/store" begin
        for (arr, VT) in [(arri32, V8I32), (arrf64, V4F64)]
            @testset "Vector ($VT)" begin
                fill!(arr.parent, 0)
                arr .= 1:length(arr)
                idx = VecRange{length(VT)}(1)
                @test arr[idx] === VT(Tuple(1:length(VT)))

                maskarr = zeros(Bool, length(VT))
                maskarr[1] = true
                mask = Vec(Tuple(maskarr))
                varr = zeros(length(VT))
                varr[1] = 1
                @test arr[idx, mask] === VT(Tuple(varr))

                @test_throws ArgumentError arr[idx, 1]
                @test_throws ArgumentError arr[idx, 1, mask]
                @test_throws ArgumentError arr[idx, mask, 1]
                @test_throws ArgumentError arr[idx, 1, mask, 1]

                lane = VecRange{length(VT)}(0)
                @test_throws BoundsError arr[lane]
                @test_throws BoundsError arr[lane + end]

                # Out-of-bound access (OK with valloc)
                slice = LinearIndices(arr)[end-length(VT)+2:end]
                varr = zeros(length(VT))
                varr[1:length(slice)] .= arr[slice]
                i = lane + first(slice)
                @test_throws BoundsError arr[i]
                f(x, i) = @inbounds x[i]
                if !is_checking_bounds
                    @test f(arr, i) === VT(Tuple(varr))
                end
            end

            arr .= 1:length(arr)
            @testset "$name" for (name, mat) in [
                        ("Matrix ($VT)", repeat(arr, outer=(1, 3))),
                        ("Matrix ($VT) with non-strided row",
                         view(repeat(arr, outer=(1, 5)), :, [2, 1, 5])),
                    ]
                idx = VecRange{length(VT)}(1)
                @test mat[idx, 1] === VT(Tuple(1:length(VT)))
                @test mat[idx, 2] === VT(Tuple(1:length(VT)))
                if mat isa SIMD.FastContiguousArray
                    @test mat[idx] === VT(Tuple(1:length(VT)))
                else
                    err = try
                        mat[idx]
                        nothing
                    catch err
                        err
                    end
                    @test err isa ArgumentError
                    @test occursin(
                        "Exactly 2 (non-mask) indices have to be specified.",
                        sprint(showerror, err))
                end

                maskarr = zeros(Bool, length(VT))
                maskarr[1] = true
                mask = Vec(Tuple(maskarr))
                varr = zeros(length(VT))
                varr[1] = 1
                @test mat[idx, 1, mask] === VT(Tuple(varr))
                if mat isa SIMD.FastContiguousArray
                    @test mat[idx, mask] === VT(Tuple(varr))
                else
                    err = try
                        mat[idx]
                        nothing
                    catch err
                        err
                    end
                    @test err isa ArgumentError
                    @test occursin(
                        "Exactly 2 (non-mask) indices have to be specified.",
                        sprint(showerror, err))
                end

                @test_throws ArgumentError mat[idx, 1, 1]
                @test_throws ArgumentError mat[idx, 1, 1, mask]
                @test_throws ArgumentError mat[idx, mask, 1]
                @test_throws ArgumentError mat[idx, 1, mask, 1]

                lane = VecRange{length(VT)}(0)
                @test_throws BoundsError mat[lane, 1]
                @test_throws BoundsError mat[lane + end, 1]
                if mat isa SIMD.FastContiguousArray
                    @test_throws BoundsError mat[lane + end]
                else
                    err = try
                        mat[lane + end]
                        nothing
                    catch err
                        err
                    end
                    @test err isa ArgumentError
                    @test occursin(
                        "Exactly 2 (non-mask) indices have to be specified.",
                        sprint(showerror, err))
                end

                # Out-of-bound access
                varr = collect(1:length(VT))
                i = lane + size(mat, 1) + 1
                @test_throws BoundsError mat[i, 1]
                f(x, i) = @inbounds x[i, 1]
                if !is_checking_bounds
                    @test f(mat, i) === VT(Tuple(varr))
                end
            end

            @testset "3D array ($VT)" begin
                arr .= 1:length(arr)
                arr3d = repeat(arr, outer=(1, 3, 5))
                idx = VecRange{length(VT)}(1)
                @test arr3d[idx, 1, 1] === VT(Tuple(1:length(VT)))
                @test arr3d[idx, 2, 1] === VT(Tuple(1:length(VT)))
                @test arr3d[idx, 1, 2] === VT(Tuple(1:length(VT)))
                @test arr3d[idx] === VT(Tuple(1:length(VT)))

                maskarr = zeros(Bool, length(VT))
                maskarr[1] = true
                mask = Vec(Tuple(maskarr))
                varr = zeros(length(VT))
                varr[1] = 1
                @test arr3d[idx, 1, 1, mask] === VT(Tuple(varr))
                @test arr3d[idx, mask] === VT(Tuple(varr))

                @test_throws ArgumentError arr3d[idx, 1, 1, 1]
                @test_throws ArgumentError arr3d[idx, 1, 1, 1, mask]
                @test_throws ArgumentError arr3d[idx, mask, 1]
                @test_throws ArgumentError arr3d[idx, 1, mask, 1]

                lane = VecRange{length(VT)}(0)
                @test_throws BoundsError arr3d[lane, 1, 1]
                @test_throws BoundsError arr3d[lane + end, 1, 1]
                @test_throws BoundsError arr3d[lane + end]

                # Out-of-bound access
                varr = collect(1:length(VT))
                i = lane + size(arr3d, 1) + 1
                @test_throws BoundsError arr3d[i, 1, 1]
                f(x, i) = @inbounds x[i, 1, 1]
                if !is_checking_bounds
                    @test f(arr3d, i) === VT(Tuple(varr))
                end
            end
        end
    end

    @testset "Real-world examples" begin

        function vadd!(xs::AbstractArray{T,1}, ys::AbstractArray{T,1},
                       ::Type{Vec{N,T}}) where {N,T}
            @assert length(ys) == length(xs)
            @assert length(xs) % N == 0
            lane = VecRange{N}(0)
            @inbounds for i in 1:N:length(xs)
                xs[lane + i] += ys[lane + i]
            end
        end

        let xs = valloc(Float64, L4, 4*L4) do i i end,
            ys = valloc(Float64, L4, 4*L4) do i 1 end
            vadd!(xs, ys, V4F64)
            @test xs == Float64[i+1 for i in 1:(4*L4)]
            # @code_native vadd!(xs, ys, V4F64)

            ir = llvm_ir(vadd!, (xs, ys, V4F64))
            @test occursin(r"( load <4 x double>.*){2}"s, ir)
            @test occursin(" store <4 x double>", ir)
            @test occursin(" fadd <4 x double>", ir)
        end

        function vsum(xs::AbstractArray{T,1}, ::Type{Vec{N,T}}) where {N,T}
            @assert length(xs) % N == 0
            sv = Vec{N,T}(0)
            lane = VecRange{N}(0)
            @inbounds for i in 1:N:length(xs)
                sv += xs[lane + i]
            end
            sum(sv)
        end

        let xs = valloc(Float64, L4, 4*L4) do i i end
            s = vsum(xs, V4F64)
            @test s === (x->(x^2+x)/2)(Float64(4*L4))
            # @code_native vsum(xs, V4F64)

            ir = llvm_ir(vsum, (xs, V4F64))
            @test occursin(" load <4 x double>", ir)
            @test occursin(" fadd <4 x double>", ir)
            # @test occursin(r"( shufflevector <4 x double>.*){2}"s, ir)
        end

        function vadd_masked!(xs::AbstractArray{T,1}, ys::AbstractArray{T,1},
                              ::Type{Vec{N,T}}) where {N, T}
            @assert length(ys) == length(xs)
            limit = length(xs) - (N-1)
            vlimit = Vec(ntuple(i -> length(xs) - i + 1, Val(N)))
            lane = VecRange{N}(0)
            @inbounds for i in 1:N:length(xs)
                if i <= limit
                    xs[lane + i] += ys[lane + i]
                else
                    mask = Vec{N,Int}(i) <= vlimit
                    xs[lane + i, mask] = xs[lane + i, mask] + ys[lane + i, mask]
                end
            end
        end

        let xs = valloc(Float64, 4, 13) do i i end,
            ys = valloc(Float64, 4, 13) do i 1 end
            vadd_masked!(xs, ys, V4F64)
            @test xs == Float64[i+1 for i in 1:13]
            # @code_native vadd!(xs, ys, V4F64)

            ir = llvm_ir(vadd_masked!, (xs, ys, V4F64))
            @test occursin(r"(masked.load.v4f64.*){2}"s, ir)
            @test occursin("masked.store.v4f64", ir)
            @test occursin(" store <4 x double>", ir)
            @test occursin(" fadd <4 x double>", ir)
        end

        function vsum_masked(xs::AbstractArray{T,1}, ::Type{Vec{N,T}}) where {N,T}
            vlimit = Vec(ntuple(i -> length(xs) - i + 1, Val(N)))
            sv = Vec{N,T}(0)
            lane = VecRange{N}(0)
            @inbounds for i in 1:N:length(xs)
                mask = Vec{N,Int}(i) <= vlimit
                sv += xs[lane + i, mask]
            end
            sum(sv)
        end

        let xs = valloc(Float64, 4, 13) do i i end
            s = vsum_masked(xs, V4F64)
            @code_llvm vsum(xs, V4F64)
            @code_native vsum(xs, V4F64)
            @test s === sum(xs)

            ir = llvm_ir(vsum_masked, (xs, V4F64))
            @test occursin("masked.load.v4f64", ir)
            @test occursin(" fadd <4 x double>", ir)
            # @test occursin(r"( shufflevector <4 x double>.*){2}"s, ir)
        end

        @inline function vcompress!(dest, pred, src, ::Val{N} = Val(4)) where {N}
            @assert axes(dest) == axes(src) == axes(pred)
            simd_bound = lastindex(src) - N + 1
            i = firstindex(src)
            j = firstindex(dest)
            lanes = VecRange{N}(0)
            @inbounds while i <= simd_bound
                mask = pred[lanes + i]
                vstorec(src[lanes + i], dest, j, mask)
                j += sum(mask)
                i += N
            end
            @inbounds while i <= lastindex(src)
                if pred[i]
                    dest[j] = src[i]
                    j += 1
                end
                i += 1
            end
            return dest
        end

        if Base.libllvm_version >= v"9" || Sys.CPU_NAME == "skylake"
            let n = 15
                src = randn(n)
                pred = Vector{Bool}(src .> 0)
                dest = zero(src)

                vcompress!(dest, pred, src)
                @test dest[1:sum(pred)] == src[src .> 0]

                @code_llvm vcompress!(dest, pred, src)
                @code_native vcompress!(dest, pred, src)

                ir = llvm_ir(vcompress!, (dest, pred, src))
                @test occursin("masked.compressstore.v4f64", ir)
            end
        end

    end

    @testset "Vector shuffles" begin

        for T in (Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float32,Float64)
            a = Vec{4,T}((1,2,3,4))
            b = Vec{4,T}((5,6,7,8))
            @test shufflevector(a, b, Val((2,3,4,5))) === Vec{4,T}((3,4,5,6))
            @test shufflevector(a, b, Val((1,7,5,5))) === Vec{4,T}((2,8,6,6))
            @test shufflevector(a, b, Val(0:3)) === a
            @test shufflevector(a, b, Val(4:7)) === b
            @test shufflevector(a, Val((1,0,2,3))) === Vec{4,T}((2,1,3,4))
            @test shufflevector(a, b, Val((0,1,4,5,2,3,6,7))) === Vec{8,T}((1,2,5,6,3,4,7,8))
            @test shufflevector(shufflevector(a, b, Val((6,:undef,0,:undef))), Val((0,2))) === Vec{2,T}((7,1))
            @test isa(shufflevector(a, Val((:undef,:undef,:undef,:undef))), Vec{4,T})
            c = Vec{8,T}((1:8...,))
            d = Vec{8,T}((9:16...,))
            @test shufflevector(c, d, Val((0,1,8,15))) === Vec{4,T}((1,2,9,16))
            @test shufflevector(c, d, Val(1:2:15)) === Vec{8,T}((2:2:16...,))
        end

        let
            a = Vec{4,Bool}((true,false,true,false))
            b = Vec{4,Bool}((false,false,true,true))
            @test shufflevector(a, b, Val((2,3,4,5))) === Vec{4,Bool}((true,false,false,false))
        end

        for T in (Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float32,Float64)
            a = Vec{4,T}((1,2,3,4))
            b = Vec{4,T}((5,6,7,8))
            @test shufflevector(a, b, Val((2,3,4,5))) === Vec{4,T}((3,4,5,6))
            @test shufflevector(a, b, Val((1,7,5,5))) === Vec{4,T}((2,8,6,6))
            @test shufflevector(a, b, Val(0:3)) === a
            @test shufflevector(a, b, Val(4:7)) === b
            @test shufflevector(a, Val((1,0,2,3))) === Vec{4,T}((2,1,3,4))
            @test shufflevector(a, b, Val((0,1,4,5,2,3,6,7))) === Vec{8,T}((1,2,5,6,3,4,7,8))
            @test shufflevector(shufflevector(a, b, Val((6,:undef,0,:undef))), Val((0,2))) === Vec{2,T}((7,1))
            @test isa(shufflevector(a, Val((:undef,:undef,:undef,:undef))), Vec{4,T})
            c = Vec{8,T}((1:8...,))
            d = Vec{8,T}((9:16...,))
            @test shufflevector(c, d, Val((0,1,8,15))) === Vec{4,T}((1,2,9,16))
            @test shufflevector(c, d, Val(1:2:15)) === Vec{8,T}((2:2:16...,))
        end

        let
            a = Vec{4,Bool}((true,false,true,false))
            b = Vec{4,Bool}((false,false,true,true))
            @test shufflevector(a, b, Val((2,3,4,5))) === Vec{4,Bool}((true,false,false,false))
        end
    end
# end
