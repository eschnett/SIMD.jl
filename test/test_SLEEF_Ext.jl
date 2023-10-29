using SLEEFPirates

@testset "SIMD+SLEEFPirates" begin
    include("test_SIMD.jl")
end

const SLEEF_Ext = Base.get_extension(SIMD, :SLEEF_Ext)
data(F, N, ::Function) = range(F(0.1), F(0.9), length = N)
data(F, N, ::typeof(acosh)) = range(F(1.1), F(1.9), length = N)

for (mod, unops) in (
    (Base, SLEEF_Ext.unops_Base_SP),
    (Base.FastMath, SLEEF_Ext.unops_FM_SP),
    (Base.FastMath, SLEEF_Ext.unops_FM_SP_slow))
    for unop in unops
        op = getfield(mod, unop)
        @testset "$(string(mod)).$(string(op))" begin
            for F in (Float32, Float64), N in (4, 8, 16, 32)
                d = data(F, N, op)
                ref = SIMD.Vec(map(op, d)...)
                arg = SIMD.Vec(d...)
                res = op(arg)
                if any(abs(res - ref) > 4eps(F))
                    @info op arg ref res abs(ref - res)
                    @test false
                else
                    @test true
                end
            end
        end
    end
end
