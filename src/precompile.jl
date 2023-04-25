using PrecompileTools

@compile_workload begin
    for dim in (2, 3, 4)
        v = SIMD.Vec(ntuple(i -> 0.0, dim))
        v + v
        v * v
        1.0 * v
        1.0 + v
        v2 = Base.setindex(v, 2.0, 1)
        muladd(v, v, v)
        SIMD.constantvector(1.0, Vec{dim, Float64})
    end
end
