using KernelAbstractions, OpenCL, pocl_jll
using SIMD
using Test

const backend = OpenCLBackend()

@testset "OpenCL SIMD Load/Store Tests" begin
    @testset "Basic load/store operations" begin
        @kernel function load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{4, Float32}, b, i)
            @inbounds vstore(xs + Vec{4, Float32}(1f0), a, i)
        end

        b = KernelAbstractions.zeros(backend, Float32, 1024)
        a = similar(b)
        load_store_kernel!(backend)(a, b; ndrange = 1024 ÷ 4)
        @test all(==(1f0), a)
    end

    @testset "Multiple data types load/store" begin
        # Test Int32 vectors
        @kernel function load_store_int32_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{4, Int32}, b, i)
            @inbounds vstore(xs + Vec{4, Int32}(10), a, i)
        end

        b_int = KernelAbstractions.zeros(backend, Int32, 256)
        fill!(b_int, 5)
        a_int = similar(b_int)
        load_store_int32_kernel!(backend)(a_int, b_int; ndrange = 256 ÷ 4)
        @test all(==(15), a_int)

        # Test different vector sizes for Float32
        @kernel function load_store_float32_vec8_kernel!(a, b)
            i = 8 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{8, Float32}, b, i)
            @inbounds vstore(xs * Vec{8, Float32}(2f0), a, i)
        end

        b_f32 = KernelAbstractions.ones(backend, Float32, 512)
        a_f32 = similar(b_f32)
        load_store_float32_vec8_kernel!(backend)(a_f32, b_f32; ndrange = 512 ÷ 8)
        @test all(==(2f0), a_f32)
    end

    @testset "Aligned load/store operations" begin
        @kernel function aligned_load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vloada(Vec{4, Float32}, b, i)
            @inbounds vstorea(xs + Vec{4, Float32}(3f0), a, i)
        end

        b = KernelAbstractions.zeros(backend, Float32, 1024)
        fill!(b, 2f0)
        a = similar(b)
        aligned_load_store_kernel!(backend)(a, b; ndrange = 1024 ÷ 4)
        @test all(==(5f0), a)
    end

    @testset "Non-temporal load/store operations" begin
        @kernel function nontemporal_load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vloadnt(Vec{4, Float32}, b, i)
            @inbounds vstorent(xs * Vec{4, Float32}(1.5f0), a, i)
        end

        b = KernelAbstractions.ones(backend, Float32, 512)
        fill!(b, 4f0)
        a = similar(b)
        nontemporal_load_store_kernel!(backend)(a, b; ndrange = 512 ÷ 4)
        @test all(==(6f0), a)
    end

    # Tested with `backend = ROCBackend()`, masked load/store not supported by the OpenCL backend

    #@testset "Masked load/store operations" begin
    #    @kernel function masked_load_store_kernel!(a, b, masks_buf)
    #        i = 4 * (@index(Global) - 1) + 1

    #        # Create mask from buffer (convert to bool)
    #        mask = Vec{4, Bool}((
    #            masks_buf[i] > 0,
    #            masks_buf[i+1] > 0,
    #            masks_buf[i+2] > 0,
    #            masks_buf[i+3] > 0
    #        ))

    #        xs = @inbounds vload(Vec{4, Float32}, b, i, mask)
    #        result = xs + Vec{4, Float32}(10f0)
    #        @inbounds vstore(result, a, i, mask)
    #    end

    #    n = 256
    #    b = KernelAbstractions.ones(backend, Float32, n)
    #    a = KernelAbstractions.zeros(backend, Float32, n)

    #    # Create mask pattern: alternating true/false
    #    mask_data = [i % 2 for i in 1:n]
    #    masks_buf = similar(a, Int32, n)
    #    copyto!(masks_buf, mask_data)

    #    masked_load_store_kernel!(backend)(a, b, masks_buf; ndrange = n ÷ 4)

    #    a_host = Array(a)
    #    # Check that masked elements were updated, unmasked remain zero
    #    for i in 1:n
    #        if i % 2 == 1  # mask was true
    #            @test a_host[i] == 11f0
    #        else  # mask was false
    #            @test a_host[i] == 0f0
    #        end
    #    end
    #end

    @testset "Gather/scatter operations" begin
        function gather_scatter_kernel!(a, b, indices_buf)
            idx = get_global_id()
            base_i = 4 * (idx - 1) + 1

            # Load indices
            indices = Vec{4, Int32}((
                indices_buf[base_i],
                indices_buf[base_i + 1],
                indices_buf[base_i + 2],
                indices_buf[base_i + 3]
            ))

            # Gather from b using indices
            gathered = @inbounds vgather(b, indices)

            # Process and scatter back to a
            result = gathered * Vec{4, Float32}(2f0)
            @inbounds vscatter(result, a, indices)
            return
        end

        n = 128
        b = CLVector{Float32}(1:n)
        a = CLVector{Float32}(undef, n)

        # Create gather indices (1-based)
        indices_data = Int32[]
        for i in 1:4:(n-3)
            # Gather in reverse order for testing
            append!(indices_data, [i+3, i+2, i+1, i])
        end
        indices_buf = CLVector{Int32}(indices_data)

        @opencl global_size = length(indices_data) ÷ 4 backend = :khronos extensions = ["SPV_INTEL_masked_gather_scatter"] validate = false gather_scatter_kernel!(a, b, indices_buf)

        a_host = Array(a)
        # Verify scattered results
        for (i, original_idx) in enumerate(indices_data)
            expected = Float32(original_idx) * 2f0
            @test a_host[original_idx] == expected
        end
    end

    @testset "Masked gather/scatter operations" begin
        function masked_gather_scatter_kernel!(a, b, indices_buf, masks_buf)
            idx = get_global_id()
            base_i = 4 * (idx - 1) + 1

            # Load indices and mask
            indices = Vec{4, Int32}((
                indices_buf[base_i],
                indices_buf[base_i + 1],
                indices_buf[base_i + 2],
                indices_buf[base_i + 3]
            ))

            mask = Vec{4, Bool}((
                masks_buf[base_i] > 0,
                masks_buf[base_i + 1] > 0,
                masks_buf[base_i + 2] > 0,
                masks_buf[base_i + 3] > 0
            ))

            # Masked gather
            gathered = @inbounds vgather(b, indices, mask)

            # Process and masked scatter
            result = gathered + Vec{4, Float32}(100f0)
            @inbounds vscatter(result, a, indices, mask)
            return
        end

        n = 64
        b = CLVector{Float32}(Float32.(1:n) .* 10f0)
        a = OpenCL.zeros(Float32, n)

        # Create indices and masks
        indices_data = Int32.(1:n)
        indices_buf = CLVector{Int32}(indices_data)

        # Checkerboard mask pattern
        mask_data = [i % 2 for i in 1:n]
        masks_buf = CLVector{Int32}(mask_data)

        @opencl global_size = n ÷ 4 backend = :khronos extensions = ["SPV_INTEL_masked_gather_scatter"] validate = false masked_gather_scatter_kernel!(a, b, indices_buf, masks_buf)

        a_host = Array(a)
        for i in 1:n
            if i % 2 == 1  # mask was true
                expected = Float32(i) * 10f0 + 100f0
                @test a_host[i] == expected
            else  # mask was false, should remain zero
                @test a_host[i] == 0f0
            end
        end
    end
end
