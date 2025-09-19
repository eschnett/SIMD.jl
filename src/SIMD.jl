module SIMD

using Base: @propagate_inbounds

export Vec, vload, vloada, vloadnt, vloadx, vstore, vstorea, vstorent, vstorec,
       vgather, vgathera, vscatter, vscattera, shufflevector, vifelse, valloc,
       VecRange, bitmaskconvert, frombitmask, tobitmask

const VE         = Base.VecElement
const LVec{N, T} = NTuple{N, VE{T}}

const IntTypes      = Union{Int8, Int16, Int32, Int64} # Int128 and UInt128 does not get passed as LLVM vectors
const BIntTypes     = Union{IntTypes, Bool}
const UIntTypes     = Union{UInt8, UInt16, UInt32, UInt64}
const IntegerTypes  = Union{IntTypes, UIntTypes}
const BIntegerTypes = Union{IntegerTypes, Bool}
const FloatingTypes = Union{Float16, Float32, Float64}
const ScalarTypes   = Union{IntegerTypes, FloatingTypes}
const VecTypes      = Union{ScalarTypes, Ptr, Bool}
include("LLVM_intrinsics.jl")
include("simdvec.jl")
include("arrayops.jl")
include("precompile.jl")

end
