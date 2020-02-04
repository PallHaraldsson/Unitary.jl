module Unitary

using Flux, LinearAlgebra, Zygote
using Zygote: @adjoint
import Base: *, transpose

const AbstractMatVec = Union{AbstractMatrix, AbstractVector}
const MatVec = Union{Matrix, Vector}
const TransposedMatVec = Union{Matrix, Vector,Transpose{T,Matrix{T}} where T, Transpose{T,Vector{T}} where T}

include("unitarymatrix.jl")
include("butterfly.jl")
include("unitarybutterfly.jl")
include("inplaceunitarybutterfly.jl")
include("givensindexes.jl")
include("diagonalrectangular.jl")
include("unitaryhouseholder.jl")
include("svd.jl")
include("scaleshift.jl")

export UnitaryHouseholder, UnitaryButterfly, SVDDense
end # module
