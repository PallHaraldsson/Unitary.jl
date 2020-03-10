struct LUDense{M, B, S}
	m::M
	b::B
	σ::S
end

Base.show(io::IO, a::LUDense) = print(io, "LUDense{$(size(a.m)), $(a.σ)}")

Flux.@functor LUDense

"""
	LUDense(n, σ)

	Dense layer with square weight matrix of dimension `n` parametrized in 
	LU decomposition.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
"""
function LUDense(n::Int, σ)
	n == 1 && return(ScaleShift(1, σ))
	return(_ludense(n, σ))
end


using LinearAlgebra

_ludense(n::Int, σ) = LUDense(lowup(Float32, n), 0.01f0.*randn(Float32,n), σ)

(a::LUDense)(x::AbstractMatVec) = a.σ.((a.m * x) .+ a.b)

function (a::LUDense)(xx::Tuple{A,B}) where {A,B}
	x, logdet = xx
	pre = (a.m * x) .+ a.b
	g = explicitgrad.(a.σ, pre)
	(a.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(a.m))
end

struct InvertedLUDense{M, B, S}
	m::M
	b::B
	σ::S
end
Flux.@functor InvertedLUDense

Base.inv(a::LUDense) = InvertedLUDense(inv(a.m), a.b, inv(a.σ))
Base.inv(a::InvertedLUDense) = LUDense(inv(a.m), a.b, inv(a.σ))

(a::InvertedLUDense)(x::AbstractMatVec)  = (a.m * (a.σ.(x) .- a.b))
