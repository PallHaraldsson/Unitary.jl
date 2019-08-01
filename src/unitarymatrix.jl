
struct UnitaryMatrix{T}
	θ::T	
end

const TransposedUnitaryMatrix = Transpose{X,A} where {X, A<:UnitaryMatrix}
const TransposedVector = Transpose{X,A} where {X, A<: AbstractVector}

Flux.param(a::UnitaryMatrix) = UnitaryMatrix(param(a.θ))
Flux.@treelike(UnitaryMatrix)
@adjoint UnitaryMatrix(θ) = UnitaryMatrix(θ), Δ -> (UnitaryMatrix(Δ),)


# @adjoint Base.getfield(a::UnitaryMatrix, i) = (getproperty(a,i), Δ -> (UnitaryMatrix(Δ),)


"""
	Matrix(a::UnitaryMatrix{T})

	create a dense matrix from the unitary
"""
function Base.Matrix(a::UnitaryMatrix{T}) where {T<:Vector}
	θ = a.θ[1]
	[cos(θ) (- sin(θ)); sin(θ) cos(θ)]
end


Base.size(a::UnitaryMatrix) = (2,2)
Base.size(a::UnitaryMatrix, i::Int) = (i, i)
Base.eltype(a::UnitaryMatrix) = eltype(a.θ)
Base.length(a::UnitaryMatrix) = 4
Base.length(a::TransposedUnitaryMatrix) = 4
LinearAlgebra.transpose(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)
Base.inv(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)
Base.inv(a::TransposedUnitaryMatrix) = transpose(a)
Base.display(a::UnitaryMatrix) = print("UnitaryMatrix{$(eltype(a.θ))} [$(a.θ[1])]")
Base.display(a::TransposedUnitaryMatrix) = print("TransposedUnitaryMatrix{$(eltype(a.parent.θ))} [$(a.parent.θ[1])]")


*(a::UnitaryMatrix, x::TransposedMatVec) = _mulax(a.θ, x)
*(x::TransposedMatVec, a::UnitaryMatrix) = _mulxa(x, a.θ)
*(a::TransposedUnitaryMatrix, x::TransposedMatVec) = _mulatx(a.parent.θ, x)
*(x::TransposedMatVec, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)


@adjoint function *(a::UnitaryMatrix, x::TransposedMatVec)
	return _mulax(a.θ, x) , Δ -> (UnitaryMatrix(_∇mulax(a.θ, Δ, x)), _mulatx(a.θ, Δ))
end

@adjoint function *(x::TransposedMatVec, a::UnitaryMatrix)
	return _mulxa(x, a.θ) , Δ -> (_mulxat(Δ, a.θ), UnitaryMatrix(_∇mulxa(a.θ, Δ, x)))
end

@adjoint function *(a::TransposedUnitaryMatrix, x::TransposedMatVec)
  return _mulatx(a.parent.θ, x) , Δ -> (transpose(UnitaryMatrix(_∇mulatx(a.parent.θ, Δ, x))), _mulax(a.parent.θ, Δ))
end


@adjoint function *(x::TransposedMatVec, a::TransposedUnitaryMatrix)
  return _mulxat(x, a.parent.θ) , Δ -> (_mulxa(Δ, a.parent.θ), transpose(UnitaryMatrix(_∇mulxat(a.parent.θ, Δ, x))))
end


"""
	_mulax(θ::Vector, x::MatVec)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_mulax(θ::Vector, x) = _mulax((sin(θ[1]), cos(θ[1])), x)
function _mulax(sincosθ::Tuple, x)
	sinθ, cosθ = sincosθ
	@assert size(x, 1) == 2
	o = similar(x)
	for i in 1:size(x, 2)
		o[1, i] =  cosθ * x[1,i] - sinθ * x[2,i]
		o[2, i] =  sinθ * x[1,i] + cosθ * x[2,i]
	end
	o
end

"""
	_∇mulax(θ, Δ, x)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_∇mulax(θ::Vector, Δ, x) = _∇mulax((sin(θ[1]), cos(θ[1])), Δ,  x)
function _∇mulax(sincosθ::Tuple, Δ,  x)
	sinθ, cosθ = sincosθ
	∇θ = similar(x, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 2)
		∇θ[1] +=  Δ[1,i] * (- sinθ * x[1,i] - cosθ * x[2,i])
		∇θ[1] +=  Δ[2,i] * (  cosθ * x[1,i] - sinθ * x[2,i])
	end
	∇θ
end
_mulatx(θ::Vector, x) = _mulax((- sin(θ[1]), cos(θ[1])), x)
_∇mulatx(θ::Vector, Δ, x) = _∇mulax((sin(θ[1]), - cos(θ[1])), Δ, x)


_mulxa(x, θ::Vector) = _mulxa(x, (sin(θ[1]), cos(θ[1])))
_mulxat(x, θ::Vector) = _mulxa(x, (- sin(θ[1]), cos(θ[1])))
function _mulxa(x, sincosθ::Tuple)
	sinθ, cosθ = sincosθ 
	@assert size(x, 2) == 2
	o = similar(x)
	for i in 1:size(x, 1)
		o[i, 1] =    cosθ * x[i, 1] + sinθ * x[i, 2]
		o[i, 2] =  - sinθ * x[i, 1] + cosθ * x[i, 2]
	end
	o
end

_∇mulxa(θ::Vector, Δ, x) = _∇mulxa(Δ, x, (sin(θ[1]), cos(θ[1])))
_∇mulxat(θ::Vector, Δ, x) = _∇mulxa(Δ, x, (sin(θ[1]), - cos(θ[1])))
function _∇mulxa(Δ, x, sincosθ::Tuple)
	sinθ, cosθ = sincosθ
	∇θ = similar(x, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 1)
		∇θ[1] +=  Δ[i, 1] * (-sinθ * x[i, 1] + cosθ * x[i, 2])
		∇θ[1] +=  Δ[i, 2] * (-cosθ * x[i, 1] - sinθ * x[i, 2])
	end
	∇θ
end




@adjoint function _mulax(θ, x)
	return _mulax(θ, x) , Δ -> (_∇mulax(Δ, θ, x), _mulatx(θ, Δ))
end

@adjoint function _mulatx(θ, x)
  return _mulatx(θ, x) , Δ -> (_∇mulatx(θ, Δ, x), _mulax(θ, Δ))
end

@adjoint function _mulxa(x, θ)
	return _mulxa(x, θ,) , Δ -> (_mulxat(Δ, θ), _∇mulxa(θ, Δ, x))
end

@adjoint function _mulxat(x, θ)
  return _mulxat(x, θ) , Δ -> (_mulxa(Δ, θ), _∇mulxat(θ, Δ, x))
end
