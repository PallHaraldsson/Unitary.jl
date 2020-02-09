struct UnitaryButterfly{T<:Tuple}
	matrices::T
	n::Int
end

Flux.@functor(UnitaryButterfly)

Base.size(a::UnitaryButterfly) = (a.n,a.n)
Base.size(a::UnitaryButterfly, i::Int) = a.n

Base.eltype(a::UnitaryButterfly) = eltype(a.matrices[1])
LinearAlgebra.transpose(a::UnitaryButterfly) = UnitaryButterfly(transpose.(a.matrices[end:-1:1]), a.n)
Base.inv(a::UnitaryButterfly) = transpose(a)
Base.show(io::IO, a::UnitaryButterfly) = print(io, "$(a.n)x$(a.n) Unitary with $(length(a.matrices)) butterfly matrices")
Base.zero(a::UnitaryButterfly) = Butterfly(zero(a.θ), a.i, a.j, a.n)

function *(a::UnitaryButterfly, x::TransposedMatVec)
	@assert size(x,1) == a.n
	# switch between differentiable and efficient in-place operation
	if Flux.istraining() 
		return(foldr((u,v) -> u*v, a.matrices, init = x))
	else
		o = deepcopy(x);
		for i in length(a.matrices):-1:1
			mul!(o, a.matrices[i], o);
		end
		return(o)
	end
end
function *(x::TransposedMatVec, a::UnitaryButterfly) 
	@assert size(x,2) == a.n
	# switch between differentiable and efficient in-place operation
	if Flux.istraining()
		return(foldl((u,v) -> u*v, a.matrices, init = x))
	else
		o = deepcopy(x);
		for i in 1:length(a.matrices)
			mul!(o, o, a.matrices[i]);
		end
		return(o)
	end
end

"""
	UnitaryButterfly(n;indexes = :random)

	create a unitary matrix of dimension `n` parametrized by a set of givens rotations
	
	indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
function UnitaryButterfly(n)
	idxs = givenses(n)
	matrices = [Butterfly(i, n) for i in  idxs]
	UnitaryButterfly(tuple(matrices...), n)
end

