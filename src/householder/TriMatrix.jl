using LinearAlgebra
struct TriMatrix{F} <: AbstractMatrix{F}
	A::Array{F, 1}
	n::Int
	lower::Bool
end

Base.size(a::TriMatrix) = a.n
Base.size(a::TriMatrix, i) = a.n
function Base.show(io::IO, ::MIME"text/plain", a::TriMatrix)
	println("TriMatrix:")
	display(Matrix(a))
end

Base.eltype(a::TriMatrix{T}) where {T} = T
TriNum(i::Int) = (i+1)*i÷2
function Base.Matrix(a::TriMatrix)
	n = a.n
	x = zeros(eltype(a), n, n)
	if a.lower
		for i = 1:n
			x[i:n, i] = a.A[(1+(i-1)*n-TriNum(i-2)):(i*n-TriNum(i-1))]
		end
	else
		for i = 1:n
			x[1:i, i] = a.A[(TriNum(i-1)+1):(TriNum(i))]
		end
	end
	x
end

TriMatrix(n::Int) = TriMatrix(Float64, n)
TriMatrix(T::DataType, n::Int) = TriMatrix(zeros(T, TriNum(n)), n, true)
function TriMatrix(Y::AbstractMatrix, lower::Bool)
	n = size(Y, 1)
	@assert n == size(Y, 2)
	A = Array{eltype(Y), 1}(undef, TriNum(n))
	if lower
		for i = 1:n
			A[(1+(i-1)*n-TriNum(i-2)):(i*n-TriNum(i-1))] = Y[i:n, i]
		end
	else
		for i = 1:n
			A[(TriNum(i-1)+1):(TriNum(i))] = Y[1:i, i]
		end
	end
	TriMatrix(A, n, lower)
end

function LinearAlgebra.transpose(a::TriMatrix)
	n = a.n
	out = Array{eltype(a), 1}(undef, TriNum(n))
	if a.lower
		for i = 1:n
			for j = 1:i
				out[TriNum(i-1)+j] = a.A[i+2*n-TriNum(n-j)]
			end
		end
	else
		for i = 1:n
			for j = 1:(n-i+1)
				out[j+(i-1)*n-TriNum(i-2)] = a.A[i + TriNum(j-2+i)]
			end
		end
	end
	TriMatrix(out, n, !a.lower)
end

using LinearAlgebra
function mul(a::TriMatrix, b::TriMatrix)
	n = a.n
	@assert n == b.n
	out = Array{eltype(a.A[1]*b.A[1]), 1}(undef, TriNum(n))
	if xor(a.lower, b.lower)
		return Matrix(a)*Matrix(b)
	end
	if a.lower
		for i = 1:n
			for j = 1:(n-i+1)
				 for k = i:(j+i-1)
					@inbounds out[j+(i-1)*n-TriNum(i-2)] += a.A[j-k+i+(k-1)*n-TriNum(k-2)]*b.A[(k+(i-1)*(n-1)-TriNum(i-2))]
				end
			end
		end
	else
		for i = 1:n
			for j = 1:i
				for k = j:i
					@inbounds out[j + TriNum(i-1)] = a.A[TriNum(k-1)+j]*b.A[(k+TriNum(i-1))]
				end
				#@views out[j + TriNum(i-1)] = dot([a.A[TriNum(k-1)+j] for k = j:i], b.A[(j+TriNum(i-1)):(TriNum(i))])
			end
		end
	end
	TriMatrix(out, n, a.lower)
end
import Base: *
function *(a::TriMatrix, b::TriMatrix)
	mul(a::TriMatrix, b::TriMatrix)
end
