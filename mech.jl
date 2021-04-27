using LinearAlgebra
using Symbolics
@variables _a, _α, _d, _θ
@variables t
@variables Fx, Fy, Fz

D = Differential(t)
dt(x) = D(x)
@register dt(x)

s(x) = sinpi(x/pi)
c(x) = cospi(x/pi)
@register s(x)
@register c(x)

T = [	c(_θ)		-s(_θ)		0		_a
	s(_θ)*c(_α)	c(_θ)*c(_α)	-s(_α)		-s(_α)*_d
	s(_θ)*s(_α)	c(_θ)*s(_α)	c(_α)		c(_α)*_d
	0		0		0		1	
]

function dh2t(dh::Array{Num,2}, k = 0) 
	if k > size(dh,1) || k < 0
		throw(ArgumentError("k"))
	end
	
	TM = Matrix(I,4,4)
	if k != 0
		TT = copy(T)
		TT = substitute.(TT,(_a => dh[k,1]))
		TT = substitute.(TT,(_α => dh[k,2]))
		TT = substitute.(TT,(_d => dh[k,3]))
		TT = substitute.(TT,(_θ => dh[k,4]))
		return TM * TT
	end
	for i in 1:size(dh,1)
		TT = copy(T)
		TT = substitute.(TT,(_a => dh[i,1]))
		TT = substitute.(TT,(_α => dh[i,2]))
		TT = substitute.(TT,(_d => dh[i,3]))
		TT = substitute.(TT,(_θ => dh[i,4]))	
		TM = TM * TT
		#simplify(TM)
	end

	return TM
end

function dh2jacob(dh::Array{Num,2}, k = 0)
	dof = count(type -> type != 0, dh[:,5])
	jacob = zeros(Num,3,dof)
	
	v = dh2vw(dh,k)[1]
	
	diffs = []
	for i in 1:size(dh,1)
		if dh[i,5] == 1
			push!(diffs, dt(dh[i,4]))
		end
		if dh[i,5] == 2
			push!(diffs, dt(dh[i,3]))
		end
	end	

	for i in 1:size(diffs,1)
		vtemp = copy(v)
	
		for j in 1:size(diffs,1)
			vtemp = substitute.(vtemp,[diffs[j]=>Int(j==i)])
		end

		jacob[:,i] = vtemp	
	end

	I(3) * jacob
end

function dh2fn(dh::Array{Num,2}, k = 0)
	if k > size(dh,1) || k < 0
		throw(ArgumentError("k"))
	end
	
	R = []	
	P = []
	for i in 1:size(dh,1)
		T = dh2t(dh,i)
		push!(R,T[1:3,1:3])
		push!(P,T[1:3,4])
	end

	F = [Fx, Fy, Fz]
	N = [0,0,0]

	i = size(dh,1)
	while i != k
		F = R[i]*F
		N = R[i]*N + cross(P[i],F)
		i -= 1
	end

	(F,N)
end

function dh2tau(dh::Array{Num,2})
	tau = []

	for i in 1:size(dh,1)
		if dh[i,5] == 1
			push!(tau, dh2fn(dh,i)[2][3])
		end
		if dh[i,5] == 2
			push!(tau, dh2fn(dh,i)[1][3])
		end
	end

	Array{Num,1}(tau)
end

function jacob2tau(jacob::Array{Num,2})
	jacob' * [Fx,Fy,Fz]
end

function tau2jacob(tau::Array{Num,1}, dh::Array{Num,2}, k = size(dh,1))
	if k > size(dh,1) || k < 0
		throw(ArgumentError("k"))
	end

	jacob = zeros(Num,3,3)
	for i in 1:3
		for j in 1:3
			row = tau[i]
	
			row = substitute.(row,Fx=>Int(1==j))
			row = substitute.(row,Fy=>Int(2==j))
			row = substitute.(row,Fz=>Int(3==j))

			jacob[j,i] = row
		end
	end

	if k==size(dh,1)
		return jacob
	end

	R = []	
	for i in size(dh,1):-1:(k+1)
		T = dh2t(dh,i)
		push!(R,T[1:3,1:3])
	end
	for tr in R
		jacob = tr * jacob 
	end

	jacob
end

function dh2vw(dh::Array{Num,2}, k = 0)
	if k > size(dh,1) || k < 0
		throw(ArgumentError("k"))
	end

	R = []	
	P = []
	for i in 1:size(dh,1)
		T = dh2t(dh,i)
		push!(R,T[1:3,1:3]')
		push!(P,T[1:3,4])
	end

	V = Array{Array{Real,1},1}(undef,size(dh,1)+1)
	W = Array{Array{Real,1},1}(undef,size(dh,1)+1)
	W[1] = [0,0,0]
	V[1] = [0,0,0]
		
	for i in 2:size(dh,1)+1
		W[i] = R[i-1]*W[i-1]
		if dh[i-1,5] == 1
			W[i] += [0,0,dt(dh[i-1,4])]
		end

		V[i] = R[i-1]*(V[i-1] + cross(W[i-1],P[i-1]))
		if dh[i-1,5] == 2
			V[i] += [0,0,dt(dh[i-1,3])]
		end
	end

	if k != 0
		return (V[k+1], W[k+1])
	end

	Rmain = dh2t(dh)[1:3,1:3]
	(Rmain*V[end], Rmain*W[end])
end

function solve(mat::Array{Num,2}, dic)
	TT = copy(mat)
	TM = Matrix(I,4,4)
	for el in dic
		TT = substitute.(TT, el)
	end

	return TM*TT
end

function view(mat::Array{Num,2})
	show(stdout,"text/plain",mat)
end

