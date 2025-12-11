using LinearAlgebra
using Plots
using Optimization
using SparseArrays


# Dimensions of hilbert space
DIMS = 2;

σx = sparse([0 1; 1 0]);
σy = sparse([0 -im; im 0]);
σz = sparse([1 0; 0 -1]);
id2 = sparse(I, 2, 2);

Delta = 1.0;
T0 = π / (2 * Delta);  # unit of time

function hamil_landau_zener(z::Real)::AbstractMatrix{ComplexF64}
	return Delta * σx + z * σz
end

H_ctrl = σz;

function step_propagator(H::AbstractMatrix{ComplexF64}, dt <: Real)::AbstractMatrix{ComplexF64}
	return exp(-im * H * dt)
end

function gen_total_propagator(z_controls::Vector{Float64}, dt <: Real)::AbstractMatrix{ComplexF64}
	U = sparse(I, DIMS, DIMS)
	for z in z_controls
		H = hamil_landau_zener(z)
		U_step = step_propagator(H, dt)
		U = U_step * U
	end
	return U
end

# Optimization and Evolution Parameters
method_gr = Optimization.BFGS();  # Gradient-based optimization method
tol_grad = 1e-6;  # Tolerance for gradient norm
n_timesteps = 20;  # Number of time steps
n_init_attempts = 5;  # Number of optimization attempts with different initial guesses

