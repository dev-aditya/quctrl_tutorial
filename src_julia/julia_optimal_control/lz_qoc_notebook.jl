using LinearAlgebra
using Plots
using Optimization
using SparseArrays
using OptimizationLBFGSB


# Dimensions of hilbert space
DIMS = 2;

σx = [0 1; 1 0];
σy = [0 -im; im 0];
σz = [1 0; 0 -1];
id2 = I(2);

Delta = 1.0;
T0 = π / (2 * Delta);  # unit of time
T = pi / Delta;  # total evolution time
function hamil_landau_zener(z::Real)::AbstractMatrix{ComplexF64}
	return Delta * σx + z * σz
end

H_ctrl = σz;

function step_propagator(H::AbstractMatrix{ComplexF64}, dt::Real)::AbstractMatrix{ComplexF64}
	return exp(-im * H * dt)
end

function gen_total_propagator(z_controls::Vector{Float64}, dt::Real)::AbstractMatrix{ComplexF64}
	U = sparse(I, DIMS, DIMS)
	propagator_list = Vector{AbstractMatrix{ComplexF64}}(undef, length(z_controls))
	for idx_z in eachindex(z_controls)
		z = z_controls[idx_z]
		H = hamil_landau_zener(z)
		U_step = step_propagator(H, dt)
		U = U_step * U
		propagator_list[idx_z] = U
	end
	return propagator_list
end

# Optimization and Evolution Parameters
method_gr = OptimizationLBFGSB.LBFGSB();  # Gradient-based optimization method
tol_grad = 1e-6;  # Tolerance for gradient norm
n_timesteps = 20;  # Number of time steps
n_init_attempts = 5;  # Number of optimization attempts with different initial guesses

function fidelity(ψf::AbstractVector, ψ0::AbstractVector)::Float64
	if norm(ψf) ≈ 0.0 || norm(ψ0) ≈ 0.0
		error("Input states must be non-zero vectors.")
	elseif length(ψf) != length(ψ0)
		error("Input states must have the same dimension.")
	elseif !isapprox(norm(ψf), 1.0; atol = 1e-8) || !isapprox(norm(ψ0), 1.0; atol = 1e-8)
		@warn("Input states are not normalized. fidelity calculated with unnormalized states.")
	end
	return abs(ψf' * ψ0)^2
end

function cost_function(z_controls::Vector{Float64}, ψ0::AbstractVector, ψ_target::AbstractVector, dt::Real)::Float64
	propagator_list = gen_total_propagator(z_controls, dt)
	U_T = propagator_list[end]
	ψf = U_T * ψ0
	return 1.0 - fidelity(ψf, ψ_target)
end

function optimize_control(ψ0::AbstractVector, ψ_target::AbstractVector, T::Real, n_timesteps::Int; n_init_attempts::Int = 5)
	dt = T / n_timesteps
	best_cost = Inf
	best_z_controls = nothing

	for attempt in 1:n_init_attempts
		z_init = randn(n_timesteps)  # Random initial guess
		prob = Optimization.OptimizationProblem((z_controls,) -> cost_function(z_controls, ψ0, ψ_target, dt), z_init)
		result = Optimization.optimize(prob, method_gr, Optim.Options(g_tol = tol_grad))

		if result.minimum < best_cost
			best_cost = result.minimum
			best_z_controls = result.minimizer
		end
	end

	return best_z_controls, best_cost
end


# Initial and Target States
ψ0 = [1.0; 0.0];  # Initial state |
ψ_target = [0.0; 1.0];  # Target state |1>

# Run Optimization
best_z_controls, best_cost = optimize_control(ψ0, ψ_target, T, n_timesteps; n_init_attempts = n_init_attempts)
println("Best cost (infidelity): ", best_cost)
println("Optimal control parameters: ", best_z_controls)
# Visualize the Optimal Control
time_array = range(0, T; length = n_timesteps)
plot(time_array, best_z_controls, xlabel = "Time", ylabel = "Control z(t))", title = "Optimal Control Function", legend = false)
