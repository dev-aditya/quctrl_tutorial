using LinearAlgebra
using Plots
using Optim
using Random

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

function step_propagator(
    H::AbstractMatrix{ComplexF64},
    dt::Real,
)::AbstractMatrix{ComplexF64}
    return exp(-im * H * dt)
end

function gen_total_propagator(
    z_controls::Vector{Float64},
    dt::Real,
)::AbstractMatrix{ComplexF64}
    U = I(DIMS)
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

function fidelity(ψf::AbstractVector, ψ0::AbstractVector)::Float64
    if norm(ψf) ≈ 0.0 || norm(ψ0) ≈ 0.0
        error("Input states must be non-zero vectors.")
    elseif length(ψf) != length(ψ0)
        error("Input states must have the same dimension.")
    elseif !isapprox(norm(ψf), 1.0; atol = 1e-8) || !isapprox(norm(ψ0), 1.0; atol = 1e-8)
        @warn(
            "Input states are not normalized. fidelity calculated with unnormalized states."
        )
    end
    return abs(ψf' * ψ0)^2
end

function cost_function(
    propagator_list::Vector{AbstractMatrix{ComplexF64}},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
    dt::Real,
)::Float64
    U_T = propagator_list[end]
    ψf = U_T * ψ0
    return 1.0 - fidelity(ψf, ψ_target)
end

"""
Frechet derivative of exp at A applied to direction E.

Returns L = d/dt exp(A + tE)|_{t=0}.
"""
function d_expm_dx(A::AbstractMatrix, E::AbstractMatrix; method::Symbol = :block)
    if size(A) != size(E)
		error("Input matrices A and E must have the same dimensions.")
	end
    n = size(A, 1)

    if method == :block
        Z = zeros(eltype(A + E), n, n)
        M = [
            A E;
            Z A
        ]
        EM = exp(M)              # matrix exponential in Julia 
        return @view EM[1:n, (n+1):2n]
    elseif method == :eig
        # Works best when A is diagonalizable and (numerically) normal/Hermitian-like.
        F = eigen(A)             # LinearAlgebra eigen-decomposition 
        vals, vecs = F.values, F.vectors
        Erb = adjoint(vecs) * E * vecs

        G = similar(Erb)
        @inbounds for i in eachindex(vals), j in eachindex(vals)
            λi, λj = vals[i], vals[j]
            if abs(λi - λj) < 1e-12
                G[i, j] = Erb[i, j] * exp(λi)
            else
                G[i, j] = Erb[i, j] * (exp(λi) - exp(λj)) / (λi - λj)
            end
        end
        return vecs * G * adjoint(vecs)
    else
        error("Unknown method=$method. Use :block or :eig.")
    end
end

function grad_cost_function(
    z_controls::Vector{Float64},
    propagator_list::Vector{AbstractMatrix{ComplexF64}},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
    dt::Real,
)::Vector{Float64}
    n_steps = length(z_controls)
    U_T = propagator_list[end]
    ψf = U_T * ψ0
    z = ψ_target' * ψf
    grad_j = zeros(Float64, n_steps)
    for j = 1:n_steps
        hj = hamil_landau_zener(z_controls[j])
        duj_dcj = d_expm_dx(-im * hj * dt, -im * H_ctrl * dt, method = :block)
        dz_dcj = ψ0
		for k = 1:(j - 1)
			dz_dcj = propagator_list[k] * dz_dcj
		end
		dz_dcj = duj_dcj * dz_dcj
		for k = (j + 1):n_steps
			dz_dcj = propagator_list[k] * dz_dcj
		end
		grad_j[j] = -2 * real(conj(z) * (ψ_target' * dz_dcj))
    end
    return grad_j
end

# -----------------------
# Optimization parameters
# -----------------------
grad_tol   = 1e-6         # tolerance for gradient
Nts        = 10           # number of time-steps
Nattempts  = 5            # number of random initializations

# -------------------------------
# Time evolution parameters
# -------------------------------
# Tfs = collect(range(0.1, 2.0; length=35))   # Python: np.linspace(0.1, 2, 35)
Tfs      = [2.0]                               # run like this to see the actual fields
fide_opt = zeros(length(Tfs))
dt       = 2.0 / Nts                           # ensure Float64 like Python's 2/Nts

# -----------------------
# Initial and target state parameters
# -----------------------
nu0     = 2.0
theta0  = atan(-Delta / nu0)                   # assumes Delta is defined
thetaf  = π - theta0

psi0 = [cos(theta0 / 2), sin(theta0 / 2)]
psiG = [cos(thetaf / 2), sin(thetaf / 2)]

# -----------------------
# Initial guess for the field
# -----------------------

seed = 42
Random.seed!(seed)

using Random
using Printf
using Optim
guess = "random"  # "random" or "zero"

# ---- assumes these exist in your code ----
# cost_function(x)::Float64
# Jgrad(x)::Vector{Float64}   # returns gradient vector
# (and globals/closures: Delta, nu0, T0, Tfs, Nts, Nattempts, grad_tol, guess, etc.)

t_script_0 = time()
field_opt = nothing  # will store the best field from previous T in the sweep

# Constraint parameters
const_crit = Delta^2 / nu0
c_const = 2.0  # if set to 0.0, there is no constraint

# Storage for multiple attempts: (length(Tfs) × Nattempts)
fide_attempts = zeros(length(Tfs), Nattempts)

# Optim.jl expects an in-place gradient function g!(G, x)
function g!(G, x)
    G .= Jgrad(x)
    return nothing
end

for nA in 1:Nattempts
    # loop Tfs in reverse: longest time first
    for mT in length(Tfs):-1:1
        Tf = Tfs[mT] * T0
        t = collect(range(0.0, Tf; length = Nts + 1))
        tfield = collect(range(0.0, Tf; length = Nts))
        dt = t[2] - t[1]

        @printf("Run %d of %d, attempt %d of %d\n", mT, length(Tfs), nA, Nattempts)

        # Initialize field for first (longest) T value, else warm-start from previous optimum
        if mT == length(Tfs)
            if guess == "random"
                field_x0 = 2.0 .* (rand(Nts) .- 0.5)   # in [-1,1) scaled by 2 -> [-1,1) [web:175]
            elseif guess == "zero"
                field_x0 = zeros(Nts)
            else
                error("Unknown guess = $guess")
            end
            cost_x0 = cost_function(field_x0)
        else
            field_x0 = field_opt
        end

        # Run optimization with or without constraints
        if c_const != 0.0
            lower = fill(-c_const, Nts)
            upper = fill(+c_const, Nts)

            inner = LBFGS()
            opts = Optim.Options(g_tol = grad_tol, show_trace = false)  # g_tol is the gradient tolerance [web:167]

            res = optimize(cost_function, g!, lower, upper, field_x0, Fminbox(inner), opts) # box constraints via Fminbox [web:174]
        else
            opts = Optim.Options(g_tol = grad_tol, show_trace = true)   # unconstrained [web:167]
            res = optimize(cost_function, g!, field_x0, LBFGS(), opts)          # unconstrained LBFGS [web:174]
        end

        # Store results
        field_opt = Optim.minimizer(res)
        final_fide = Optim.minimum(res)

        println(final_fide)
        fide_attempts[mT, nA] = final_fide
    end
end

# For each value of T, pick the best fidelity out of all attempts
fide_opt = [minimum(view(fide_attempts, mT, :)) for mT in 1:length(Tfs)]

# Plotting parameters (colors are arbitrary; Plots.jl doesn't use Matplotlib's "tab:orange" names)
colore = [:orange, :blue]
lab    = ["zero", "random"]
ind    = 2  # Julia is 1-based; pick 1 or 2

if length(Tfs) == 1
    p = plot(size=(500, 300),
             title = @sprintf("nu0 = %.2f, cost0=%.5f, costF=%.2E", nu0, cost_x0, final_fide),
             titlefontsize = 8)

    # "stairs": use step line types; x for stairs is edges (length Nts+1), y is values (length Nts)
    edges = t ./ (2 * T0)

    plot!(p, edges[1:end-1], field_x0;
          linetype=:steppost, linestyle=:dash, linewidth=1.5,
          color=:gray, label="Initial")  # steppost is one standard step type [web:187][web:189]

    plot!(p, edges[1:end-1], field_opt;
          linetype=:steppost, linewidth=2,
          color=colore[ind], label="Optimized") [web:187][web:189]

    ylims!(p, (-2, 2))
    xlabel!(p, "Time tΔ/π")
    ylabel!(p, "Field α(t)")
    display(p)

    # Optional saves (data + plot)
    # writedlm("Data_Plots_QOC/Fig1-A/Time.txt", t)
    # writedlm("Data_Plots_QOC/Fig1-A/field_ini_nu0_$(Int(nu0))_$(guess).txt", field_x0)
    # writedlm("Data_Plots_QOC/Fig1-A/field_opt_nu0_$(Int(nu0))_$(guess).txt", field_opt)
    # savefig(p, "plots/LZ_fields_nu0$(Int(nu0))_alfa0_$(lab[ind]).svg")

elseif length(Tfs) > 1
    p = plot(Tfs, fide_opt;
             yscale=:log10, marker=:circle, linewidth=2,
             label=@sprintf("nu0=%.2f", nu0),
             size=(500, 300))  # semilogy equivalent via yscale=:log10 [web:195][web:192]

    xlabel!(p, "Evolution time T/T0")
    ylabel!(p, "Optimized cost J(α_opt)")
    display(p)

    # Optional saves
    # writedlm(@sprintf("data/LZ_final_cost_nu%.2f_const%.0f_M%d.txt", nu0, c_const, Nts), fide_opt)
    # writedlm("data/LZ_Tfs.txt", Tfs)
end