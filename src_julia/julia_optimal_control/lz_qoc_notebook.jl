using LinearAlgebra
using Plots
using Random
using Printf
using Optim

const DIMS = 2

const SIGMA_X = [0 1; 1 0]
const SIGMA_Y = [0 -im; im 0]
const SIGMA_Z = [1 0; 0 -1]
const ID2 = I(2)
const H_CTRL = SIGMA_Z
const Delta = 1.0
const T0 = π / (2 * Delta)  # unit of time
const T = pi / Delta        # total evolution time (unused but kept for reference)

"""
Landau–Zener Hamiltonian for control value `z`.
"""
function landau_zener_hamiltonian(z::Real)::Matrix{ComplexF64}
    return Delta * SIGMA_X + z * SIGMA_Z
end

"""
Single-step propagator for Hamiltonian `H` and time step `dt`.
"""
function step_propagator(H::AbstractMatrix{ComplexF64}, dt::Real)::Matrix{ComplexF64}
    return exp(-im * H * dt)
end

"""
Forward-propagate all steps and return the cumulative propagator after each control.
Result: propagator_list[k] == U(t_k) for step k.
"""
function forward_propagators(
    controls::AbstractVector{<:Real},
    dt::Real,
)::Vector{Matrix{ComplexF64}}
    U = Matrix(ID2)
    propagators = Vector{Matrix{ComplexF64}}(undef, length(controls))
    for idx in eachindex(controls)
        H = landau_zener_hamiltonian(controls[idx])
        U = step_propagator(H, dt) * U
        propagators[idx] = U
    end
    return propagators
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

"""
Cost from a list of propagators (1 - fidelity between evolved `ψ0` and `ψ_target`).
"""
function cost_from_propagators(
    propagators::Vector{Matrix{ComplexF64}},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
)::Float64
    ψf = propagators[end] * ψ0
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

function control_gradient_from_propagators(
    controls::AbstractVector{<:Real},
    propagators::Vector{Matrix{ComplexF64}},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
    dt::Real,
)::Vector{Float64}
    n_steps = length(controls)
    ψf = propagators[end] * ψ0
    z = ψ_target' * ψf
    grad = zeros(Float64, n_steps)
    for j = 1:n_steps
        hj = landau_zener_hamiltonian(controls[j])
        duj_dcj = d_expm_dx(-im * hj * dt, -im * H_CTRL * dt, method = :block)
        ∂ψ = ψ0
        for k = 1:(j-1)
            ∂ψ = propagators[k] * ∂ψ
        end
        ∂ψ = duj_dcj * ∂ψ
        for k = (j+1):n_steps
            ∂ψ = propagators[k] * ∂ψ
        end
        grad[j] = -2 * real(conj(z) * (ψ_target' * ∂ψ))
    end
    return grad
end

"""
Convenience wrappers used by Optim.jl: cost and in-place gradient on control vector.
"""
function control_cost(
    controls::AbstractVector{<:Real},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
    dt::Real,
)::Float64
    propagators = forward_propagators(controls, dt)
    return cost_from_propagators(propagators, ψ0, ψ_target)
end

function control_gradient!(
    grad_out::AbstractVector{<:Real},
    controls::AbstractVector{<:Real},
    ψ0::AbstractVector,
    ψ_target::AbstractVector,
    dt::Real,
)
    propagators = forward_propagators(controls, dt)
    grad_out .= control_gradient_from_propagators(controls, propagators, ψ0, ψ_target, dt)
    return nothing
end

# -----------------------
# Optimization parameters
# -----------------------
grad_tol = 1e-6         # tolerance for gradient
Nts = 10           # number of time-steps
Nattempts = 5            # number of random initializations

# -------------------------------
# Time evolution parameters
# -------------------------------
# Tfs = collect(range(0.1, 2.0; length=35))   # Python: np.linspace(0.1, 2, 35)
Tfs = [2.0]                               # run like this to see the actual fields
fide_opt = zeros(length(Tfs))
dt = 2.0 / Nts                           # ensure Float64 like Python's 2/Nts

# -----------------------
# Initial and target state parameters
# -----------------------
nu0 = 2.0
theta0 = atan(-Delta / nu0)                   # assumes Delta is defined
thetaf = π - theta0

psi0 = [cos(theta0 / 2), sin(theta0 / 2)]
psiG = [cos(thetaf / 2), sin(thetaf / 2)]

# -----------------------
# Initial guess for the field
# -----------------------

guess = "random"  # "random" or "zero"

# Helpers defined above:
# control_cost(x, ψ0, ψG, dt)::Float64
# control_gradient!(G, x, ψ0, ψG, dt) fills gradient in-place
# (and globals/closures: Delta, nu0, T0, Tfs, Nts, Nattempts, grad_tol, guess, etc.)

function run_landau_zener_optimization()
    start_time = time()
    controls_opt = nothing      # best controls from previous Tf in the sweep
    initial_controls = nothing  # store first initialization for plotting when Tfs has length 1

    const_crit = Delta^2 / nu0
    c_const = 2.0  # if set to 0.0, there is no constraint

    # Storage for multiple attempts: (length(Tfs) × Nattempts)
    fide_attempts = zeros(length(Tfs), Nattempts)

    # Warm-up to trigger JIT and cache linear algebra paths
    _warm_x = zeros(Nts)
    _warm_grad = similar(_warm_x)
    control_gradient!(_warm_grad, _warm_x, psi0, psiG, 0.1)
    _ = control_cost(_warm_x, psi0, psiG, 0.1)

    # Run Nattempts with optional warm-starting across different Tf values
    for attempt = 1:Nattempts
        for idx_T = length(Tfs):-1:1  # longest time first
            Tf = Tfs[idx_T] * T0
            t = collect(range(0.0, Tf; length = Nts + 1))
            dt_local = t[2] - t[1]

            @printf(
                "Run %d of %d, attempt %d of %d\n",
                idx_T,
                length(Tfs),
                attempt,
                Nattempts
            )

            # Define objective + gradient closures for this Tf
            cost_fun = x -> control_cost(x, psi0, psiG, dt_local)
            grad_fun! = (G, x) -> control_gradient!(G, x, psi0, psiG, dt_local)

            # Initialize control field
            if idx_T == length(Tfs)
                if guess == "random"
                    controls_0 = 2.0 .* (rand(Nts) .- 0.5)   # uniform in [-1, 1)
                elseif guess == "zero"
                    controls_0 = zeros(Nts)
                else
                    error("Unknown guess = $guess")
                end
                if isnothing(initial_controls)
                    initial_controls = controls_0
                end
            else
                controls_0 = controls_opt
            end

            # Run optimization with or without box constraints
            if c_const != 0.0
                lower = fill(-c_const, Nts)
                upper = fill(+c_const, Nts)
                inner = LBFGS()
                opts = Optim.Options(
                    g_tol = grad_tol,
                    show_trace = false,
                    iterations = 200,
                    time_limit = 30.0,
                )
                res = optimize(
                    cost_fun,
                    grad_fun!,
                    lower,
                    upper,
                    controls_0,
                    Fminbox(inner),
                    opts,
                )
            else
                opts = Optim.Options(
                    g_tol = grad_tol,
                    show_trace = false,
                    iterations = 200,
                    time_limit = 30.0,
                )
                res = optimize(cost_fun, grad_fun!, controls_0, LBFGS(), opts)
            end

            # Store results
            controls_opt = Optim.minimizer(res)
            final_fide = Optim.minimum(res)

            println(final_fide)
            fide_attempts[idx_T, attempt] = final_fide
        end
    end

    # For each value of T, pick the best fidelity out of all attempts
    fide_opt = [minimum(view(fide_attempts, mT, :)) for mT = 1:length(Tfs)]

    # Plotting parameters (colors are arbitrary; Plots.jl doesn't use Matplotlib's "tab:orange" names)
    colore = [:orange, :blue]
    lab = ["zero", "random"]
    ind = 2  # Julia is 1-based; pick 1 or 2

    if length(Tfs) == 1
        p = plot(
            size = (500, 300),
            cost_x0 = control_cost(initial_controls, psi0, psiG, dt),
            title = @sprintf(
                "nu0 = %.2f, cost0=%.5f, costF=%.2E",
                nu0,
                cost_x0,
                final_fide
            ),
            titlefontsize = 8,
        )

        # "stairs": use step line types; x for stairs is edges (length Nts+1), y is values (length Nts)
        edges = t ./ (2 * T0)

        plot!(
            p,
            edges[1:(end-1)],
            initial_controls;
            linetype = :steppost,
            linestyle = :dash,
            linewidth = 1.5,
            color = :gray,
            label = "Initial",
        )

        plot!(
            p,
            edges[1:(end-1)],
            controls_opt;
            linetype = :steppost,
            linewidth = 2,
            color = colore[ind],
            label = "Optimized",
        )

        ylims!(p, (-2, 2))
        xlabel!(p, "Time tΔ/π")
        ylabel!(p, "Field α(t)")
        display(p)

        # Optional saves (data + plot)
        # writedlm("Data_Plots_QOC/Fig1-A/Time.txt", t)
        # writedlm("Data_Plots_QOC/Fig1-A/field_ini_nu0_$(Int(nu0))_$(guess).txt", initial_controls)
        # writedlm("Data_Plots_QOC/Fig1-A/field_opt_nu0_$(Int(nu0))_$(guess).txt", controls_opt)
        # savefig(p, "plots/LZ_fields_nu0$(Int(nu0))_alfa0_$(lab[ind]).svg")

    elseif length(Tfs) > 1
        p = plot(
            Tfs,
            fide_opt;
            yscale = :log10,
            marker = :circle,
            linewidth = 2,
            label = @sprintf("nu0=%.2f", nu0),
            size = (500, 300),
        )

        xlabel!(p, "Evolution time T/T0")
        ylabel!(p, "Optimized cost J(α_opt)")
        display(p)

        # Optional saves
        # writedlm(@sprintf("data/LZ_final_cost_nu%.2f_const%.0f_M%d.txt", nu0, c_const, Nts), fide_opt)
        # writedlm("data/LZ_Tfs.txt", Tfs)
    end

    return (fide_opt = fide_opt, controls_opt = controls_opt, runtime = time() - start_time)
end

# Execute when run as a script
run_landau_zener_optimization()
