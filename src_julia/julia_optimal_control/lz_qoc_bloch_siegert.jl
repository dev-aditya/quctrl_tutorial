using LinearAlgebra
using Plots
using Random
using Printf
using Optim
using ProgressMeter

const DIMS = 2

const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 -im; im 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]
const ID2 = I(2)
const H_CTRL = SIGMA_Z
const Delta = 1.0
#
# Bloch–Siegert shift (beyond the RWA)
# -----------------------------------
# Keeping the counter-rotating term of a near-resonant drive produces an
# amplitude-dependent frequency shift δω_BS(t) that is *quadratic* in the drive
# amplitude, δω_BS ∝ Ω(t)^2 / ω0 (up to convention-dependent factors).
#
# In the common NMR rotating-frame convention one writes
#   H = Δ σ^z + Ω(t) σ^x    (RWA),
# and the Bloch–Siegert shift appears as an extra detuning term δω_BS(t) σ^z.
# Our Landau–Zener convention in this script is
#   H = Δ σ^x + ν(t) σ^z,
# i.e. the role of x and z is swapped, so the same correction is represented as
# a quadratic term along σ^x.
#
# `OMEGA0` plays the role of the drift frequency (omega0) that suppresses the
# Bloch–Siegert effect; tune it to match your platform / units. Set
# `BS_COEFF = 0.0` to recover the original (pure-RWA) model.
const OMEGA0 = Delta
const BS_COEFF = 1.0 / (4 * OMEGA0)
const H_BS = SIGMA_X
const T0 = π / (2 * Delta)  # unit of time
const T = pi / Delta        # total evolution time (unused but kept for reference)
SHOW_TRACE = false  # set to true to see optimization trace
# -----------------------
# Optimization parameters
# -----------------------
grad_tol = 1e-8         # tolerance for gradient
Nts = 100          # number of time-steps
Nattempts = 10            # number of random initializations

# -------------------------------
# Time evolution parameters
# -------------------------------
Tfs = collect(range(0.1, 10.0; length = 10))   # Python: np.linspace(0.1, 2, 35)
#Tfs = [3.0]                               # run like this to see the actual fields
fide_opt = zeros(length(Tfs))
dt = 2.0 / Nts                           # ensure Float64 like Python's 2/Nts

# -----------------------
# Initial and target state parameters
# -----------------------
nu0 = 2.0
Delta_eff_nu0 = Delta + BS_COEFF * nu0^2       # Bloch–Siegert-renormalized gap at |ν| = ν0
theta0 = atan(-Delta_eff_nu0 / nu0)
thetaf = π - theta0

psi0 = [cos(theta0 / 2), sin(theta0 / 2)]
psiG = [cos(thetaf / 2), sin(thetaf / 2)]

# -----------------------
# Initial guess for the field
# -----------------------

guess = "random"  # "random" or "zero"

"""
Landau-Zener Hamiltonian for control value `z`.
"""
function landau_zener_hamiltonian(z::Real)::Matrix{ComplexF64}
    # Bloch–Siegert correction: Δ → Δ + δω_BS(z), with δω_BS ∝ z^2.
    Delta_eff = Delta + BS_COEFF * z^2
    return Delta_eff * SIGMA_X + z * SIGMA_Z
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
    U = Matrix{ComplexF64}(ID2)
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
    U_total = propagators[end]
    z = ψ_target' * (U_total * ψ0)

    grad = zeros(Float64, n_steps)
    for j = 1:n_steps
        hj = landau_zener_hamiltonian(controls[j])
        # Because H(z) is no longer linear in z, we must use dH/dz at each slice:
        #   H(z) = (Δ + BS_COEFF*z^2) σ^x + z σ^z  ⇒  dH/dz = σ^z + 2*BS_COEFF*z σ^x.
        dH_dz = H_CTRL + (2 * BS_COEFF * controls[j]) * H_BS
        duj_dcj = d_expm_dx(-im * hj * dt, -im * dH_dz * dt; method = :block)

        # Forward and backward pieces around the j-th control slice
        U_forward = (j == 1) ? Matrix(ID2) : propagators[j-1]
        U_j_plus_1 = propagators[j]
        U_backward = U_total * adjoint(U_j_plus_1)

        dU_dcj = U_backward * duj_dcj * U_forward
        dz_dcj = ψ_target' * (dU_dcj * ψ0)

        grad[j] = -2 * real(dz_dcj * conj(z))
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

# Helpers defined above:
# control_cost(x, ψ0, ψG, dt)::Float64
# control_gradient!(G, x, ψ0, ψG, dt) fills gradient in-place
# (and globals/closures: Delta, nu0, T0, Tfs, Nts, Nattempts, grad_tol, guess, etc.)


start_time = time()
controls_opt = nothing      # best controls from previous Tf in the sweep
initial_controls = nothing  # store first initialization for plotting when Tfs has length 1
final_fide = NaN            # will be updated inside the optimization loops
t_last = nothing            # keep last time grid for plotting

const_crit = Delta^2 / nu0
c_const = 1.0  # if set to 0.0, there is no constraint

# Storage for multiple attempts: (length(Tfs) × Nattempts)
fide_attempts = zeros(length(Tfs), Nattempts)

progress = Progress(Nattempts * length(Tfs); desc = "Optimizing", dt = 0.5)

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
        global t_last = t

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
                global initial_controls = controls_0
            end
        else
            controls_0 = controls_opt
        end

        # Run optimization with or without box constraints
        if c_const != 0.0
            lower = fill(-c_const, Nts)
            upper = fill(+c_const, Nts)
            inner = LBFGS()
            opts = Optim.Options(g_tol = grad_tol, show_trace = SHOW_TRACE)
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
            opts = Optim.Options(g_tol = grad_tol, show_trace = SHOW_TRACE)
            res = optimize(cost_fun, grad_fun!, controls_0, LBFGS(), opts)
        end

        # Store results
        global controls_opt = Optim.minimizer(res)
        global final_fide = Optim.minimum(res)

        fide_attempts[idx_T, attempt] = final_fide

        next!(
            progress;
            showvalues = [(:attempt, attempt), (:Tf, Tf / T0), (:cost, final_fide)],
        )
    end
end

finish!(progress)

# For each value of T, pick the best fidelity out of all attempts
fide_opt = [minimum(view(fide_attempts, mT, :)) for mT = 1:length(Tfs)]

# Plotting parameters (colors are arbitrary; Plots.jl doesn't use Matplotlib's "tab:orange" names)
colore = [:orange, :blue]
lab = ["zero", "random"]
ind = 2  # Julia is 1-based; pick 1 or 2

if length(Tfs) == 1
    cost_x0 = control_cost(initial_controls, psi0, psiG, dt)
    p = plot(
        size = (500, 300),
        title = @sprintf("nu0 = %.2f, cost0=%.5f, costF=%.2E", nu0, cost_x0, final_fide),
        titlefontsize = 8,
    )

    # "stairs": use step line types; x for stairs is edges (length Nts+1), y is values (length Nts)
    edges = t_last ./ (2 * T0)

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
        abs.(fide_opt),
        yscale = :log10,
        marker = :circle,
        linewidth = 2,
        grid = true,
        color = colore[ind],
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

@info "Finished" runtime = time() - start_time
