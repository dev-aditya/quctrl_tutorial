using LinearAlgebra
using Plots
using Random
using Printf
using Optim
using ProgressMeter

# ------------------------------------------------------------------------------
# 1. System Definitions (Spin-J)
# ------------------------------------------------------------------------------

"""
    cg(s, j, m)

Clebsch-Gordan coefficients for spin operators. s = +1 or -1.
"""
function cg(s::Real, j::Real, m::Real)
    return sqrt(j * (j + 1) - m * (m + s))
end

"""
    gen_j_mat(n)

Generate collective spin matrices Jx, Jy, Jz for N particles (Spin J = N/2).
Returns [Id, Jx, Jy, Jz].
"""
function gen_j_mat(n::Int)
    J = n / 2.0
    dim = Int(2 * J + 1)

    # Jz operator (diagonal)
    # Python: np.arange(J, -(J+1), -1) -> [J, J-1, ..., -J]
    # Julia: We can construct it similarly.
    mz_vals = [J - i for i = 0:(dim-1)]
    Jz = Diagonal(ComplexF64.(mz_vals))

    # J+ and J- operators
    # Note: Julia indices are 1-based.
    # We'll construct sparse-like dense matrices or just dense. 
    # The dimension is small enough for dense.

    Jplus = zeros(ComplexF64, dim, dim)
    Jminus = zeros(ComplexF64, dim, dim)

    # Basis |J, m>: index i corresponds to m -> starts from m=J (index 1) to m=-J (index dim)
    # <m'|J+|m> is non-zero if m' = m+1.
    # In our ordering (decreasing m), index i (m) goes to index i-1 (m+1).
    # So J+ has entries on the super-diagonal (like in the Python script: diag(..., 1)).

    for i = 1:(dim-1)
        # m corresponds to the column index i+1 (which is m_val at i+1)
        # target m' is m_val at i
        # m = J - ((i+1) - 1) = J - i
        m = mz_vals[i+1] # This is the 'm' in <m+1|J+|m>

        c_plus = cg(1, J, m)
        Jplus[i, i+1] = c_plus

        # J- is just Hermitian conjugate of J+, but let's implement explicit logic if needed.
        # Python script: Jm = np.diag(Jm_vec[1:], -1)
        # m for J- is <m-1|J-|m>
        # m corresponds to index i. Target is i+1.
        m_curr = mz_vals[i]
        c_minus = cg(-1, J, m_curr)
        Jminus[i+1, i] = c_minus
    end

    # Jx and Jy
    Jx = 0.5 * (Jplus + Jminus)
    Jy = -0.5im * (Jplus - Jminus)

    Id = Matrix{ComplexF64}(I, dim, dim)

    return Id, Jx, Jy, Jz
end

# ------------------------------------------------------------------------------
# 2. Parameters & Hamiltonian
# ------------------------------------------------------------------------------

# System parameters
const N_PARTICLES = 4
const J_SPIN = N_PARTICLES / 2
const DIM = Int(N_PARTICLES + 1)
const ID_DIM = Matrix{ComplexF64}(I, DIM, DIM)

# Generate operators
const _Id, _Jx, _Jy, _Jz = gen_j_mat(N_PARTICLES)
const Jz2 = _Jz * _Jz
const Jx2 = _Jx * _Jx

# Control parameters
const OMEGA = 3.0    # Driving strength
const BETA_MAX = 1.0 # Interaction strength

"""
    hamiltonian(x1, x2)

Full Hamiltonian: H = ω[cos(x1)Jx + cos(x2)Jy] + β_max*Jz²
"""
function hamiltonian(x1::Real, x2::Real)
    # H1 = OMEGA * (cos(x1)*Jx + cos(x2)*Jy)
    # H2 = BETA_MAX * Jz^2
    return OMEGA * (cos(x1) * _Jx + cos(x2) * _Jy) + BETA_MAX * Jz2
end

"""
    dH_dx(x1, x2)

Returns derivatives of H w.r.t x1 and x2:
dH/dx1 = -ω sin(x1) Jx
dH/dx2 = -ω sin(x2) Jy
"""
function dH_dx(x1::Real, x2::Real)
    # dH/dx1 = -OMEGA * sin(x1) * Jx
    # dH/dx2 = -OMEGA * sin(x2) * Jy
    dH_d1 = -OMEGA * sin(x1) * _Jx
    dH_d2 = -OMEGA * sin(x2) * _Jy
    return dH_d1, dH_d2
end

# ------------------------------------------------------------------------------
# 3. GRAPE / Time Evolution Utils
# ------------------------------------------------------------------------------

"""
    d_expm_dx(A, E; method=:block)

Frechet derivative of exp(A) in direction E.
"""
function d_expm_dx(A::AbstractMatrix, E::AbstractMatrix; method::Symbol = :block)
    n = size(A, 1)
    if method == :block
        # Higham's method: exp([A E; 0 A]) = [exp(A) L(A,E); 0 exp(A)]
        M = [A E; zeros(eltype(A), n, n) A]
        expM = exp(M)
        return expM[1:n, (n+1):2n]
    else
        error("Method $method not implemented.")
    end
end

"""
    step_propagator(H, dt)
"""
function step_propagator(H::AbstractMatrix, dt::Real)
    return exp(-im * H * dt)
end

"""
    forward_propagators(x, dt)

Forward propagate state. x is vector of size 2*Nts (concatenated controls).
Returns list of unitary propagators U_k = exp(-i Hk dt).
"""
function forward_propagators(x::Vector{Float64}, dt::Real)
    Nts = length(x) ÷ 2
    # x contains [x1_0...x1_N, x2_0...x2_N]

    U_total = Matrix(_Id)
    propagators = Vector{Matrix{ComplexF64}}(undef, Nts)

    for k = 1:Nts
        x1_k = x[k]
        x2_k = x[Nts+k]

        H = hamiltonian(x1_k, x2_k)
        Uk = step_propagator(H, dt)

        # Accumulate U_total = Uk * ... * U1
        U_total = Uk * U_total
        propagators[k] = U_total
    end
    return propagators
end

# ------------------------------------------------------------------------------
# 4. Cost and Gradient Functions
# ------------------------------------------------------------------------------

"""
    calculate_cost(x, dt, psi0, psi_target)

Cost = 1 - |<psi_target | U(T) | psi0>|^2
"""
function calculate_cost(
    x::Vector{Float64},
    dt::Real,
    psi0::Vector{ComplexF64},
    psi_target::Vector{ComplexF64},
)
    Nts = length(x) ÷ 2
    U_total = Matrix(_Id)

    for k = 1:Nts
        x1_k = x[k]
        x2_k = x[Nts+k]
        H = hamiltonian(x1_k, x2_k)
        # Using simple exp here for cost calculation (no gradient overhead)
        Uk = exp(-im * H * dt)
        U_total = Uk * U_total
    end

    psi_final = U_total * psi0
    overlap = dot(psi_target, psi_final) # <target|final>
    infidelity = 1.0 - abs2(overlap)
    return infidelity
end

"""
    calculate_gradient!(G, x, dt, psi0, psi_target)

Fills gradient G in-place using analytic GRAPE gradients.
"""
function calculate_gradient!(
    G::Vector{Float64},
    x::Vector{Float64},
    dt::Real,
    psi0::Vector{ComplexF64},
    psi_target::Vector{ComplexF64},
)
    Nts = length(x) ÷ 2

    # Forward pass: store full propagators U_k (cumulative) OR just step propagators?
    # The reference implementation stores cumulative propagators.
    # Let's align with reference: propagators[k] = U_k ... U_1

    propagators = forward_propagators(x, dt)
    U_final = propagators[end]

    # Target overlap term: z = <psi_target | U_final | psi0>
    psi_f = U_final * psi0
    z = dot(psi_target, psi_f) # <target|final>

    # We want dJ/dx_k. J = 1 - |z|^2 = 1 - z* z.
    # dJ/dx = - (dz/dx * z_conj + z * dz_conj/dx) = -2 Real(dz/dx * z_conj)

    # dz/dx_k = <psi_target | dU_final/dx_k | psi0>
    # U_final = U_N ... U_k ... U_1
    # dU_final/dx_k = U_N ... dU_k/dx_k ... U_1

    # Let U_forward = U_{k-1} ... U_1 (Identity if k=1)
    # Let U_backward = U_final * U_k^dagger ( = U_N ... U_{k+1})

    for k = 1:Nts
        x1_k = x[k]
        x2_k = x[Nts+k]

        # Calculate dU_k / dx1_k and dU_k / dx2_k
        H = hamiltonian(x1_k, x2_k)
        dH_d1, dH_d2 = dH_dx(x1_k, x2_k)

        # dU/dx = d/dx exp(-i H dt)
        #       = d_expm_dx(-i H dt, -i dH/dx dt)

        dU_d1 = d_expm_dx(-im*H*dt, -im*dH_d1*dt; method = :block)
        dU_d2 = d_expm_dx(-im*H*dt, -im*dH_d2*dt; method = :block)

        # Get surrounding propagators
        if k == 1
            U_fwd = Matrix(_Id)
        else
            U_fwd = propagators[k-1]
        end

        # U_bwd = U_total * inv(U_upto_k)
        # U_upto_k = propagators[k]
        # Since unitary, inv = adjoint
        U_bwd = U_final * adjoint(propagators[k])

        # Full chain rule for overlap derivative
        dz_dx1 = dot(psi_target, U_bwd * dU_d1 * U_fwd * psi0)
        dz_dx2 = dot(psi_target, U_bwd * dU_d2 * U_fwd * psi0)

        # Gradient entries
        G[k] = -2 * real(dz_dx1 * conj(z))
        G[Nts+k] = -2 * real(dz_dx2 * conj(z))
    end
end

# ------------------------------------------------------------------------------
# 5. Main Script
# ------------------------------------------------------------------------------

function main()
    println("=== Spin-J State Transfer Optimization (Julia) ===")
    println("System: N = $N_PARTICLES particles, J = $J_SPIN")
    println("Control: Two-axis driving (Jx, Jy) + Jz^2 interaction")

    # Setup States
    # Initial: |J, J> (all up). Corresponds to index 1 in our basis.
    psi0 = zeros(ComplexF64, DIM)
    psi0[1] = 1.0

    # Target: |J, J-k>. k=1 -> one excitation. Index 2.
    k_exc = 1
    psi_target = zeros(ComplexF64, DIM)
    psi_target[1+k_exc] = 1.0 # Index is 1-based

    println("Target State: Dicke[$k_exc] (Index $(1+k_exc))")

    # Time parameters (same as Python)
    Tf_base = 2 * pi / BETA_MAX
    # Python: pasos = np.linspace(0.01, 1.5, 15)
    # Let's pick a few points or just the best one if we want speed.
    # The user asked to "translate", so let's do the sweep.
    pasos = collect(range(0.01, 1.5, length = 15))

    # Python: Nts = 15.
    # NOTE: The Python script used Nts=15 for the optimization grid. 
    # That is quite coarse, but we'll stick to it to replicate results first.
    Nts = 15
    Num_tries = 3 # Increased slightly from 1 to ensure we find good minima

    # Storage
    best_cost_overall = 1.0
    best_x_overall = nothing
    best_time_overall = 0.0

    cost_history = zeros(length(pasos))

    # Progress Bar
    pbar = Progress(Num_tries * length(pasos), desc = "Optimizing...")

    # Loop
    for (i_t, p_val) in enumerate(reverse(pasos))
        T_total = p_val * Tf_base
        dt = T_total / Nts

        # We need a closure for Optim
        # But wait, we iterate tries first usually? 
        # Python script iterates tries then steps backwards through time.
        # We can implement similar logic.

        current_best_for_time = 1.0

        for attempt = 1:Num_tries
            # Initial guess
            # x has size 2 * Nts
            x0 = 2 * pi * (rand(2 * Nts) .- 0.5)

            # functions for Optim
            f_obj = (x) -> calculate_cost(x, dt, psi0, psi_target)
            g_obj! = (G, x) -> calculate_gradient!(G, x, dt, psi0, psi_target)

            # Optimize
            # LBFGS is generally good.
            res = optimize(
                f_obj,
                g_obj!,
                x0,
                LBFGS(),
                Optim.Options(g_tol = 1e-4, iterations = 1000, store_trace = false),
            )

            final_cost = Optim.minimum(res)
            final_x = Optim.minimizer(res)

            if final_cost < current_best_for_time
                current_best_for_time = final_cost
            end

            if final_cost < best_cost_overall
                best_cost_overall = final_cost
                best_x_overall = final_x
                best_time_overall = T_total
            end

            next!(pbar)
        end
        # Store for plotting (taking the best of the tries for this time point)
        # Note: We are iterating in reverse, so index is length - i_t + 1
        idx_store = length(pasos) - i_t + 1
        cost_history[idx_store] = current_best_for_time
    end

    finish!(pbar)

    println("\nOptimization completed.")
    println("Best Fidelity: $(1.0 - best_cost_overall)")
    println(
        "Best Time: $(best_time_overall) (approx $(best_time_overall/Tf_base) * 2pi/beta)",
    )

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------
    # Plot Cost vs Time
    plt = plot(
        pasos,
        cost_history,
        xlabel = "Time (units of 2π/β)",
        ylabel = "Infidelity",
        title = "Spin-J ($N_PARTICLES) State Transfer (k=$k_exc)",
        marker = :circle,
        yscale = :log10,
        label = "Best Found",
    )

    # Display plot if possible, or save it
    display(plt)
    # savefig("spinj_optimization.png")
end

main()
