using LinearAlgebra
using Random
using Statistics
using Zygote
using Printf
using Plots

# -----------------------------------------------------------------------------
# Quantum-stochastic qubit + ancilla environment (from PG_quant-env_qubit-3.ipynb)
# -----------------------------------------------------------------------------

mutable struct QuantumEnv
    n_time_steps::Int
    p_ent::Float64
    p_emit::Float64
    rng::MersenneTwister
    psi::Vector{ComplexF64}
    psi_ini::Vector{ComplexF64}
    psi_tgt::Vector{ComplexF64}
    state::Matrix{Float64}          # shape (T, 3)
    env_step::Int
    # operators
    Pz_q::Matrix{ComplexF64}
    Pz_a::Matrix{ComplexF64}
    H_a::Matrix{ComplexF64}
    X_a::Matrix{ComplexF64}
    Rx_q::Function
    Ry_q::Function
    Rz_q::Function
    S_tgt_q::Matrix{ComplexF64}
    E::Matrix{ComplexF64}
end

function QuantumEnv(
    n_time_steps::Int;
    seed::Int = 0,
    p_ent::Float64 = 0.1,
    p_emit::Float64 = 0.05,
)
    rng = MersenneTwister(seed)

    # Pauli matrices
    Id = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -1im; 1im 0]
    Z = ComplexF64[1 0; 0 -1]
    H = (X + Z) / sqrt(2)                  # Hadamard on ancilla
    Pz = (Id - Z) / 2                       # |1⟩ projector

    compute_2q_op = (Q, A) -> kron(Q, A)

    # initial |10⟩ (qubit in |1>, ancilla in |0>)
    psi_ini = ComplexF64[0, 0, 1, 0]

    # target qubit state (theta, phi fixed)
    theta = pi / 4
    phi = pi / 3
    psi_tgt_qubit = ComplexF64[cos(theta / 2), exp(1im*phi)*sin(theta/2)]
    psi_tgt = kron(psi_tgt_qubit, ComplexF64[1, 0])

    # single-qubit ops embedded in 2-qubit space
    X_q = compute_2q_op(X, Id)
    Y_q = compute_2q_op(Y, Id)
    Z_q = compute_2q_op(Z, Id)
    Pz_q = compute_2q_op(Pz, Id)

    # rotations on qubit conditioned on action angles
    Rx_q = angle -> I * cos(angle / 2) - 1im * X_q * sin(angle / 2)
    Ry_q = angle -> I * cos(angle / 2) - 1im * Y_q * sin(angle / 2)
    Rz_q = angle -> I * cos(angle / 2) - 1im * Z_q * sin(angle / 2)

    # ancilla ops
    H_a = compute_2q_op(Id, H)
    X_a = compute_2q_op(Id, X)
    Pz_a = compute_2q_op(Id, Pz)

    # control-S gate with target stabilizer
    S_tgt = sin(theta) * cos(phi) * X + sin(theta) * sin(phi) * Y + cos(theta) * Z
    S_tgt_q = Matrix{ComplexF64}(I, 4, 4)
    S_tgt_q[2, 2] = S_tgt[1, 1];
    S_tgt_q[4, 4] = S_tgt[2, 2]
    S_tgt_q[2, 4] = S_tgt[1, 2];
    S_tgt_q[4, 2] = S_tgt[2, 1]

    # weak entangling noise E
    a, b, c = p_ent .* (rand(rng, 3) .* (2pi) .- pi)
    XX = compute_2q_op(X, X)
    YY = compute_2q_op(Y, Y)
    ZZ = compute_2q_op(Z, Z)
    E =
        (I * cos(a) - 1im * XX * sin(a)) *
        (I * cos(b) - 1im * YY * sin(b)) *
        (I * cos(c) - 1im * ZZ * sin(c))

    state = zeros(Float64, n_time_steps, 3)

    env = QuantumEnv(
        n_time_steps,
        p_ent,
        p_emit,
        rng,
        psi_ini,
        psi_ini,
        psi_tgt,
        state,
        0,
        Pz_q,
        Pz_a,
        H_a,
        X_a,
        Rx_q,
        Ry_q,
        Rz_q,
        S_tgt_q,
        E,
    )
    reset!(env)
    return env
end

function reset!(env::QuantumEnv)
    env.psi .= env.psi_ini
    env.env_step = 0
    env.state .= 0.0
    env.state[1, 1] = 1.0      # time-step indicator one-hot (t=1)
    env.state[:, 2] .= 1.0     # ancilla default measurement outcome
    env.state[:, 3] .= 1.0     # no photon detected yet
    return env.state
end

function measure_ancilla!(env::QuantumEnv)
    prob_gs = real(env.psi' * env.Pz_a * env.psi)
    if rand(env.rng) <= prob_gs
        env.psi = env.Pz_a * env.psi / sqrt(prob_gs)
        return -1.0
    else
        env.psi = (I - env.Pz_a) * env.psi / sqrt(1 - prob_gs)
        return +1.0
    end
end

function apply_ent_noise!(env::QuantumEnv)
    env.psi = env.E * env.psi
end

function apply_spont_emission!(env::QuantumEnv)
    if rand(env.rng) <= env.p_emit
        prob = real(env.psi' * env.Pz_q * env.psi)
        env.psi = env.Pz_q * env.psi / sqrt(prob)
        return -1.0
    else
        return +1.0
    end
end

function step!(env::QuantumEnv, action::NTuple{3,Float64})
    env.env_step += 1
    α, β, γ = action

    # control rotations on qubit
    env.psi = env.Rx_q(α) * env.psi
    env.psi = env.Ry_q(β) * env.psi
    env.psi = env.Rz_q(γ) * env.psi

    apply_ent_noise!(env)
    detected = apply_spont_emission!(env)
    msmt = measure_ancilla!(env)

    if env.env_step == env.n_time_steps
        env.psi = env.H_a * env.psi
        env.psi = env.S_tgt_q * env.psi
        env.psi = env.H_a * env.psi
        reward_msmt = measure_ancilla!(env)
        reward = 0.5 * (1 + reward_msmt)
        done = true
    else
        if msmt < 0
            env.psi = env.X_a * env.psi
        end
        reward = 0.0
        done = false
    end

    env.state .= 0.0
    env.state[env.env_step, 1] = 1.0
    env.state[env.env_step, 2] = msmt
    env.state[env.env_step, 3] = detected

    return env.state, reward, done
end

# -----------------------------------------------------------------------------
# Policy network: shared trunk + (mean, std) heads for 3 continuous actions
# -----------------------------------------------------------------------------

relu(x) = max.(x, 0)
softplus(x) = log1p.(exp.(x))

function init_params_gaussian(rng::AbstractRNG, input_dim::Int, hidden::Int, n_outputs::Int)
    scale = 0.1f0
    W1 = scale .* randn(rng, Float32, hidden, input_dim)
    b1 = zeros(Float32, hidden)
    Wm = scale .* randn(rng, Float32, n_outputs, hidden)
    bm = zeros(Float32, n_outputs)
    Ws = scale .* randn(rng, Float32, n_outputs, hidden)
    bs = zeros(Float32, n_outputs)
    return (W1 = W1, b1 = b1, Wm = Wm, bm = bm, Ws = Ws, bs = bs)
end

function forward_gaussian(nn_params, x::AbstractMatrix{Float32})
    h = relu.(nn_params.W1 * x .+ nn_params.b1)
    means = nn_params.Wm * h .+ nn_params.bm
    stds = softplus.(nn_params.Ws * h .+ nn_params.bs) .+ 1.0f-3
    return means, stds
end

function policy_gaussian(nn_params, state_vec::AbstractVector{<:Real})
    x = reshape(Float32.(state_vec), :, 1)
    return forward_gaussian(nn_params, x)
end

function policy_gaussian(nn_params, states::Array{Float32,3})
    n_mc, t_steps, input_dim = size(states)
    x = permutedims(states, (3, 1, 2))
    x = reshape(x, input_dim, :)
    means, stds = forward_gaussian(nn_params, x)
    means = reshape(means, :, n_mc, t_steps)
    stds = reshape(stds, :, n_mc, t_steps)
    return permutedims(means, (2, 3, 1)), permutedims(stds, (2, 3, 1))
end

# -----------------------------------------------------------------------------
# REINFORCE pseudo-loss (Gaussian log-prob)
# -----------------------------------------------------------------------------

function l2_regularizer(nn_params, λ::Float32)
    tot = 0.0f0
    for p in Tuple(nn_params)
        tot += sum(abs2, p)
    end
    return λ * tot
end

function pseudo_loss(nn_params, batch; λ::Float32 = 1.0f-3)
    states, actions, returns = batch
    means, stds = policy_gaussian(nn_params, states)  # shapes (N, T, 3)
    baseline = mean(returns; dims = 1)
    diff = (actions .- means) ./ stds
    log_pi = .-0.5f0 .* sum(diff .^ 2 .+ 2.0f0 .* log.(stds); dims = 3)
    adv = returns .- baseline
    return -mean(sum(log_pi .* adv; dims = 2)) + l2_regularizer(nn_params, λ)
end

# -----------------------------------------------------------------------------
# Optimizer: simple Adam
# -----------------------------------------------------------------------------

function zero_like(p)
    return zeros(eltype(p), size(p))
end
map_params(f, p) = NamedTuple{keys(p)}(map(f, Tuple(p)))
map_params(f, p, q) = NamedTuple{keys(p)}(map(f, Tuple(p), Tuple(q)))
map_params(f, p, q, r) = NamedTuple{keys(p)}(map(f, Tuple(p), Tuple(q), Tuple(r)))

function adam_init(
    nn_params;
    beta1::Float32 = 0.9f0,
    beta2::Float32 = 0.999f0,
    eps::Float32 = 1.0f-8,
)
    m = map_params(zero_like, nn_params)
    v = map_params(zero_like, nn_params)
    return (m = m, v = v, t = 0, beta1 = beta1, beta2 = beta2, eps = eps)
end

function adam_update(state, nn_params, grads, lr::Float32)
    t = state.t + 1
    β1, β2, ϵ = state.beta1, state.beta2, state.eps
    m = map_params((m, g) -> β1 * m + (1 - β1) * g, state.m, grads)
    v = map_params((v, g) -> β2 * v + (1 - β2) * (g .^ 2), state.v, grads)
    mhat = map_params(m -> m ./ (1 - β1^t), m)
    vhat = map_params(v -> v ./ (1 - β2^t), v)
    params_new = map_params(
        (p, mh, vh) -> p .- lr .* (mh ./ (sqrt.(vh) .+ ϵ)),
        nn_params,
        mhat,
        vhat,
    )
    return params_new, (m = m, v = v, t = t, beta1 = β1, beta2 = β2, eps = ϵ)
end

# -----------------------------------------------------------------------------
# Rollout + training
# -----------------------------------------------------------------------------

function sample_action(
    rng::AbstractRNG,
    μ::AbstractVector{<:Real},
    σ::AbstractVector{<:Real},
)
    @assert length(μ) == 3
    return ntuple(k -> randn(rng) * σ[k] + μ[k], 3)
end

function rollout!(
    states::Array{Float32,3},
    actions::Array{Float32,3},
    returns::Array{Float32,2},
    j::Int,
    env::QuantumEnv,
    nn_params,
)
    reset!(env)
    rewards = Vector{Float32}(undef, env.n_time_steps)
    for t = 1:env.n_time_steps
        s = vec(env.state)
        states[j, t, :] = Float32.(s)
        μ, σ = policy_gaussian(nn_params, s)
        a = sample_action(env.rng, vec(μ), vec(σ))
        actions[j, t, :] = Float32.(collect(a))
        _, r, _ = step!(env, a)
        rewards[t] = r
    end
    g = 0.0f0
    for t = env.n_time_steps:-1:1
        g += rewards[t]
        returns[j, t] = g
    end
end

function train_quantum_env(;
    seed::Int = 0,
    n_time_steps::Int = 5,
    n_episodes::Int = 50,
    n_mc::Int = 256,
    hidden::Int = 8,
    step_size = 1e-2,
    λ = 1.0f-3,
    plot_path::Union{Nothing,String} = "RL_quant_env_training_curve.pdf",
)
    env = QuantumEnv(n_time_steps; seed = seed)
    rng = env.rng
    input_dim = 3 * n_time_steps
    nn_params = init_params_gaussian(rng, input_dim, hidden, 3)
    opt_state = adam_init(nn_params)

    states = Array{Float32}(undef, n_mc, n_time_steps, input_dim)
    actions = Array{Float32}(undef, n_mc, n_time_steps, 3)
    returns = Array{Float32}(undef, n_mc, n_time_steps)

    mean_final = zeros(Float32, n_episodes)
    std_final = similar(mean_final)
    min_final = similar(mean_final)
    max_final = similar(mean_final)
    p_loss = similar(mean_final)

    @printf("\nStarting training (quantum stochastic env)...\n\n")

    for episode = 1:n_episodes
        start = time()
        for j = 1:n_mc
            rollout!(states, actions, returns, j, env, nn_params)
        end
        batch = (states, actions, returns)
        grads = Zygote.gradient(p -> pseudo_loss(p, batch; λ = Float32(λ)), nn_params)[1]
        nn_params, opt_state = adam_update(opt_state, nn_params, grads, Float32(step_size))

        final_rewards = @view returns[:, end]
        mean_final[episode] = mean(final_rewards)
        std_final[episode] = std(final_rewards)
        min_final[episode] = minimum(final_rewards)
        max_final[episode] = maximum(final_rewards)
        p_loss[episode] = pseudo_loss(nn_params, batch; λ = λ)

        @printf(
            "episode %d in %.2f sec | mean R=%.4f | max R=%.4f | loss=%.4f\n",
            episode - 1,
            time() - start,
            mean_final[episode],
            max_final[episode],
            p_loss[episode]
        )
    end

    if plot_path !== nothing
        episodes = collect(0:(n_episodes-1))
        p = plot(episodes, mean_final; label = "mean final reward", color = :black)
        ribbon = 0.5 .* std_final
        plot!(p, episodes, mean_final; ribbon = ribbon, fillalpha = 0.25, label = "")
        plot!(p, episodes, min_final; ls = :dash, color = :blue, label = "min")
        plot!(p, episodes, max_final; ls = :dash, color = :red, label = "max")
        xlabel!(p, "episode");
        ylabel!(p, "final reward");
        legend!(p, :bottomright);
        grid!(p, true)
        savefig(p, plot_path)
    end

    return nn_params,
    (
        mean_final = mean_final,
        std_final = std_final,
        min_final = min_final,
        max_final = max_final,
        loss = p_loss,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_quantum_env(n_episodes = 3, n_mc = 32, hidden = 8, plot_path = nothing)
end
