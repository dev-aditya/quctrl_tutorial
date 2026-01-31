using LinearAlgebra
using Random
using Statistics
using Zygote
using Printf
using Plots

# -----------------------------------------------------------------------------
# Qubit environment for shortcuts to adiabaticity (from PG-STA-2.ipynb)
# -----------------------------------------------------------------------------

mutable struct STAEnv
    n_time_steps::Int
    rng::MersenneTwister
    n_actions::Int
    action_space::Vector{Int}
    actions::Vector{Int}
    T::Float64
    dt::Float64
    Delta::Float64
    h_init::Float64
    h_target::Float64
    delta_h::Float64
    Id::Matrix{ComplexF64}
    sigma_x::Matrix{ComplexF64}
    sigma_y::Matrix{ComplexF64}
    sigma_z::Matrix{ComplexF64}
    psi_init::Vector{ComplexF64}
    psi_target::Vector{ComplexF64}
    state::Vector{Float64}
    psi::Vector{ComplexF64}
    ep_step::Int
    CD_drive::Float64
end

function STAEnv(n_time_steps::Int; seed::Int = 0)
    rng = MersenneTwister(seed)
    max_action = 10
    action_space = collect((-max_action):max_action)
    n_actions = length(action_space)
    actions = collect(1:n_actions)

    T = 1.2
    dt = T / n_time_steps
    Delta = 1.0
    h_init = 2.0
    h_target = -2.0
    delta_h = (h_target - h_init) / (max_action * n_time_steps)

    Id = ComplexF64[1 0; 0 1]
    sigma_x = ComplexF64[0 1; 1 0]
    sigma_y = ComplexF64[0 -1im; 1im 0]
    sigma_z = ComplexF64[1 0; 0 -1]

    H = h -> Delta * sigma_x + h * sigma_z
    E_init, V_init = eigen(H(h_init))
    psi_init = V_init[:, argmin(real.(E_init))]
    E_target, V_target = eigen(H(h_target))
    psi_target = V_target[:, argmin(real.(E_target))]

    state = zeros(Float64, 2)
    env = STAEnv(
        n_time_steps,
        rng,
        n_actions,
        action_space,
        actions,
        T,
        dt,
        Delta,
        h_init,
        h_target,
        delta_h,
        Id,
        sigma_x,
        sigma_y,
        sigma_z,
        psi_init,
        psi_target,
        state,
        psi_init,
        0,
        0.0,
    )
    reset!(env)
    return env
end

function h_t(env::STAEnv, t)
    return (env.h_target - env.h_init) * t / env.T + env.h_init
end

h_t_prime(env::STAEnv) = (env.h_target - env.h_init) / env.T

function compute_gate(env::STAEnv, action_idx::Int)
    drive = h_t(env, env.dt * env.ep_step)
    env.CD_drive += env.delta_h * env.action_space[action_idx]
    norm = env.dt * sqrt(env.Delta^2 + drive^2 + env.CD_drive^2)
    return cos(norm) * env.Id -
           1im * sin(norm) * env.dt / norm *
           (env.Delta * env.sigma_x + drive * env.sigma_z - env.CD_drive * env.sigma_y)
end

function reset!(env::STAEnv)
    env.ep_step = 0
    env.CD_drive =
        0.5 * env.Delta * h_t_prime(env) / (env.Delta^2 + h_t(env, env.dt * env.ep_step)^2)
    env.psi = copy(env.psi_init)
    env.state .= qubit_to_state(env, env.psi)
    return env.state
end

function step!(env::STAEnv, action_idx::Int)
    gate = compute_gate(env, action_idx)
    env.psi = gate * env.psi
    env.state .= qubit_to_state(env, env.psi)
    env.ep_step += 1

    reward =
        env.ep_step == env.n_time_steps ? -log10(1 - abs2(env.psi_target' * env.psi)) : 0.0
    done = false
    return env.state, reward, done
end

function state_to_qubit(env::STAEnv, s::AbstractVector{<:Real})
    theta, phi = s
    return ComplexF64[cos(0.5 * theta), exp(1im*phi)*sin(0.5*theta)]
end

function qubit_to_state(env::STAEnv, psi::AbstractVector{ComplexF64})
    alpha = angle(psi[1])
    psin = exp(-1im * alpha) * psi
    theta = 2 * acos(real(psin[1]))
    phi = angle(psin[2])
    return [theta, phi]
end

# -----------------------------------------------------------------------------
# Policy network: single head (log-softmax over discrete actions)
# -----------------------------------------------------------------------------

relu(x) = max.(x, 0)

function init_params_discrete(rng::AbstractRNG, input_dim::Int, hidden::Int, n_actions::Int)
    scale = 0.1f0
    W1 = scale .* randn(rng, Float32, hidden, input_dim)
    b1 = zeros(Float32, hidden)
    W2 = scale .* randn(rng, Float32, n_actions, hidden)
    b2 = zeros(Float32, n_actions)
    return (W1 = W1, b1 = b1, W2 = W2, b2 = b2)
end

function forward_logits(nn_params, x::AbstractMatrix{Float32})
    h = relu.(nn_params.W1 * x .+ nn_params.b1)
    return nn_params.W2 * h .+ nn_params.b2
end

function logsoftmax(x::AbstractMatrix)
    xmax = maximum(x; dims = 1)
    y = x .- xmax
    lse = log.(sum(exp.(y); dims = 1))
    return y .- lse
end

function policy_logprobs(nn_params, state::AbstractVector{<:Real})
    x = reshape(Float32.(state), :, 1)
    logits = forward_logits(nn_params, x)
    return vec(logsoftmax(logits))
end

function policy_logprobs(nn_params, states::Array{Float32,3})
    n_mc, t_steps, input_dim = size(states)
    x = permutedims(states, (3, 1, 2))
    x = reshape(x, input_dim, :)
    logits = forward_logits(nn_params, x)
    logp = logsoftmax(logits)
    logp = reshape(logp, size(logp, 1), n_mc, t_steps)
    return permutedims(logp, (2, 3, 1))
end

# -----------------------------------------------------------------------------
# Loss (REINFORCE + entropy bonus)
# -----------------------------------------------------------------------------

function l2_regularizer(nn_params, l2::Float32 = 1.0f-3)
    tot = 0.0f0
    for p in Tuple(nn_params)
        tot += sum(abs2, p)
    end
    return l2 * tot
end

function policy_entropy(preds_select, n_actions::Int, betainv = 1e-1)
    ent_max = log(Float32(n_actions))
    ent = -sum(preds_select .* exp.(preds_select); dims = 2) ./ ent_max
    return betainv .* ent
end

function pseudo_loss(nn_params, batch, n_actions; l2::Float32 = 1.0f-3)
    states, actions, returns = batch
    preds = policy_logprobs(nn_params, states)
    baseline = mean(returns; dims = 1)
    total = 0.0f0
    total_ent = 0.0f0
    n_mc, t_steps = size(actions)
    @inbounds for j = 1:n_mc, t = 1:t_steps
        lp = preds[j, t, actions[j, t]]
        adv = returns[j, t] - baseline[1, t]
        total += lp * adv
        total_ent += lp * exp(lp)
    end
    loss_core = -(total / n_mc)
    loss_ent = -(0.1f0 / log(Float32(n_actions))) * (total_ent / n_mc)
    return loss_core + loss_ent + l2_regularizer(nn_params, l2)
end

# -----------------------------------------------------------------------------
# Adam (same helpers as before)
# -----------------------------------------------------------------------------

function zero_like(p)
    ;
    zeros(eltype(p), size(p));
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
    beta1, beta2, eps = state.beta1, state.beta2, state.eps
    m = map_params((m, g) -> beta1 * m + (1 - beta1) * g, state.m, grads)
    v = map_params((v, g) -> beta2 * v + (1 - beta2) * (g .^ 2), state.v, grads)
    mhat = map_params(m -> m ./ (1 - beta1^t), m)
    vhat = map_params(v -> v ./ (1 - beta2^t), v)
    params_new = map_params(
        (p, mh, vh) -> p .- lr .* (mh ./ (sqrt.(vh) .+ eps)),
        nn_params,
        mhat,
        vhat,
    )
    return params_new, (m = m, v = v, t = t, beta1 = beta1, beta2 = beta2, eps = eps)
end

# -----------------------------------------------------------------------------
# Rollout + training
# -----------------------------------------------------------------------------

function sample_action(rng::AbstractRNG, logp::AbstractVector{<:Real})
    probs = exp.(logp)
    r = rand(rng)
    c = 0.0
    @inbounds for (i, p) in enumerate(probs)
        c += p
        if r <= c
            return i
        end
    end
    return length(probs)
end

function rollout!(
    states::Array{Float32,3},
    actions::Array{Int,2},
    returns::Array{Float32,2},
    j::Int,
    env::STAEnv,
    nn_params,
)
    reset!(env)
    rewards = Vector{Float32}(undef, env.n_time_steps)
    for t = 1:env.n_time_steps
        states[j, t, :] = Float32.(env.state)
        logp = policy_logprobs(nn_params, env.state)
        action_idx = sample_action(env.rng, logp)
        actions[j, t] = action_idx
        _, r, _ = step!(env, action_idx)
        rewards[t] = r
    end
    g = 0.0f0
    for t = env.n_time_steps:-1:1
        g += rewards[t]
        returns[j, t] = g
    end
end

function train_sta(;
    seed::Int = 0,
    n_time_steps::Int = 40,
    n_episodes::Int = 200,
    n_mc::Int = 256,
    hidden::Int = 512,
    step_size = 1e-3,
    l2 = 1.0f-3,
    plot_path::Union{Nothing,String} = "RL_STA_training_curve.pdf",
)
    env = STAEnv(n_time_steps; seed = seed)
    rng = env.rng
    nn_params = init_params_discrete(rng, 2, hidden, env.n_actions)
    opt_state = adam_init(nn_params)

    states = Array{Float32}(undef, n_mc, env.n_time_steps, 2)
    actions = Array{Int}(undef, n_mc, env.n_time_steps)
    returns = Array{Float32}(undef, n_mc, env.n_time_steps)

    mean_final = zeros(Float32, n_episodes)
    std_final = similar(mean_final)
    min_final = similar(mean_final)
    max_final = similar(mean_final)

    best_actions = zeros(Int, env.n_time_steps)
    best_return = -Inf

    @printf("\nStarting training (STA env)...\n\n")

    for episode = 1:n_episodes
        start = time()
        for j = 1:n_mc
            rollout!(states, actions, returns, j, env, nn_params)
        end
        batch = (states, actions, returns)
        grads = Zygote.gradient(
            p -> pseudo_loss(p, batch, env.n_actions; l2 = Float32(l2)),
            nn_params,
        )[1]
        nn_params, opt_state = adam_update(opt_state, nn_params, grads, Float32(step_size))

        final_rewards = @view returns[:, end]
        mean_final[episode] = mean(final_rewards)
        std_final[episode] = std(final_rewards)
        min_final[episode] = minimum(final_rewards)
        max_final[episode] = maximum(final_rewards)

        if max_final[episode] > best_return
            best_return = max_final[episode]
            idx = argmax(final_rewards)
            best_actions .= actions[idx, :]
            @printf(
                "New best trajectory (episode %d) return=%.4f\n",
                episode - 1,
                best_return
            )
        end

        @printf(
            "episode %d in %.2f sec | mean R=%.4f | max R=%.4f\n",
            episode - 1,
            time() - start,
            mean_final[episode],
            max_final[episode]
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
        best_actions = best_actions,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_sta(n_episodes = 3, n_mc = 32, hidden = 64, plot_path = nothing)
end
