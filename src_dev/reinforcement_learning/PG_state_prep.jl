using LinearAlgebra
using Random
using Statistics
using Zygote
using Printf
using Plots

# ---------------------------
# Qubit environment (functional)
# ---------------------------

function rl_to_qubit_state(s::AbstractVector{<:Real})
    theta, phi = s
    return ComplexF64[cos(0.5 * theta), exp(1im*phi)*sin(0.5*theta)]
end

function qubit_to_rl_state(psi::AbstractVector{ComplexF64})
    # Remove global phase using the first component.
    alpha = angle(psi[1])
    psi_new = exp(-1im * alpha) * psi
    theta = 2.0 * acos(real(psi_new[1]))
    phi = angle(psi_new[2])
    return [theta, phi]
end

function init_env(n_time_steps::Int)
    delta_t = pi / n_time_steps

    id = ComplexF64[1 0; 0 1]
    sigma_x = ComplexF64[0 1; 1 0]
    sigma_y = ComplexF64[0 -1im; 1im 0]
    sigma_z = ComplexF64[1 0; 0 -1]

    generators = [
        ("I", id),
        ("+X", sigma_x),
        ("+Y", sigma_y),
        ("+Z", sigma_z),
        ("-X", -sigma_x),
        ("-Y", -sigma_y),
        ("-Z", -sigma_z),
    ]

    action_space = [exp(-0.5im * delta_t * g) for (_, g) in generators]
    s_target = [0.0, 0.0]
    psi_target = rl_to_qubit_state(s_target)

    return (
        n_time_steps = n_time_steps,
        delta_t = delta_t,
        action_space = action_space,
        action_names = [name for (name, _) in generators],
        n_actions = length(action_space),
        s_target = s_target,
        psi_target = psi_target,
    )
end

function reset_state(rng::AbstractRNG; random_init::Bool = true)
    if random_init
        theta = pi * rand(rng)
        phi = 2.0 * pi * rand(rng)
    else
        theta = pi
        phi = 0.0
    end
    s = [theta, phi]
    psi = rl_to_qubit_state(s)
    return s, psi
end

function step_env(env, psi::Vector{ComplexF64}, action::Int)
    psi_new = env.action_space[action] * psi
    s_new = qubit_to_rl_state(psi_new)
    reward = abs2(dot(env.psi_target, psi_new))
    return psi_new, s_new, reward
end

# ---------------------------
# Policy network (functional)
# ---------------------------

relu(x) = max.(x, 0)

function logsoftmax(x::AbstractMatrix)
    xmax = maximum(x; dims = 1)
    y = x .- xmax
    lse = log.(sum(exp.(y); dims = 1))
    return y .- lse
end

function init_params(
    rng::AbstractRNG,
    input_dim::Int,
    hidden1::Int,
    hidden2::Int,
    n_actions::Int,
)
    scale = 0.1f0
    W1 = scale .* randn(rng, Float32, hidden1, input_dim)
    b1 = zeros(Float32, hidden1)
    W2 = scale .* randn(rng, Float32, hidden2, hidden1)
    b2 = zeros(Float32, hidden2)
    W3 = scale .* randn(rng, Float32, n_actions, hidden2)
    b3 = zeros(Float32, n_actions)
    return (W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 = W3, b3 = b3)
end

function forward_logits(params, x::AbstractMatrix{Float32})
    z1 = params.W1 * x .+ params.b1
    a1 = relu(z1)
    z2 = params.W2 * a1 .+ params.b2
    a2 = relu(z2)
    return params.W3 * a2 .+ params.b3
end

function policy_logprobs(params, state::AbstractVector{<:Real})
    x = reshape(Float32.(state), :, 1)
    logits = forward_logits(params, x)
    logp = logsoftmax(logits)
    return vec(logp)
end

function policy_logprobs(params, states::AbstractArray{<:Real,3})
    n_mc, t_steps, input_dim = size(states)
    x = permutedims(states, (3, 1, 2))
    x = reshape(x, input_dim, :)
    x = Float32.(x)
    logits = forward_logits(params, x)
    logp = logsoftmax(logits)
    logp = reshape(logp, size(logp, 1), n_mc, t_steps)
    return permutedims(logp, (2, 3, 1))
end

function l2_regularizer(params, lmbda::Float32)
    total = 0.0f0
    for p in Tuple(params)
        total += sum(abs2, p)
    end
    return lmbda * total
end

# ---------------------------
# REINFORCE loss + optimizer
# ---------------------------

function pseudo_loss(params, batch; l2::Float32 = 1.0f-3)
    states, actions, returns = batch
    logp = policy_logprobs(params, states)
    baseline = mean(returns; dims = 1)
    total = 0.0f0
    n_mc, t_steps = size(actions)
    @inbounds for j = 1:n_mc, t = 1:t_steps
        total += logp[j, t, actions[j, t]] * (returns[j, t] - baseline[1, t])
    end
    return -(total / n_mc) + l2_regularizer(params, l2)
end

function zero_like(p)
    return zeros(eltype(p), size(p))
end

function map_params(f, p)
    return NamedTuple{keys(p)}(map(f, Tuple(p)))
end

function map_params(f, p, q)
    return NamedTuple{keys(p)}(map(f, Tuple(p), Tuple(q)))
end

function map_params(f, p, q, r)
    return NamedTuple{keys(p)}(map(f, Tuple(p), Tuple(q), Tuple(r)))
end

function adam_init(
    params;
    beta1::Float32 = 0.9f0,
    beta2::Float32 = 0.999f0,
    eps::Float32 = 1.0f-8,
)
    m = map_params(zero_like, params)
    v = map_params(zero_like, params)
    return (m = m, v = v, t = 0, beta1 = beta1, beta2 = beta2, eps = eps)
end

function adam_update(state, params, grads, lr::Float32)
    t = state.t + 1
    beta1, beta2, eps = state.beta1, state.beta2, state.eps

    m = map_params((m, g) -> beta1 * m + (1 - beta1) * g, state.m, grads)
    v = map_params((v, g) -> beta2 * v + (1 - beta2) * (g .^ 2), state.v, grads)

    mhat = map_params(m -> m ./ (1 - beta1 ^ t), m)
    vhat = map_params(v -> v ./ (1 - beta2 ^ t), v)

    params_new =
        map_params((p, mh, vh) -> p .- lr .* (mh ./ (sqrt.(vh) .+ eps)), params, mhat, vhat)

    return params_new, (m = m, v = v, t = t, beta1 = beta1, beta2 = beta2, eps = eps)
end

# ---------------------------
# Training utilities
# ---------------------------

function sample_action(rng::AbstractRNG, probs::AbstractVector{<:Real})
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

function rollout_trajectory!(
    states::Array{Float32,3},
    actions::Array{Int,2},
    returns::Array{Float32,2},
    j::Int,
    env,
    params,
    rng::AbstractRNG;
    random_init::Bool = true,
)
    s, psi = reset_state(rng; random_init = random_init)
    rewards = Vector{Float32}(undef, env.n_time_steps)

    for t = 1:env.n_time_steps
        states[j, t, 1] = Float32(s[1])
        states[j, t, 2] = Float32(s[2])

        logp = policy_logprobs(params, s)
        probs = exp.(logp)
        action = sample_action(rng, probs)
        actions[j, t] = action

        psi, s, reward = step_env(env, psi, action)
        rewards[t] = Float32(reward)
    end

    g = 0.0f0
    for t = env.n_time_steps:-1:1
        g += rewards[t]
        returns[j, t] = g
    end
end

function train_reinforce(;
    seed::Int = 0,
    n_time_steps::Int = 15,
    n_episodes::Int = 401,
    n_mc::Int = 256,
    hidden1::Int = 512,
    hidden2::Int = 256,
    step_size::Float32 = 1.0f-3,
    l2::Float32 = 1.0f-3,
    random_init::Bool = true,
    print_every::Int = 1,
    plot_path::Union{Nothing,String} = "RL_1q_training_curve.pdf",
)
    rng = MersenneTwister(seed)
    env = init_env(n_time_steps)
    params = init_params(rng, 2, hidden1, hidden2, env.n_actions)
    opt_state = adam_init(params)

    states = Array{Float32}(undef, n_mc, env.n_time_steps, 2)
    actions = Array{Int}(undef, n_mc, env.n_time_steps)
    returns = Array{Float32}(undef, n_mc, env.n_time_steps)

    mean_final = zeros(Float32, n_episodes)
    std_final = similar(mean_final)
    min_final = similar(mean_final)
    max_final = similar(mean_final)

    @printf("\nStarting training...\n\n")

    for episode = 1:n_episodes
        start_time = time()

        for j = 1:n_mc
            rollout_trajectory!(
                states,
                actions,
                returns,
                j,
                env,
                params,
                rng;
                random_init = random_init,
            )
        end

        batch = (states, actions, returns)
        grads = Zygote.gradient(p -> pseudo_loss(p, batch; l2 = l2), params)[1]
        params, opt_state = adam_update(opt_state, params, grads, step_size)

        final_rewards = @view returns[:, end]
        mean_final[episode] = mean(final_rewards)
        std_final[episode] = std(final_rewards)
        min_final[episode] = minimum(final_rewards)
        max_final[episode] = maximum(final_rewards)

        if print_every > 0 && (episode - 1) % print_every == 0
            @printf("episode %d in %.2f sec\n", episode - 1, time() - start_time)
            @printf("mean reward: %.4f\n", mean_final[episode])
            @printf("return standard deviation: %.4f\n", std_final[episode])
            @printf(
                "min return: %.4f; max return: %.4f\n\n",
                min_final[episode],
                max_final[episode]
            )
        end
    end

    metrics = (
        mean_final = mean_final,
        std_final = std_final,
        min_final = min_final,
        max_final = max_final,
    )

    if plot_path !== nothing
        plot_learning_curves(metrics; path = plot_path)
    end

    return params, metrics
end

function plot_learning_curves(metrics; path::String = "RL_1q_training_curve.pdf")
    episodes = collect(0:(length(metrics.mean_final)-1))
    p = plot(episodes, metrics.mean_final, label = "mean final reward", color = :black)
    ribbon = 0.5 .* metrics.std_final
    plot!(p, episodes, metrics.mean_final; ribbon = ribbon, fillalpha = 0.25, label = "")
    plot!(
        p,
        episodes,
        metrics.min_final;
        ls = :dash,
        color = :blue,
        label = "min final reward",
    )
    plot!(
        p,
        episodes,
        metrics.max_final;
        ls = :dash,
        color = :red,
        label = "max final reward",
    )
    xlabel!(p, "episode")
    ylabel!(p, "final reward")
    plot!(p; legend = :bottomright)
    grid!(p, true)
    savefig(p, path)
    return p
end

function run_smoke_test()
    # Fast sanity check: small network + few episodes.
    params, metrics = train_reinforce(
        n_episodes = 3,
        n_mc = 32,
        hidden1 = 64,
        hidden2 = 32,
        plot_path = nothing,
        print_every = 1,
    )
    return params, metrics
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_smoke_test()
end
