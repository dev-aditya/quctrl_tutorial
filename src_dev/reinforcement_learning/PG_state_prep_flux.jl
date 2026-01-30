using LinearAlgebra
using Random
using Statistics
using Flux
using Printf

# ---------------------------
# Qubit environment (same physics, lighter code)
# ---------------------------

rl_to_qubit_state(s) = ComplexF64[cos(0.5 * s[1]), exp(1im*s[2])*sin(0.5*s[1])]

function qubit_to_rl_state(psi::AbstractVector{ComplexF64})
    α = angle(psi[1])
    ψ = exp(-1im * α) * psi
    θ = 2.0 * acos(real(ψ[1]))
    ϕ = angle(ψ[2])
    return [θ, ϕ]
end

function init_env(n_time_steps::Int)
    Δt = π / n_time_steps
    id = ComplexF64[1 0; 0 1]
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -1im; 1im 0]
    σz = ComplexF64[1 0; 0 -1]

    gens = [id, σx, σy, σz, -σx, -σy, -σz]
    action_names = ["I", "+X", "+Y", "+Z", "-X", "-Y", "-Z"]
    actions = [exp(-0.5im * Δt * g) for g in gens]

    return (
        actions = actions,
        action_names = action_names,
        n_actions = length(actions),
        n_time_steps = n_time_steps,
        psi_target = rl_to_qubit_state([0.0, 0.0]),
    )
end

function reset_state(rng::AbstractRNG; random_init::Bool = true)
    θ = random_init ? π * rand(rng) : π
    ϕ = random_init ? 2π * rand(rng) : 0.0
    s = [θ, ϕ]
    return s, rl_to_qubit_state(s)
end

function step_env(env, ψ::Vector{ComplexF64}, action::Int)
    ψ′ = env.actions[action] * ψ
    s′ = qubit_to_rl_state(ψ′)
    r = abs2(dot(env.psi_target, ψ′))
    return ψ′, s′, r
end

# ---------------------------
# Policy network (Flux)
# ---------------------------

function build_policy(n_actions::Int)
    return Chain(Dense(2, 128, relu), Dense(128, 64, relu), Dense(64, n_actions))
end

function log_probs(model, states::Array{Float32,3})
    # states: (n_mc, T, 2) -> x: (2, n_mc*T)
    n_mc, T, _ = size(states)
    x = reshape(permutedims(states, (3, 1, 2)), 2, :)
    logits = model(x)                               # (n_actions, batch)
    lps = Flux.logsoftmax(logits; dims = 1)
    lps = reshape(lps, :, n_mc, T)
    return permutedims(lps, (2, 3, 1))              # (n_mc, T, n_actions)
end

sample_action(rng, probs) = something(findfirst(cumsum(probs) .>= rand(rng)), length(probs))

# ---------------------------
# Rollout + REINFORCE loss
# ---------------------------

function rollout!(states, actions, returns, j, env, model, rng; random_init = true)
    s, ψ = reset_state(rng; random_init = random_init)
    rewards = similar(returns, size(returns, 2))

    for t = 1:env.n_time_steps
        states[j, t, 1] = Float32(s[1])
        states[j, t, 2] = Float32(s[2])

        lp = log_probs(model, reshape(states[j:j, t:t, :], 1, 1, :))
        probs = vec(exp.(lp[1, 1, :]))
        a = sample_action(rng, probs)
        actions[j, t] = a

        ψ, s, r = step_env(env, ψ, a)
        rewards[t] = Float32(r)
    end

    g = 0.0f0
    for t = env.n_time_steps:-1:1
        g += rewards[t]
        returns[j, t] = g
    end
end

function reinforce_loss(model, states, actions, returns, ps; l2 = 1.0f-4)
    logp = log_probs(model, states)
    baseline = mean(returns; dims = 1)
    adv = returns .- baseline
    total = 0.0f0
    n_mc, T = size(actions)
    @inbounds for j = 1:n_mc, t = 1:T
        total += logp[j, t, actions[j, t]] * adv[j, t]
    end
    reg = l2 * sum(p -> sum(abs2, p), ps)
    return -(total / n_mc) + reg
end

# ---------------------------
# Training loop
# ---------------------------

function train_pg_flux(;
    seed = 0,
    n_time_steps = 15,
    n_episodes = 200,
    n_mc = 128,
    lr = 1e-3,
    random_init = true,
    print_every = 10,
)
    rng = MersenneTwister(seed)
    env = init_env(n_time_steps)
    model = build_policy(env.n_actions)
    opt_state = Flux.setup(Flux.Optimisers.Adam(Float32(lr)), model)
    ps = Flux.params(model)

    states = Array{Float32}(undef, n_mc, env.n_time_steps, 2)
    actions = Array{Int}(undef, n_mc, env.n_time_steps)
    returns = Array{Float32}(undef, n_mc, env.n_time_steps)

    metrics = Float32[]

    for ep = 1:n_episodes
        for j = 1:n_mc
            rollout!(
                states,
                actions,
                returns,
                j,
                env,
                model,
                rng;
                random_init = random_init,
            )
        end

        loss, back = Flux.withgradient(model) do m
            reinforce_loss(m, states, actions, returns, ps)
        end
        opt_state, model = Flux.Optimisers.update!(opt_state, model, back[1])

        push!(metrics, mean(@view returns[:, end]))
        if print_every > 0 && ep % print_every == 0
            @printf "ep %3d  mean final reward: %.4f  loss: %.4f\n" ep metrics[end] loss
        end
    end

    return model, metrics
end

function run_smoke_test()
    train_pg_flux(n_episodes = 20, n_mc = 64, n_time_steps = 10, lr = 3e-3, print_every = 5)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_smoke_test()
end
