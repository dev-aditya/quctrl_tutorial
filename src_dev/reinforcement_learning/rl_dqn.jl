using Random
using Statistics
using Printf
using Flux

# Continuous 1D particle: state = (position, velocity); actions: left, none, right
mutable struct ContinuousWorld
    state::Vector{Float64}
    step::Int
    max_steps::Int
end

ContinuousWorld() = ContinuousWorld([0.0, 0.0], 0, 200)

actions() = (:left, :none, :right)

function reset!(env::ContinuousWorld)
    env.state = [rand(Random.default_rng()) * 4 - 2, 0.0]
    env.step = 0
    return env.state
end

function step!(env::ContinuousWorld, action::Int)
    pos, vel = env.state
    force = action == 1 ? -1.0 : action == 3 ? 1.0 : 0.0

    vel += 0.1 * force
    pos += 0.1 * vel

    pos = clamp(pos, -5.0, 5.0)
    vel = clamp(vel, -2.0, 2.0)

    env.state = [pos, vel]
    env.step += 1

    reward = -abs(pos) - 0.1 * abs(vel)
    done = env.step >= env.max_steps
    return env.state, reward, done
end

# Simple replay buffer implemented with a bounded vector
mutable struct ReplayBuffer
    data::Vector{Tuple{Vector{Float64},Int,Float64,Vector{Float64},Bool}}
    capacity::Int
end

ReplayBuffer(capacity::Int=10_000) = ReplayBuffer(Tuple{Vector{Float64},Int,Float64,Vector{Float64},Bool}[], capacity)

function push!(buf::ReplayBuffer, transition)
    push!(buf.data, transition)
    if length(buf.data) > buf.capacity
        popfirst!(buf.data)
    end
end

function sample(buf::ReplayBuffer, batch::Int, rng::AbstractRNG)
    idx = rand(rng, 1:length(buf.data), batch)
    return buf.data[idx]
end

# DQN agent with a tiny MLP
mutable struct DQNAgent
    net::Chain
    optim::Flux.Optimiser
    γ::Float64
    ε::Float64
    ε_decay::Float64
    ε_min::Float64
    batch::Int
    buffer::ReplayBuffer
end

function DQNAgent(state_size::Int, action_size::Int; γ=0.99, ε=1.0, ε_decay=0.995, ε_min=0.05,
                  batch=64, lr=1e-3, buffer_cap=10_000)
    net = Chain(Dense(state_size, 64, relu), Dense(64, 64, relu), Dense(64, action_size))
    optim = Flux.setup(Adam(lr), net)
    return DQNAgent(net, optim, γ, ε, ε_decay, ε_min, batch, ReplayBuffer(buffer_cap))
end

# Choose action with epsilon–greedy policy; actions are 1..3
function choose_action(agent::DQNAgent, state::Vector{Float64}, rng::AbstractRNG)
    if rand(rng) < agent.ε
        return rand(rng, 1:3)
    else
        q = agent.net(Flux.f32.(state))
        return argmax(q)
    end
end

function learn!(agent::DQNAgent, rng::AbstractRNG)
    buf = agent.buffer
    if length(buf.data) < agent.batch
        return
    end

    batch = sample(buf, agent.batch, rng)
    states  = Flux.f32.(hcat([t[1] for t in batch]...))  # 2 x B
    actions = [t[2] for t in batch]
    rewards = [t[3] for t in batch]
    nexts   = Flux.f32.(hcat([t[4] for t in batch]...))
    dones   = [t[5] for t in batch]

    function loss_fn()
        q_values = agent.net(states)                    # (3, B)
        chosen = Float32.([q_values[a, i] for (i, a) in enumerate(actions)])
        next_q = maximum(agent.net(nexts), dims=1)      # (1, B)
        targets = Float32.([rewards[i] + (dones[i] ? 0.0 : agent.γ * next_q[1, i]) for i in eachindex(rewards)])
        return Flux.Losses.mse(chosen, targets)
    end

    grads = Flux.gradient(loss_fn, Flux.params(agent.net))
    Flux.Optimise.update!(agent.optim, Flux.params(agent.net), grads)
end

function decay!(agent::DQNAgent)
    agent.ε = max(agent.ε_min, agent.ε * agent.ε_decay)
end

function run_dqn(; episodes=300, rng=Random.default_rng())
    env = ContinuousWorld()
    agent = DQNAgent(2, 3)

    println("Training DQN (Flux) to center the particle at zero...")
    for ep in 1:episodes
        state = reset!(env)
        total = 0.0
        done = false

        while !done
            action = choose_action(agent, state, rng)
            next_state, reward, done = step!(env, action)
            push!(agent.buffer, (state, action, reward, next_state, done))
            learn!(agent, rng)
            state = next_state
            total += reward
        end

        decay!(agent)
        if ep % 20 == 0
            @printf "Episode %4d/%4d  Reward: %7.2f  epsilon: %.3f\n" ep episodes total agent.ε
        end
    end

    println("\nGreedy rollout starting at position = 2.0")
    env.state = [2.0, 0.0]
    env.step = 0
    for t in 0:20
        pos, vel = env.state
        action = choose_action(agent, env.state, rng)
        name = (action == 1 ? "LEFT" : action == 2 ? "NONE" : "RIGHT")
        @printf "t=%2d  pos=%5.2f  vel=%5.2f  action=%s\n" t pos vel name
        _, _, done = step!(env, action)
        done && break
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_dqn()
end
