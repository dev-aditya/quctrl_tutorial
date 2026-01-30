using Random
using Printf

# Simple 1D grid world with two actions: 0 = left, 1 = right
mutable struct SimpleGridWorld
    state::Int
    goal::Int
    width::Int
end

SimpleGridWorld() = SimpleGridWorld(0, 4, 5)

function reset!(env::SimpleGridWorld)
    env.state = 0
    return env.state
end

function step!(env::SimpleGridWorld, action::Int)
    move = action == 1 ? 1 : -1
    env.state = clamp(env.state + move, 0, env.width - 1)

    reward = env.state == env.goal ? 10.0 : -1.0
    done = env.state == env.goal
    return env.state, reward, done
end

# Tabular Q-learning agent
mutable struct QLearningAgent
    q::Matrix{Float64}
    ε::Float64
    ε_decay::Float64
    ε_min::Float64
    α::Float64
    γ::Float64
end

function QLearningAgent(
    states::Int,
    actions::Int;
    α = 0.1,
    γ = 0.9,
    ε = 1.0,
    ε_decay = 0.995,
    ε_min = 0.01,
)
    q = zeros(states, actions)
    return QLearningAgent(q, ε, ε_decay, ε_min, α, γ)
end

# Epsilon–greedy policy
function policy(agent::QLearningAgent, state::Int, rng::AbstractRNG)
    if rand(rng) < agent.ε
        return rand(rng, 0:1)
    else
        return argmax(agent.q[state+1, :]) - 1  # convert 1-based index back to 0/1
    end
end

function learn!(agent::QLearningAgent, s::Int, a::Int, r::Float64, s′::Int)
    # Julia uses 1-based indexing; shift action by +1 when indexing
    aidx = a + 1
    current = agent.q[s+1, aidx]
    best_future = maximum(agent.q[s′+1, :])
    agent.q[s+1, aidx] = current + agent.α * (r + agent.γ * best_future - current)
end

function decay!(agent::QLearningAgent)
    agent.ε = max(agent.ε_min, agent.ε * agent.ε_decay)
end

function run_training(; episodes = 50, rng = Random.default_rng())
    env = SimpleGridWorld()
    agent = QLearningAgent(env.width, 2)

    println("Training Q-learning on a 1D grid (0→4)...")
    for ep = 1:episodes
        state = reset!(env)
        total = 0.0
        done = false

        while !done
            action = policy(agent, state, rng)
            next_state, reward, done = step!(env, action)
            learn!(agent, state, action, reward, next_state)
            state = next_state
            total += reward
        end

        decay!(agent)
        if ep % 10 == 0
            @printf "Episode %3d/%3d  Reward: %6.2f  epsilon: %.3f\n" ep episodes total agent.ε
        end
    end

    println("\nLearned Q-table (rows = states 0..4, cols = [left right]):")
    for s = 0:(env.width-1)
        left = agent.q[s+1, 1]
        right = agent.q[s+1, 2]
        best = right > left ? "RIGHT" : "LEFT"
        @printf "state %d:  %.2f  %.2f   best: %s\n" s left right best
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_training()
end
