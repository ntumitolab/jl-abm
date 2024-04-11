#===
# Event-driven continuous-time multiagent model

The spatial rock-paper-scissors (RPS) is an ABM with the following rules:

+ Agents can be any of three "kinds": Rock, Paper, or Scissors.
+ Agents live in a 2D periodic grid space allowing only one agent per cell.
+ When an agent activates, it can do one of three actions:
    1. Attack: choose a random nearby agent and attack it. If the agent loses the RPS game it gets removed.
    2. Move: choose a random nearby position. If it is empty move to it, otherwise swap positions with the agent there.
    3. Reproduce: choose a random empty nearby position (if any exist). Generate there a new agent of the same type.
===#

using Agents
using Random
using LinearAlgebra
using Agents.DataFrames
using CairoMakie
CairoMakie.activate!(px_per_unit = 1.0)

# Rock, Paper, or Scissors (RPS) agent
# The agent type could be obtained via `kindof(agent)`
@multiagent struct RPS(GridAgent{2})
    @subagent struct Rock end
    @subagent struct Paper end
    @subagent struct Scissors end
end

# Attack actions
function attack!(agent, model)
    ## Randomly pick a nearby agent
    contender = random_nearby_agent(agent, model)
    ## do nothing if there isn't anyone nearby
    isnothing(contender) && return
    ## else perform standard rock paper scissors logic
    ## and remove the contender if you win.
    ## Remember to compare agents with `kindof` instead of
    ## `typeof` since we use `@multiagent`
    kind = kindof(agent)
    kindc = kindof(contender)
    if kind === :Rock && kindc === :Scissors
        remove_agent!(contender, model)
    elseif kind === :Scissors && kindc === :Paper
        remove_agent!(contender, model)
    elseif kind === :Paper && kindc === :Rock
        remove_agent!(contender, model)
    end
    return
end

# Move actions
# Use `move_agent!` and `swap_agents!` functions
function move!(agent, model)
    rand_pos = random_nearby_position(agent.pos, model)
    if isempty(rand_pos, model)
        move_agent!(agent, rand_pos, model)
    else
        occupant_id = id_in_position(rand_pos, model)
        occupant = model[occupant_id]
        swap_agents!(agent, occupant, model)
    end
    return
end

# Reproduce actions
# Use `replicate!` function
function reproduce!(agent, model)
    pos = random_nearby_position(agent, model, 1, pos -> isempty(pos, model))
    isnothing(pos) && return
    ## pass target position as a keyword argument
    replicate!(agent, model; pos)
    return
end

# Defining the propensity ("rate" in Gillespie stochastic simulations) and timing of the events
attack_propensity = 1.0
movement_propensity = 0.5
reproduction_propensity(agent, model) = cos(abmtime(model))^2

# Register events with `AgentEvent` structures
attack_event = AgentEvent(action! = attack!, propensity = attack_propensity)
reproduction_event = AgentEvent(action! = reproduce!, propensity = reproduction_propensity)

# We want a different distribution other than exponential dist. for movement time
function movement_time(agent, model, propensity)
    ## Make time around 1
    t = 0.1 * randn(abmrng(model)) + 1
    return clamp(t, 0, Inf)
end

# Also the rocks do not move
movement_event = AgentEvent(
    action! = move!, propensity = movement_propensity,
    kinds = (:Scissors, :Paper), timing = movement_time
)

# Those are all the events
events = (attack_event, reproduction_event, movement_event)

# Model factory function
# `EventQueueABM` for an event-driven ABM
function initialize_rps(; n = 100, nx = n, ny = n, seed = 42)
    space = GridSpaceSingle((nx, ny))
    rng = Xoshiro(seed)
    model = EventQueueABM(RPS, events, space; rng, warn = false)
    for p in positions(model)
        ## Randomly assign one of the agent
        type = rand(abmrng(model), (Rock, Paper, Scissors))
        add_agent!(p, type, model)
    end
    return model
end

# Create model
model = initialize_rps()

# Have a look at the event queue
abmqueue(model)

# The time in `EventQueueABM` is continuous, so we can pass real-valued time
step!(model, 123.456)
nagents(model)

# step! also accepts a terminating condition
function terminate(model, t)
    willterm = length(allagents(model)) < 5000
    return willterm || (t > 1000.0)
end

model = initialize_rps()
step!(model, terminate)
abmtime(model)

# ## Data collection
# adata: aggregated data to extract information from the execution stats
# adf: agent data frame
model = initialize_rps()
adata = [(a -> kindof(a) === X, count) for X in allkinds(RPS)]

adf, mdf = run!(model, 100.0; adata, when = 0.5, dt = 0.01)
adf[1:10, :]

# ## Visualize population change
tvec = adf[!, :time]  ## time as x axis
populations = adf[:, Not(:time)]  ## agents as data
alabels = ["rocks", "papers", "scissors"]

fig = Figure();
ax = Axis(fig[1,1]; xlabel = "time", ylabel = "population")
for (i, l) in enumerate(alabels)
    lines!(ax, tvec, populations[!, i]; label = l)
end
axislegend(ax)
fig

# ## Visualize agent distribution
const colormap = Dict(:Rock => "black", :Scissors => "gray", :Paper => "orange")
agent_color(agent) = colormap[kindof(agent)]
plotkw = (agent_color, agent_marker = :rect, agent_size = 5)
fig, ax, abmobs = abmplot(model; plotkw...)

fig

# ## Animation
model = initialize_rps()
abmvideo("docs/_static/rps_eventqueue.mp4", model;
    dt = 0.5, frames = 300,
    title = "Rock Paper Scissors (event based)",
    plotkw...,
)

#===
<video autoplay controls src="../_static/rps_eventqueue.mp4"></video>
===#
