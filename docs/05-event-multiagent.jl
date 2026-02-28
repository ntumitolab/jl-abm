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
using Base64
using Agents.DataFrames
using CairoMakie
CairoMakie.activate!(px_per_unit = 1.0)

# The helper function displays video files in Jupyter notebooks
function display_mp4(filename)
    display("text/html", string("""<video autoplay controls><source src="data:video/x-m4v;base64,""",
        base64encode(open(read, filename)),
        """" type="video/mp4"></video>"""))
end

# Define rock, paper, and scissors (RPS) agents. One can use `variant(agent)` to see the agent type.
@agent struct Rock(GridAgent{2}) end
@agent struct Paper(GridAgent{2}) end
@agent struct Scissors(GridAgent{2}) end
@multiagent RPS(Rock, Paper, Scissors)

# Agent actions
function attack!(agent, model)
    ## Randomly pick a nearby agent
    contender = random_nearby_agent(agent, model)
    isnothing(contender) && return # do nothing if there isn't anyone nearby
    ## The attacking action will be dispatched to the following methods.
    attack!(variant(agent), variant(contender), contender, model)
    return nothing
end

attack!(::AbstractAgent, ::AbstractAgent, contender, model) = nothing
attack!(::Rock, ::Scissors, contender, model) = remove_agent!(contender, model)
attack!(::Scissors, ::Paper, contender, model) = remove_agent!(contender, model)
attack!(::Paper, ::Rock, contender, model) = remove_agent!(contender, model)

# Move actions use `move_agent!` and `swap_agents!` functions
function move!(agent, model)
    rand_pos = random_nearby_position(agent.pos, model)
    if isempty(rand_pos, model)
        move_agent!(agent, rand_pos, model)
    else
        occupant_id = id_in_position(rand_pos, model)
        occupant = model[occupant_id]
        swap_agents!(agent, occupant, model)
    end
    return nothing
end

# Reproduce actions use `replicate!` function
function reproduce!(agent, model)
    pos = random_nearby_position(agent, model, 1, pos -> isempty(pos, model))
    isnothing(pos) && return
    ## pass target position as a keyword argument
    replicate!(agent, model; pos)
    return nothing
end

# Defining the propensity ("rate" in Gillespie stochastic simulations) and timing of the events
attack_propensity = 1.0
movement_propensity = 0.5
reproduction_propensity(agent, model) = cos(abmtime(model))^2

# Register events with `AgentEvent`
attack_event = AgentEvent(action! = attack!, propensity = attack_propensity)
reproduction_event = AgentEvent(action! = reproduce!, propensity = reproduction_propensity)

# We want a different distribution other than exponential distribution for movement time
function movement_time(agent, model, propensity)
    ## Make time around 1
    t = 0.1 * randn(abmrng(model)) + 1
    return clamp(t, 0, Inf)
end

# Also rocks do not move
movement_event = AgentEvent(
    action! = move!, propensity = movement_propensity,
    types = Union{Scissors, Paper}, timing = movement_time
)

# Collect all events
events = (attack_event, reproduction_event, movement_event)

# Model function `EventQueueABM` for an event-driven ABM
const alltypes = (Rock, Paper, Scissors)

function initialize_rps(; n = 100, nx = n, ny = n, seed = 42)
    space = GridSpaceSingle((nx, ny))
    rng = Xoshiro(seed)
    model = EventQueueABM(RPS, events, space; rng, warn = false)
    for p in positions(model)
        ## Randomly assign one of the agent
        type = rand(abmrng(model), alltypes)
        add_agent!(p, constructor(RPS, type), model)
    end
    return model
end

# Have a look at the event queue
model = initialize_rps()
abmqueue(model)

# The time in `EventQueueABM` is continuous, so we can pass real-valued time.
step!(model, 123.456)
nagents(model)

# The `step!` function also accepts a terminating condition.
function terminate(model, t)
    willterm = length(allagents(model)) < 5000
    return willterm || (t > 1000.0)
end

model = initialize_rps()
step!(model, terminate)
abmtime(model)

# ## Data collection
# - adata: aggregated data to extract information from the execution stats
# - adf: agent data frame
model = initialize_rps()
adata = [(a -> variantof(a) === X, count) for X in alltypes]

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
const colormap = Dict(Rock => "black", Scissors => "gray", Paper => "orange")
agent_color(agent) = colormap[variantof(agent)]
plotkw = (agent_color, agent_marker = :rect, agent_size = 5)
fig, ax, abmobs = abmplot(model; plotkw...)

fig

# # Animation
model = initialize_rps()
abmvideo("rps.mp4", model;
    dt = 0.5, frames = 300,
    title = "Rock Paper Scissors (event based)",
    plotkw...,
)

#nb display_mp4("rps.mp4")
