# # ABM + DiffEq
# https://juliadynamics.github.io/Agents.jl/stable/examples/diffeq/
using Agents
using Distributions
using CairoMakie
using OrdinaryDiffEq
using DiffEqCallbacks
CairoMakie.activate!(px_per_unit = 1.0)

# Fisher agents
@agent struct Fisher(NoSpaceAgent)
    competence::Int
    yearly_catch::Float64
end

# Set fishing quota
function fish!(integrator, model)
    integrator.p[2] = integrator.u[1] > model.min_threshold ?
        sum(a.yearly_catch for a in allagents(model)) : 0.0
    Agents.step!(model, 1)
end

# Fish population change
function fish_stock!(ds, s, p, t)
    max_population, h = p
    ds[1] = s[1] * (1 - (s[1] / max_population)) - h
end

# Agents catch the fish
function agent_cb_step!(agent, model)
    agent.yearly_catch = rand(abmrng(model), Poisson(agent.competence))
end

# ABM model embedded in the callback
function initialise_cb(; min_threshold = 60.0, nagents = 50)
    model = StandardABM(Fisher; agent_step! = agent_cb_step!,
                        properties = Dict(:min_threshold => min_threshold))
    for _ in 1:nagents
        competence = floor(rand(abmrng(model), truncated(LogNormal(), 1, 6)))
        add_agent!(model, competence, 0.0)
    end
    return model
end

# Setup the problem
modelcb = initialise_cb()
tspan = (0.0, 20.0 * 365.0)
initial_stock = 400.0
max_population = 500.0
prob = OrdinaryDiffEq.ODEProblem(fish_stock!, [initial_stock], tspan, [max_population, 0.0])

## Each Dec 31st, we call fish! that adds our catch modifier to the stock, and steps the ABM model
fish = DiffEqCallbacks.PeriodicCallback(i -> fish!(i, modelcb), 364)
## Stocks are replenished again
reset = DiffEqCallbacks.PeriodicCallback(i -> i.p[2] = 0.0, 365)

@time sol = solve(prob, Tsit5();  callback = CallbackSet(fish, reset))
#---
discrete = vcat(sol(0:365:(365 * 20))[:,:]...)
f = Figure(size = (600, 400))
ax = f[1, 1] = Axis(
        f,
        xlabel = "Year",
        ylabel = "Stock",
        title = "Fishery Inventory",
    )
lines!(ax, discrete, linewidth = 2, color = :blue)
f
