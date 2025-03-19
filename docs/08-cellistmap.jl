#===
# ABM and CellListMap

https://github.com/m3g/CellListMap.jl can quickly calculate partical pairs interactions within a cutoff.

Modeling the particle repulsion:

$$
\begin{align}
U(r) &= k_i k_j (r^2 - (r_i - r_j)^2)^2, \quad \text{for} r ≤ (r_i + r_j) \\
U(r) &= 0, \quad \text{otherwise}
\end{align}
$$
===#
using Agents
using CellListMap
using CairoMakie
CairoMakie.activate!(px_per_unit = 1.0)

# Define particle agents
@agent struct Particle(ContinuousAgent{2,Float64})
    r::Float64 ## radius
    k::Float64 ## repulsion force constant
    mass::Float64
end

Particle(; vel, r, k, mass) = (vel, r, k, mass)

# Building the model
function initialize_bouncing(;
    number_of_particles=10_000,
    sides=SVector(500.0, 500.0),
    dt=0.001,
    max_radius=10.0,
    parallel=true
)
    ## initial random positions
    positions = [sides .* rand(SVector{2,Float64}) for _ in 1:number_of_particles]

    ## We will use CellListMap to compute forces, with similar structure as the positions
    forces = similar(positions)

    ## Space for the agents
    space2d = ContinuousSpace(sides; periodic=true)

    ## Initialize CellListMap particle system
    system = ParticleSystem(
        positions=positions,
        unitcell=sides,
        cutoff=2 * max_radius,
        output=forces,
        output_name=:forces, ## allows the system.forces alias for clarity
        parallel=parallel,
    )

    ## define the ABModel properties
    ## The system field contains the data required for CellListMap.jl
    properties = (;dt, number_of_particles, system)

    model = StandardABM(Particle,
        space2d;
        agent_step!,
        model_step!,
        agents_first = false,
        properties=properties
    )

    ## Create active agents
    for id in 1:number_of_particles
        pos = positions[id]
        prop_particle = Particle(
            r = (0.5 + 0.9 * rand()) * max_radius,
            k = 10 + 20 * rand(), ## random force constants
            mass = 10.0 + 100 * rand(), ## random masses
            vel = 100 * randn(SVector{2}) ## initial velocities)
            )
        add_agent!(pos, Particle, model, prop_particle...)
    end

    return model
end

# Computing the repulsion force
# It must follow CellListMap API and return a array for forces acting upon the particles
function calc_forces!(x, y, i, j, d2, forces, model)
    p_i = model[i]
    p_j = model[j]
    d = sqrt(d2)
    if d ≤ (p_i.r + p_j.r)
        dr = y - x ## x and y are minimum-image relative coordinates
        fij = 2 * (p_i.k * p_j.k) * (d2 - (p_i.r + p_j.r)^2) * (dr / d)
        forces[i] += fij
        forces[j] -= fij
    end
    return forces
end

# Update the pairwise forces using CellListMap API
function model_step!(model::ABM)
    map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, model),
        model.system,
    )
    return nothing
end

# Update agent positions and velocity
function agent_step!(agent, model::ABM)
    id = agent.id
    dt = abmproperties(model).dt
    force = model.system.forces[id]
    acc = force / agent.mass
    ## Update positions and velocities
    vel = agent.vel + acc * dt
    x = agent.pos + vel * dt + (acc / 2) * dt^2
    x = normalize_position(x, model)  ## Wraps agent position
    agent.vel = vel
    move_agent!(agent, x, model)
    ## !!! Remember to update positions in the ParticleSystem
    model.system.positions[id] = agent.pos
    return nothing
end

# Run simulation
function simulate(model=nothing; nsteps=1_000, number_of_particles=10_000)
    if isnothing(model)
        model = initialize_bouncing(number_of_particles=number_of_particles)
    end
    Agents.step!(model, nsteps)
end

# Test the performance
model = initialize_bouncing(number_of_particles=10_000)
@time simulate(model)

# The helper function below is adapted from `Agents.abmvideo` and correctly displays animations in Jupyter notebooks
function abmvio(model;
    dt = 1, framerate = 30, frames = 300, title = "", showstep = true,
    figure = (size = (600, 600),), axis = NamedTuple(),
    recordkwargs = (compression = 23, format ="mp4"), kwargs...
)
    ## title and steps
    abmtime_obs = Observable(abmtime(model))
    if title ≠ "" && showstep
        t = lift(x -> title*", time = "*string(x), abmtime_obs)
    elseif showstep
        t = lift(x -> "time = "*string(x), abmtime_obs)
    else
        t = title
    end

    axis = (title = t, titlealign = :left, axis...)
    ## First frame
    fig, ax, abmobs = abmplot(model; add_controls = false, warn_deprecation = false, figure, axis, kwargs...)
    resize_to_layout!(fig)
    ## Animation
    Makie.Record(fig; framerate, recordkwargs...) do io
        for j in 1:frames-1
            recordframe!(io)
            Agents.step!(abmobs, dt)
            abmtime_obs[] = abmtime(model)
        end
        recordframe!(io)
    end
end

# Visualize
model = initialize_bouncing(number_of_particles=1000)
vio = abmvio(
    model;
    framerate=20, frames=200, dt=5,
    title="Softly bouncing particles",
    agent_size=p -> p.r,
    agent_color=p -> p.k
)

vio |> display
