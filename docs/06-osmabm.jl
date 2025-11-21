# # Open street map example
# The zombie outbreak model showcases an ABM running on a map, using `OpenStreetMapSpace`.
using Agents
using Random
using CairoMakie
using OSMMakie
CairoMakie.activate!(px_per_unit = 1.0)

# The helper function is adapted from `Agents.abmvideo` and correctly displays animations in Jupyter notebooks
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

# Agents for zombies and healthy humans
@agent struct Zombie(OSMAgent)
    infected::Bool
    speed::Float64
end

# initialise model
function initialise_zombies(; seed = 1234)
    map_path = OSM.test_map()
    properties = Dict(:dt => 1 / 60)

    model = StandardABM(
        Zombie,
        OpenStreetMapSpace(map_path);
        agent_step! = zombie_step!,
        properties = properties,
        rng = Random.MersenneTwister(seed)
    )

    for id in 1:100
        start = random_position(model) ## At an intersection
        speed = rand(abmrng(model)) * 5.0 + 2.0 ## Random speed from 2-7kmph
        human = add_agent!(start, Zombie, model, false, speed)
        OSM.plan_random_route!(human, model; limit = 50) ## try 50 times to find a random route
    end

    ## We'll add patient zero at a specific (longitude, latitude)
    start = OSM.nearest_road((9.9351811, 51.5328328), model)
    finish = OSM.nearest_node((9.945125635913511, 51.530876112711745), model)

    speed = rand(abmrng(model)) * 5.0 + 2.0 ## Random speed from 2-7kmph
    zombie = add_agent!(start, model, true, speed)
    plan_route!(zombie, finish, model)
    return model
end

# Stepping function
function zombie_step!(agent, model)
    ## Each agent will progress along their route
    ## Keep track of distance left to move this step, in case the agent reaches its
    ## destination early
    distance_left = move_along_route!(agent, model, agent.speed * model.dt)

    if is_stationary(agent, model) && rand(abmrng(model)) < 0.1
        ## When stationary, give the agent a 10% chance of going somewhere else
        OSM.plan_random_route!(agent, model; limit = 50)
        ## Start on new route, moving the remaining distance
        move_along_route!(agent, model, distance_left)
    end

    ## Agents will be infected if they get too close (within 10m) to a zombie.
    if agent.infected
        map(i -> model[i].infected = true, nearby_ids(agent, model, 0.01))
    end
    return
end

# ## Animation
zombie_color(agent) = agent.infected ? :green : :black
zombie_size(agent) = agent.infected ? 10 : 8
zombies = initialise_zombies()

vio = abmvio(zombies;
    title = "Zombie outbreak", framerate = 15, frames = 200,
    agent_color = zombie_color, agent_size = zombie_size
)

save("zombie.mp4", vio)
vio |> display
