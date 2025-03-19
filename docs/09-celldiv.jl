#===
# Cell division model

https://juliadynamics.github.io/Agents.jl/stable/examples/delaunay/

Using Agents.jl and https://github.com/JuliaGeometry/DelaunayTriangulation.jl
===#

# Define agents (3 colors of cells)
using Agents
using StaticArrays
import DelaunayTriangulation as DT
using DelaunayTriangulation
using LinearAlgebra
using StreamSampling
using Random
using StatsBase

@enum CellType begin
    Red
    Blue
    Orange
end

@agent struct Cell(ContinuousAgent{2,Float64})
    const color::CellType
    const birth::Float64
    death::Float64 = Inf
end

# Defining methods for the agents
DT.getx(cell::Cell) = cell.pos[1]
DT.gety(cell::Cell) = cell.pos[2]
DT.number_type(::Type{Cell}) = Float64
DT.number_type(::Type{Vector{Cell}}) = Float64
DT.is_point2(::Cell) = true

# Define model parameters as functions
spring_constant(p, q) = 20.0 # μ
heterotypic_spring_constant(p, q) = p.color == q.color ? 1.0 : 0.1 # μₕₑₜ
drag_coefficient(p) = 1 / 2 # η
mature_cell_spring_rest_length(p, q) = 1.0 # s
expansion_rate(p, q) = 0.05 * mature_cell_spring_rest_length(p, q) # ε
perturbation(p) = 0.01 # ξ
cutoff_distance(p, q) = 1.5 # ℓₘₐₓ
intrinsic_proliferation_rate(p) = p.color == Red ? 0.4 : p.color == Blue ? 0.5 : 0.8 # β
carrying_capacity_density(p) = 100.0^2 # K
min_division_age(p) = 1.0 # tₘᵢₙ
max_division_age(p) = p.color == Red ? 15.0 : p.color == Blue ? 20.0 : 3.0 # tₘₐₓ
max_age(p) = p.color == Red ? 10.0 : p.color == Blue ? 10.0 : 3.0 # dₘₐₓ
death_rate(p) = p.color == Red ? 0.001 : p.color == Blue ? 0.00005 : 0.0001 # psick
mutation_probability(p) = p.color == Red ? 0.3 : p.color == Blue ? 0.5 : 0.05 # pₘᵤₜ
min_area(p) = 1e-2 # Aₘᵢₙ

# Compute parameters for a pair of cells
spring_constant(model, i::Int, j::Int, t) = spring_constant(model, model[i], model[j], t)
function spring_constant(model, p, q, t)
    δ = norm(p.pos - q.pos)
    s = rest_length(model, p, q, t)
    μ = spring_constant(p, q)
    t < 1 && return μ # no adhesion for the initial population
    μₕₑₜ = heterotypic_spring_constant(p, q)
    if δ > s
        return μₕₑₜ * μ
    else
        return μ
    end
end

#---
rest_length(model, i::Int, j::Int, t) = rest_length(model, model[i], model[j]..., t)
function rest_length(model, p, q, t)
    s = mature_cell_spring_rest_length(p, q)
    ε = expansion_rate(p, q)
    return min(s, (s - ε) * t + ε)
end

#---
function proliferation_rate(model, i::Int, t)
    p = model[i]
    age = t - p.birth
    tₘᵢₙ = min_division_age(p)
    tₘₐₓ = max_division_age(p)
    A = get_area(model.tessellation, i)
    if age ≤ tₘᵢₙ || age ≥ tₘₐₓ || A < min_area(p)
        return 0.0
    end
    vorn = model.tessellation
    Aᵢ = get_area(vorn, i)
    β = intrinsic_proliferation_rate(p)
    K = carrying_capacity_density(p)
    return max(0.0, β * (1 - 1 / (K * Aᵢ)))
end

# Forces between two cells
force(model, i::Int, j::Int, t) = force(model, model[i], model[j], t)
function force(model, p, q, t)
    δ = norm(p.pos - q.pos)
    if δ > cutoff_distance(p, q)
        return SVector(0.0, 0.0)
    end
    μ = spring_constant(model, p, q, t)
    s = rest_length(model, p, q, t)
    rᵢⱼ = q.pos - p.pos
    return μ * (norm(rᵢⱼ) - s) * rᵢⱼ / norm(rᵢⱼ)
end

# Final random tug
function random_force(model, i)
    p = model[i]
    ξ = perturbation(p)
    η₁, η₂ = randn(), randn()
    Δt = model.dt
    return sqrt(2ξ / Δt) * SVector(η₁, η₂)
end

# Forces acting on one cell
function force(model, i::Int, t)
    F = SVector(0.0, 0.0)
    for j in get_neighbours(model.triangulation, i)
        DT.is_ghost_vertex(j) && continue
        F = F + force(model, i, j, t)
    end
    F = F + random_force(model, i)
    return F
end

# Cell movement velocity
velocity(model, i, t) = force(model, i, t) / drag_coefficient(model[i])

# First `update_velocities!()` and then `update_positions!()` for each cell
function update_velocities!(model, t)
    for i in each_solid_vertex(model.triangulation)
        model[i].vel = velocity(model, i, t)
    end
    return model
end

function new_position(model, i, t)
    xᵢ = model[i]
    vel = xᵢ.vel
    r = xᵢ.pos + model.dt * vel
    x, y = r
    xmax, ymax = spacesize(model)
    if x < 0 || x > xmax || y < 0 || y > ymax
        r = xᵢ.pos
    end
    return r
end

function update_positions!(model, t)
    update_velocities!(model, t)
    for i in each_solid_vertex(model.triangulation)
        model[i].pos = new_position(model, i, t)
    end
    return model
end

# Search and select which Voronoi cell to proliferate
function proliferation_probability(model, t)
    Δt = model.dt
    ## Technically nagents is not the number of alive agents, but with the way we are handling agents this is correct
    probs = zeros(nagents(model))
    for i in allids(model)
        if !DT.has_vertex(model.triangulation, i) || i in model.dead_cells
            i > 1 && (probs[i] = probs[i-1])
            continue
        end

        Gᵢ = proliferation_rate(model, i, t)

        ## Cumulative sum of the probabilities
        if i > 1
            probs[i] = probs[i-1] + Gᵢ * Δt
        else
            probs[i] = Gᵢ * Δt
        end
    end
    return probs
end

function select_proliferative_cell(model, probs)
    E = probs[end]
    u = rand() * E
    i = searchsortedlast(probs, u) + 1 ## searchsortedlast instead of searchsortedfirst since we skip over some agents in probs
    return i
end

# sampling from a Voronoi cell
function sample_triangle(tri::Triangulation, T)
    i, j, k = triangle_vertices(T)
    p, q, r = get_point(tri, i, j, k)
    px, py = getxy(p)
    qx, qy = getxy(q)
    rx, ry = getxy(r)
    a = (qx - px, qy - py)
    b = (rx - px, ry - py)
    u₁, u₂ = rand(), rand()
    if u₁ + u₂ > 1
        u₁, u₂ = 1 - u₁, 1 - u₂
    end
    ax, ay = getxy(a)
    bx, by = getxy(b)
    wx, wy = u₁ * ax + u₂ * bx, u₁ * ay + u₂ * by
    return SVector(px + wx, py + wy)
end

# select a random triangle from a triangulation
function random_triangle(tri::Triangulation)
    triangles = DT.each_solid_triangle(tri)
    area(T) = DT.triangle_area(get_point(tri, triangle_vertices(T)...)...)
    T = itsample(triangles, area)
    return T
end

# First, triangulate the Voronoi cell. Then, sample a triangle from the triangulation, and finally sample a point from the triangle
function triangulate_voronoi_cell(vorn::VoronoiTessellation, i)
    S = @view get_polygon(vorn, i)[1:end-1]
    points = DT.get_polygon_points(vorn)
    return triangulate_convex(points, S)
end
function sample_voronoi_cell(vorn::VoronoiTessellation, i)
    tri = triangulate_voronoi_cell(vorn, i)
    T = random_triangle(tri)
    return sample_triangle(tri, T)
end

# computing the daughter cell and performing the proliferation event.
function place_daughter_cell!(model, i, t)
    parent = model[i]
    daughter = sample_voronoi_cell(model.tessellation, i) # this is an SVector, not a Cell
    u = rand()
    clr = parent.color
    if u < mutation_probability(parent)
        newclr = clr == Red ? Blue : clr == Blue ? Orange : Red
    else
        newclr = clr
    end
    add_agent!(daughter, model; color=newclr, birth=t, vel=SVector(0.0, 0.0))
    return daughter
end

function proliferate_cells!(model, t)
    probs = proliferation_probability(model, t)
    u = rand()
    event = u < probs[end]
    !event && return false
    i = select_proliferative_cell(model, probs)
    daughter = place_daughter_cell!(model, i, t)
    return true
end

# Mark cells as dead
function cull_cell!(model, i, t)
    p = model[i]
    elder = t - p.birth > max_age(p)
    sick = rand() < model.dt * death_rate(p)
    xmax, ymax = spacesize(model)
    x, y = p.pos
    outside = x < 0 || x > xmax || y < 0 || y > ymax
    if elder || sick || outside
        push!(model.dead_cells, i)
        p.death = t
    end
    return model
end
function cull_cells!(model, t)
    for i in each_solid_vertex(model.triangulation)
        cull_cell!(model, i, t)
    end
    return model
end

# Define the stepping function of the ABModel
function model_step!(model)
    stepn = abmtime(model)
    t = stepn * model.dt
    cull_cells!(model, t)
    proliferate_cells!(model, t)
    update_positions!(model, t)
    model.triangulation = retriangulate(model.triangulation, allagents(model); skip_points=model.dead_cells)
    model.tessellation = voronoi(model.triangulation, clip=true)
    return model
end

# Initialize the model
function initialize_cell_model(;
    ninit=50,
    radius=2.0,
    dt=0.01,
    sides=SVector(20.0, 20.0),
    seed=0)
    Random.seed!(seed)
    ## Generate the initial random positions
    cent = SVector(sides[1] / 2, sides[2] / 2)
    cells = map(1:ninit) do i
        θ = 2π * rand()
        r = radius * sqrt(rand())
        pos = cent + SVector(r * cos(θ), r * sin(θ))
        cell = Cell(; id=i, pos=pos,
            color=Red, birth=0.0, vel=SVector(0.0, 0.0))
    end
    positions = [cell.pos for cell in cells]

    ## Compute the triangulation and the tessellation
    triangulation = triangulate(positions)
    tessellation = voronoi(triangulation, clip=true)

    ## Define the model parameters
    properties = Dict(
        :triangulation => triangulation,
        :tessellation => tessellation,
        :dt => dt,
        :dead_cells => Set{Int}()
    )

    ## Define the space
    space = ContinuousSpace(sides; periodic=false)

    ## Define the model
    model = StandardABM(Cell, space; model_step!, properties, container=Vector)

    ## Add the agents
    for (id, pos) in pairs(positions)
        add_agent!(pos, model; color=Red, birth=0.0, vel=SVector(0.0, 0.0))
    end

    return model
end

# Helper functions
function count_cell_type(model, type)
    stepn = abmtime(model)
    t = stepn * model.dt
    n = 0
    for i in each_solid_vertex(model.triangulation)
        n += model[i].color == type
    end
    return n
end
count_red(model) = count_cell_type(model, Red)
count_blue(model) = count_cell_type(model, Blue)
count_orange(model) = count_cell_type(model, Orange)
count_total(model) = num_solid_vertices(model.triangulation)
function average_cell_area(model)
    area_itr = (get_area(model.tessellation, i) for i in each_solid_vertex(model.triangulation))
    mean_area = mean(area_itr)
    return mean_area
end
function cell_diameter(vorn, i)
    S = get_polygon(vorn, i)
    ## This is an O(|S|^2) method, but |S| is small so it is fine
    max_d = 0.0
    for i in S
        p = get_polygon_point(vorn, i)
        for j in S
            i == j && continue
            q = get_polygon_point(vorn, j)
            d = norm(getxy(p) .- getxy(q))
            max_d = max(max_d, d)
        end
    end
    return max_d
end
function average_cell_diameter(model)
    diam_itr = (cell_diameter(model.tessellation, i) for i in each_solid_vertex(model.triangulation))
    mean_diam = mean(diam_itr)
    return mean_diam
end
function average_spring_length(model)
    spring_itr = (norm(model[i].pos - model[j].pos) for (i, j) in each_solid_edge(model.triangulation))
    mean_spring = mean(spring_itr)
    return mean_spring
end

# Run the simulations
finalT = 50.0
model = initialize_cell_model()
nsteps = Int(finalT / model.dt)
mdata = [count_red, count_blue, count_orange, count_total,
    average_cell_area, average_cell_diameter, average_spring_length]
agent_df, model_df = run!(model, nsteps; mdata);
