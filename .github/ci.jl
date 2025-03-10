using Distributed
using Tables
using MarkdownTables
using SHA

@everywhere begin
    ENV["GKSwstype"] = "100"
    using Literate, Pkg, JSON
end

# Strip SVG output from a Jupyter notebook
@everywhere function strip_svg(ipynb)
    oldfilesize = filesize(ipynb)
    nb = open(JSON.parse, ipynb, "r")
    for cell in nb["cells"]
        !haskey(cell, "outputs") && continue
        for output in cell["outputs"]
            !haskey(output, "data") && continue
            datadict = output["data"]
            if haskey(datadict, "image/png") || haskey(datadict, "image/jpeg")
                delete!(datadict, "text/html")
                delete!(datadict, "image/svg+xml")
            end
        end
    end
    write(ipynb, JSON.json(nb, 1))
    @info "Stripped SVG in $(ipynb). The original size is $(oldfilesize). The new size is $(filesize(ipynb))."
    return ipynb
end

# Remove cached notebook and sha files if there is no corresponding notebook
function clean_cache(cachedir)
    for (root, dirs, files) in walkdir(cachedir)
        for file in files
            if endswith(file, ".ipynb") || endswith(file, ".sha")
                fn = joinpath(joinpath(splitpath(root)[2:end]), splitext(file)[1])
                nb = fn * ".ipynb"
                lit = fn * ".jl"
                if !isfile(nb) && !isfile(lit)
                    fullfn = joinpath(root, file)
                    @info "Notebook $(nb) or $(lit) not found. Removing $(fullfn)."
                    rm(fullfn)
                end
            end
        end
    end
end

"Recursively list Literate notebooks. Also process caching."
function list_notebooks(basedir, cachedir)
    litnbs = String[]
    for (root, dirs, files) in walkdir(basedir)
        for file in files
            name, ext = splitext(file)
            if ext == ".jl"
                nb = joinpath(root, file)
                shaval = read(nb, String) |> sha1 |> bytes2hex
                @info "$(nb) SHA1 = $(shaval)"
                shafilename = joinpath(cachedir, root, name * ".sha")
                if isfile(shafilename) && read(shafilename, String) == shaval
                    @info "$(nb) cache hits and will not be executed."
                else
                    @info "$(nb) cache misses. Writing hash to $(shafilename)."
                    mkpath(dirname(shafilename))
                    write(shafilename, shaval)
                    push!(litnbs, nb)
                end
            end
        end
    end
    return litnbs
end

# Run a Literate.jl notebook
@everywhere function run_literate(file, cachedir; rmsvg=true)
    outpath = joinpath(abspath(pwd()), cachedir, dirname(file))
    mkpath(outpath)
    ipynb = Literate.notebook(file, outpath; mdstrings=true, execute=true)
    rmsvg && strip_svg(ipynb)
    return ipynb
end

function main(;
    basedir=get(ENV, "DOCDIR", "docs"),
    cachedir=get(ENV, "NBCACHE", ".cache"),
    rmsvg=true
)

    mkpath(cachedir)
    clean_cache(cachedir)
    litnbs = list_notebooks(basedir, cachedir)
    # Execute literate notebooks in worker process(es)
    ts_lit = pmap(litnbs; on_error=ex -> NaN) do nb
        @elapsed run_literate(nb, cachedir; rmsvg)
    end
    rmprocs(workers()) # Remove worker processes to release some memory
    # Debug notebooks one by one if there are errors
    for (nb, t) in zip(litnbs, ts_lit)
        if isnan(t)
            println("Debugging notebook: ", nb)
            try
                withenv("JULIA_DEBUG" => "Literate") do
                    run_literate(nb, cachedir; rmsvg)
                end
            catch e
                println(e)
            end
        end
    end
    any(isnan, ts_lit) && error("Please check literate notebook error(s).")

    # Print execution result
    Tables.table([litnbs ts_lit]; header=["Notebook", "Elapsed (s)"]) |> markdown_table(String) |> print
end

# Run code
main()
