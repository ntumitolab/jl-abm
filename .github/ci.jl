using JSON
using Literate

ENV["GKSwstype"] = "100"

function main(; dopostproc=true)
    file = get(ENV, "NB", "test.ipynb")
    cachedir = get(ENV, "NBCACHE", ".cache")
    if endswith(file, ".jl")
        run_literate(file; cachedir, dopostproc)
    elseif endswith(file, ".ipynb")
        lit = to_literate(file)
        run_literate(lit; cachedir, dopostproc)
    end
end

# Post-process Jupyter notebook
function postprocess(ipynb)
    oldfilesize = filesize(ipynb)
    nb = open(JSON.parse, ipynb, "r")
    for cell in nb["cells"]
        !haskey(cell, "outputs") && continue
        for output in cell["outputs"]
            !haskey(output, "data") && continue
            datadict = output["data"]
            ## Remove SVG to reduce file size
            if haskey(datadict, "image/png") || haskey(datadict, "image/jpeg")
                delete!(datadict, "text/html")
                delete!(datadict, "image/svg+xml")
            end
            ## Process LaTeX output, wrap in an array if needed
            if haskey(datadict, "text/latex")
                latexcode = datadict["text/latex"]
                if (! (latexcode isa Vector))
                    datadict["text/latex"] = [latexcode]
                end
            end
        end
    end
    rm(ipynb; force=true)
    write(ipynb, JSON.json(nb, 2))
    @info "The original size is $(Base.format_bytes(oldfilesize)). The new size is $(Base.format_bytes(filesize(ipynb)))."
    return ipynb
end

# Convert a Jupyter notebook into a Literate notebook. Adapted from https://github.com/JuliaInterop/NBInclude.jl.
function to_literate(nbpath; shell_or_help = r"^\s*[;?]")
    nb = open(JSON.parse, nbpath, "r")
    jlpath = splitext(nbpath)[1] * ".jl"
    open(jlpath, "w") do io
        separator = ""
        for cell in nb["cells"]
            if cell["cell_type"] == "code"
                s = join(cell["source"])
                isempty(strip(s)) && continue # Jupyter doesn't number empty cells
                occursin(shell_or_help, s) && continue  # Skip cells with shell and help commands
                print(io, separator, "#---\n", s)  # Literate code block mark
                separator = "\n\n"
            elseif cell["cell_type"] == "markdown"
                txt = join(cell["source"])
                print(io, separator, "#===\n", txt, "\n===#")
                separator = "\n\n"
            end
        end
    end
    return jlpath
end

function run_literate(file; cachedir = ".cache", dopostproc=true)
    outpath = joinpath(abspath(pwd()), cachedir, dirname(file))
    mkpath(outpath)
    ipynb = Literate.notebook(file, dirname(file); mdstrings=true, execute=true)
    dopostproc && postprocess(ipynb)
    cp(ipynb, joinpath(outpath, basename(ipynb)); force=true)
    return ipynb
end

main()
