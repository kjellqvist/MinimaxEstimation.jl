using MinimaxEstimation
using Documenter

makedocs(;
    modules=[MinimaxEstimation],
    authors="Olle Kjellqvist",
    repo="https://github.com/kjellqvist/MinimaxEstimation.jl/blob/{commit}{path}#L{line}",
    sitename="MinimaxEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kjellqvist.github.io/MinimaxEstimation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => Any[
            "Library" => "lib.md",
            "Example" => "example.md"
        ]
    ],
)

deploydocs(;
    devbranch = "main",
    repo="github.com/kjellqvist/MinimaxEstimation.jl",
)
