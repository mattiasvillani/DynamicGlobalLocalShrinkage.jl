using DynamicGlobalLocalShrinkage
using Documenter

DocMeta.setdocmeta!(DynamicGlobalLocalShrinkage, :DocTestSetup, :(using DynamicGlobalLocalShrinkage); recursive=true)

makedocs(;
    modules=[DynamicGlobalLocalShrinkage],
    authors="Mattias Villani",
    sitename="DynamicGlobalLocalShrinkage.jl",
    format=Documenter.HTML(;
        canonical="https://mattiasvillani.github.io/DynamicGlobalLocalShrinkage.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattiasvillani/DynamicGlobalLocalShrinkage.jl",
    devbranch="main",
)
