This is an introduction into Julia for datascience.

It assumes you are somewhat familiar with Python, and it compares typical Python approaches with Julia.

To setup the notebook:

- Download and install [Julia](https://julialang.org)
- Open your [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/), which can be done in VScode after you installed the Julia plugin, or in a bash terminal.
- In the REPL, go to package mode by typing `]`. You will see the mode change color and it says `pkg>` instead of `julia>`.
- instantiate the environments by typing `instantiate` , which will install all dependencies and precompile everything.
- In packagemode, install Pluto by typing `add Pluto`. This will add Pluto to your environment.
- After Pluto is installed, leave pkg mode by typing backspace.
- Back in julia mode, type:

```bash
julia> using Pluto
julia> Pluto.run()
```

- This will start the Pluto interface in your browser. There are some sample notebooks included. If you like to have an introduction in Pluto notebooks, [watch this](https://youtu.be/IAF8DjrQSSk)
- In Pluto, open the first notebook 01_dataframes.jl .

There are four notebooks. They cover: data loading, preprocessing, visualisation and creating a very basic model. It's not complete, but it is a nice intro-by-doing, and it should give you a starting point to dive deeper.
