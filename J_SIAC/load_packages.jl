# TO RUN THIS FILE AND GET ALL PACKAGES INSTALLED GO TO TERMINAL AND TYPE julia getting_started.jl


using Pkg

# To add a new package that is part of the Julia release. NOTE: you only add packages ONCE. Then, as long as you don't delete your Julia folder, they will be there.

Pkg.add("Revise")
Pkg.add("GaussQuadrature")
Pkg.add("Polynomials")
Pkg.add("SpecialPolynomials")
Pkg.add("PyPlot")
Pkg.add("PyCall")
Pkg.add("Printf")
Pkg.add("Einsum")
Pkg.add("BSplines")
