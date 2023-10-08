# Define the AbstractGP type and its API.

"""
    abstract type AbstractGP end

Supertype for various Gaussian process (GP) types. A common interface is provided for
interacting with each of these objects. See [1] for an overview of GPs.

[1] - C. E. Rasmussen and C. Williams. "Gaussian processes for machine learning". 
MIT Press. 2006.
"""
abstract type AbstractGP end

"""
    mean(f::AbstractGP, x::AbstractVector)

Computes the mean vector of the multivariate Normal `f(x)`.
"""
Statistics.mean(::AbstractGP, ::AbstractVector)

"""
    mean!(out::AbstractVector, f::AbstractGP, x::AbstractVector)

Computes the mean vector of the multivariate Normal `f(x)` and 
store it in `out`.
"""
Statistics.mean!(::AbstractVector, ::AbstractGP, ::AbstractVector)

"""
    cov(f::AbstractGP, x::AbstractVector)

Compute the `length(x)` by `length(x)` covariance matrix of the multivariate Normal `f(x)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector)

"""
    cov!(out::AbstractMatrix, f::AbstractGP, x::AbstractVector)

Compute the covariance matrix of the multivariate Normal `f(x)`
and store it in `out`.
"""
Statistics.cov!(::AbstractMatrix, ::AbstractGP, x::AbstractVector)

"""
    var(f::AbstractGP, x::AbstractVector)

Compute only the diagonal elements of `cov(f(x))`.
"""
Statistics.var(::AbstractGP, ::AbstractVector)

"""
    var!(out::AbstractVectorm f::AbstractGP, x::AbstractVector)

Compute only the diagonal elements of `cov(f(x))` and store them in `out`.
"""
var!(::AbstractVector, ::AbstractGP, ::AbstractVector)

"""
    cov(f::AbstractGP, x::AbstractVector, y::AbstractVector)

Compute the `length(x)` by `length(y)` cross-covariance matrix between `f(x)` and `f(y)`.
"""
Statistics.cov(::AbstractGP, x::AbstractVector, y::AbstractVector)

"""
    cov!(out::AbstractMatrix, f::AbstractGP, x::AbstractVector, y::AbstractVector)

Compute the cross-covariance matrix between `f(x)` and `f(y)` and store them in `out`.
"""
cov!(::AbstractMatrix, ::AbstractGP, x::AbstractVector, y::AbstractVector)

"""
    mean_and_cov(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and `cov(f(x))`. Sometimes more efficient than separately
computation, particularly for posteriors.
"""
StatsBase.mean_and_cov(f::AbstractGP, x::AbstractVector) = (mean(f, x), cov(f, x))

"""
    mean_and_cov(mout::AbstractVector, cout::AbstractMatrix, f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and `cov(f(x))` and store them in `mout` resp. `cout`.
"""
function mean_and_cov!(
        mout::AbstractVector, cout::AbstractMatrix, f::AbstractGP, x::AbstractVector) 
    (mean!(mout, f, x), cov!(cout, f, x))
end

"""
    mean_and_var(f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and the diagonal elements of `cov(f(x))`. Sometimes more efficient
than separately computation, particularly for posteriors.
"""
StatsBase.mean_and_var(f::AbstractGP, x::AbstractVector) = (mean(f, x), var(f, x))

"""
    mean_and_var!(mout::AbstractVector, vout::AbstractVector, f::AbstractGP, x::AbstractVector)

Compute both `mean(f(x))` and the diagonal elements of `cov(f(x))` and store them in 
`mout` resp. `vout`.
"""
function mean_and_var!(
        mout::AbstractVector, vout::AbstractVector, f::AbstractGP, x::AbstractVector) 
    (mean!(mout, f, x), var!(vout, f, x))
end


for (m, f) in [
    (:Statistics, :mean),
    (:Statistics, :var),
    (:Statistics, :cov),
    (:StatsBase, :mean_and_cov),
    (:StatsBase, :mean_and_var),

    (:Statistics, :mean!),
]
    @eval function $m.$f(::AbstractGP)
        return error(
            "`",
            $f,
            "(f::AbstractGP)` is not defined (on purpose!).\n",
            "Please provide an `AbstractVector` of locations `x` at which you wish to compute your ",
            $f,
            $((f === :mean_and_cov || f === :mean_and_var) ? " vectors" : " vector"),
            ", and call `",
            $f,
            "(f(x))`\n",
            "For more details please have a look at the AbstractGPs docs.",
        )
    end
end
