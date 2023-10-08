@testset "mean_functions" begin
    @testset "ZeroMean" begin
        P = 3
        Q = 2
        D = 4
        # X = ColVecs(randn(rng, D, P))
        x = randn(P)
        x̄ = randn(P)
        f = ZeroMean{Float64}()

        for x in [x]
            @test AbstractGPs._map(f, x) == zeros(size(x))
            y = similar(x)
            AbstractGPs._map!(f, y, x)
            @test y == zeros(size(y))
            # differentiable_mean_function_tests(f, randn(rng, P), x)
        end

        # Manually verify the ChainRule. Really, this should employ FiniteDifferences, but
        # currently ChainRulesTestUtils isn't up to handling this, so this will have to do
        # for now.
        y, pb = rrule(AbstractGPs._map, f, x)
        @test y == AbstractGPs._map(f, x)
        Δmap, Δf, Δx = pb(randn(P))
        @test iszero(Δmap)
        @test iszero(Δf)
        @test iszero(Δx)
    end
    @testset "ConstMean" begin
        rng, D, N = MersenneTwister(123456), 5, 3
        # X = ColVecs(randn(rng, D, N))
        x = randn(rng, N)
        c = randn(rng)
        m = ConstMean(c)

        for x in [x]
            @test AbstractGPs._map(m, x) == fill(c, N)            
            y = similar(x)
            AbstractGPs._map!(m, y, x)
            @test y == fill(c, N)
 
            # differentiable_mean_function_tests(m, randn(rng, N), x)
        end
    end
    @testset "CustomMean" begin
        rng, N, D = MersenneTwister(123456), 11, 2
        x = randn(rng, N)
        foo_mean = x -> sum(abs2, x)
        f = CustomMean(foo_mean)

        @test AbstractGPs._map(f, x) == map(foo_mean, x)
        y = similar(x)
        AbstractGPs._map!(f, y, x)
        @test y == map(foo_mean, x)
        # differentiable_mean_function_tests(f, randn(rng, N), x)
    end
end
