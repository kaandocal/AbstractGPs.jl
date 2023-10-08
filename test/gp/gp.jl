@testset "gp" begin

    # Ensure that GP implements the AbstractGP API consistently.
    @testset "GP" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        m, k = CustomMean(sin), Matern32Kernel()
        f = GP(m, k)
        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean(f, x) == AbstractGPs._map(m, x)
        y = similar(x)
        mean!(y, f, x)
        @test y ≈ mean(f, x)

        @test cov(f, x) == kernelmatrix(k, x)
        z = similar(cov(f, x))
        cov!(z, f, x)
        @test z ≈ cov(f, x)

        abstractgp_interface_tests(f, x, x′)
    end

    # Check that mean-function specialisations work as expected.
    @testset "sugar" begin
        @test GP(5, Matern32Kernel()).mean isa ConstMean
        @test GP(Matern32Kernel()).mean isa ZeroMean
    end
end
