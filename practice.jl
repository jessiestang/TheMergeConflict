# Julia script to solve the 1D shallow water equations as a DAE problem
using NonlinearSolve, LinearAlgebra, Parameters, Plots, Sundials, Statistics

# --- 1. Parameter setup ---
function make_parameters()
    D = 10.0
    nx = 200
    x_domain = (0.0, 5.0)
    x = range(x_domain[1], x_domain[2], length=nx)
    dx = step(x)
    z_b_2 = fill(-D, nx)  # flat bottom to help solver
    h_0 = 0.1 .* exp.(-100 .* ((x .- mean(x)) .^ 2)) .- z_b_2
    h_0 = max.(h_0, 1e-6)
    q_0 = zeros(nx)
    t_0 = 0.0
    t_stop = 1.0
    g = 9.81
    c_f = 0.0  # friction off for now

    return Dict(
        :nx => nx,
        :x => x,
        :dx => dx,
        :z_b_2 => z_b_2,
        :h_0 => h_0,
        :q_0 => q_0,
        :t_0 => t_0,
        :t_stop => t_stop,
        :g => g,
        :c_f => c_f,
    )
end

# --- 2. Initial condition ---
function initial_conditions(params)
    h = copy(params[:h_0])
    q = copy(params[:q_0])
    return h, q
end

# --- 3. DAE residual function ---
function swe_dae_residual!(residual, du, u, p, t)
    nx = p[:nx]
    dx = p[:dx]
    g = p[:g]
    z_b = p[:z_b_2]
    c_f = p[:c_f]

    h = u[1:nx]
    q = u[nx+1:2nx]
    dhdt = du[1:nx]
    dqdt = du[nx+1:2nx]

    h_safe = max.(h, 1e-6)

    f_h = q
    f_q = q .^ 2 ./ h_safe .+ 0.5 * g .* h_safe .^ 2

    ∂f_h_∂x = similar(f_h)
    ∂f_q_∂x = similar(f_q)

    for i in 2:nx-1
        ∂f_h_∂x[i] = (f_h[i+1] - f_h[i-1]) / (2dx)
        ∂f_q_∂x[i] = (f_q[i+1] - f_q[i-1]) / (2dx)
    end
    ∂f_h_∂x[1] = (f_h[2] - f_h[1]) / dx
    ∂f_h_∂x[end] = (f_h[end] - f_h[end-1]) / dx
    ∂f_q_∂x[1] = (f_q[2] - f_q[1]) / dx
    ∂f_q_∂x[end] = (f_q[end] - f_q[end-1]) / dx

    ζ = h .+ z_b
    ∂ζ_∂x = similar(ζ)
    for i in 2:nx-1
        ∂ζ_∂x[i] = (ζ[i+1] - ζ[i-1]) / (2dx)
    end
    ∂ζ_∂x[1] = (ζ[2] - ζ[1]) / dx
    ∂ζ_∂x[end] = (ζ[end] - ζ[end-1]) / dx

    friction = c_f .* q .* abs.(q) ./ h_safe .^ 2

    residual[1:nx] .= dhdt .+ ∂f_h_∂x
    residual[nx+1:2nx] .= dqdt .+ ∂f_q_∂x .+ g .* h_safe .* ∂ζ_∂x .+ friction

    # Dirichlet boundary conditions
    residual[1] = h[1] - 1.0
    residual[nx] = h[end] - 1.0
    residual[nx+1] = q[1] - 0.0
    residual[2nx] = q[end] - 0.0

    return nothing
end

# --- 4. Time integration ---
function timeloop(params)
    nx = params[:nx]
    x = params[:x]
    dx = params[:dx]
    tstart = params[:t_0]
    tstop = params[:t_stop]
    N = length(x)

    h0, q0 = initial_conditions(params)
    u0 = vcat(h0, q0)
    du0 = zeros(2N)
    tspan = (tstart, tstop)
    differential_vars = trues(2N)

    dae_prob = DAEProblem(swe_dae_residual!, du0, u0, tspan, params;
                          differential_vars=differential_vars)

    println("Solving DAE...")
    sol = solve(dae_prob, IDA(linear_solver=:Dense), reltol=1e-8, abstol=1e-8)

    return sol
end

# --- 5. Visualization ---
function animate_solution(solution, params)
    x = params[:x]
    z_b = params[:z_b_2]
    n = length(x)
    h_mat = hcat([u[1:n] for u in solution.u]...)
    t_vals = solution.t
    h_min = minimum(h_mat .+ z_b)
    h_max = maximum(h_mat .+ z_b)
    buffer = 0.1 * (h_max - h_min)
    ylims = (h_min - buffer, h_max + buffer)
    max_frames = min(100, size(h_mat, 2))
    indices = round.(Int, range(1, stop=size(h_mat, 2), length=max_frames))
    anim = @animate for (j, i) in enumerate(indices)
        println("Frame $j of $(length(indices)) - Bottom + Surface")
        surface = h_mat[:, i] .+ z_b
        plot(x, surface, label="Water surface", lw=2, ylim=ylims)
        plot!(x, z_b, label="Bottom", linestyle=:dash, color=:black)
        title!("t = $(round(t_vals[i], digits=2)) s")
    end
    gif(anim, "bottom_and_surface2.gif", fps=5)
end

function plot_mass_conservation_gif(solution, params)
    dx = params[:dx]
    n = params[:nx]
    t_vals = solution.t
    mass = [sum(u[1:n]) * dx for u in solution.u]
    m0 = mass[1]
    buffer = 0.005 * m0
    ylims = (m0 - buffer, m0 + buffer)
    max_frames = min(100, length(t_vals))
    indices = round.(Int, range(1, stop=length(t_vals), length=max_frames))
    anim = @animate for (j, i) in enumerate(indices)
        println("Frame $j of $(length(indices)) - Mass plot")
        plot(t_vals[1:i], mass[1:i],
            label="Total Mass", lw=2, c=:blue,
            xlabel="Time (s)", ylabel="Mass", legend=false, ylim=ylims)
        scatter!([t_vals[i]], [mass[i]], label="", c=:red)
        title!("t = $(round(t_vals[i], digits=2)) s")
    end
    gif(anim, "mass_conservation2.gif", fps=5)
end

function plot_velocity(solution, params)
    x = params[:x]
    n = length(x)
    t_vals = solution.t
    velocity_mat = hcat([
        solution.u[i][n+1:2n] ./ max.(solution.u[i][1:n], 1e-6)
        for i in 1:length(t_vals)
    ]...)
    vmin = minimum(velocity_mat)
    vmax = maximum(velocity_mat)
    buffer = 0.05 * (vmax - vmin)
    ylims = (vmin - buffer, vmax + buffer)
    indices = round.(Int, range(1, stop=length(t_vals), length=100))
    anim = @animate for i in indices
        plot(x, velocity_mat[:, i],
            label="Velocity", lw=2, ylim=ylims,
            xlabel="x", ylabel="Velocity",
            title="Velocity at t = $(round(t_vals[i], digits=2)) s")
    end
    gif(anim, "velocity_evolution2.gif", fps=3)
end

# --- 6. Main script ---
params = make_parameters()
solution = timeloop(params)
animate_solution(solution, params)
plot_mass_conservation_gif(solution, params)
plot_velocity(solution, params)
