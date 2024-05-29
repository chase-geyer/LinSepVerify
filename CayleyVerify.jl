import MathOptInterface
using LinearAlgebra
using JuMP, Gurobi
const GT = MOI.GreaterThan{Float64}
const LT = MOI.LessThan{Float64}

abstract type UnivariateFunction end
struct UnivariateAffineFunction <: UnivariateFunction
    a::Float64
    b::Float64
end

struct StaircaseFunction <: UnivariateFunction
    # each piece is a linear function as a section of the staircase function
    breakpoints::Vector{Float64}
    slopes::Vector{Float64}
    constant_terms::Vector{Float64}
    s::Float64
    function StaircaseFunction(bp::Vector{Float64}, slps::Vector{Float64}, cons::Vector{Float64})
        s = findmax(slps)[1]
        @assert length(Set(slps)) <= 2
        return new(bp, slps, cons, s)
    end
end

function eval(f::UnivariateFunction, x::Float64) end
function eval(f::UnivariateAffineFunction, x::Float64)
    # calculating neuron with weight a and bias b
    return f.a*x + f.b
end

function eval(f::StaircaseFunction, x::Float64)
    bp = f.breakpoints
    n = length(bp)
    @assert x >= bp[1] && x <= bp[n]

    if x == bp[n]
        return f.slopes[n-1]*x + f.constant_terms[n-1]
    else
        i = searchsortedlast(bp, x)
        return f.slopes[i]*x + f.constant_terms[i]
    end
end

struct BoxDomain
    lowerbound::Vector{Float64}
    upperbound::Vector{Float64}
    function BoxDomain(L::Vector{Float64}, U::Vector{Float64})
        #for (l, u) in zip(L, U)
        #    if l > u
        #        error("srsly dude ?!?")
        #    end
        #end
        return new(L, U)
    end
end

struct Neuron{F<:Union{StaircaseFunction, UnivariateAffineFunction}}
    weight::Vector{Float64}
    bias::Float64
    activation::F
    input_domain::BoxDomain
    Δ::Vector{Float64}
    H₁::Vector{Vector{Float64}} # coefficients of ψ when θ² = 0 
    H₂::Vector{Vector{Float64}} # coefficients of ψ when θ¹ = 0
end

function Neuron(
    w::Vector{Float64},
    b::Float64,
    f::StaircaseFunction,
    D::BoxDomain,
    )
    # box domain vectors
    U = D.upperbound
    L = D.lowerbound
    # number of neurons (based on weights)
    n = length(w)
    # breakpoints of the activation function
    h = f.breakpoints
    # number of breakpoints
    k = length(h) - 1

    # incorporating neuron's specific weights for a given piece of the piecewise linear gunctions
    Δ = [(U[j] - L[j])*abs(w[j]) for j in 1:n]
    

    h₁ = h[2:k+1] .- b .- sum([w[i]*U[i] for i in 1:n if w[i] > 0]) .- 
                        sum([w[i]*L[i] for i in 1:n if w[i] < 0])
    
    H₁ = [h₁ .+ sum(Δ[i:n]) for i in 1:n+1]
    
    h₂ = -h[k:-1:1] .+ b .+ sum([w[i]*U[i] for i in 1:n if w[i] < 0]) .+ 
                        sum([w[i]*L[i] for i in 1:n if w[i] > 0])
    H₂ = [h₂ .+ sum(Δ[i:n]) for i in 1:n+1]
    return Neuron(w, b, f, D, Δ, H₁, H₂)
end

struct NeuralNetwork
    weights::Vector{Matrix{Float64}}
    biases::Vector{Vector{Float64}}
    activation::Vector{StaircaseFunction}
end

function NeuralNetwork(
        net_from_pickle::Vector{Any},
        activation::Vector{StaircaseFunction},
    )
    n = length(net_from_pickle[1]) # since it's structured [layer, bias], can pull from either
    weights = []
    biases = []
    for i in 1:n 
        weight = vcat([w' for w in net_from_pickle[1][i]]...) # calculate weight matrix for each layer
        bias = net_from_pickle[2][i]
        push!(weights, weight)
        push!(biases, bias)
    end
    return NeuralNetwork(weights, biases, activation)
end

function generate_zcoef_from_alpha(
        neuron::Neuron{StaircaseFunction},
        alpha::Vector{Float64},
        sign::Union{LT, GT},
    )
    w = neuron.weight
    #@assert all(abs.(alpha) .<= abs.(w))
    bias = neuron.bias
    b = neuron.activation.constant_terms
    
    h = neuron.activation.breakpoints .- bias
    c₀ = -alpha
    c₁ = neuron.activation.s*w - alpha
    
    if sign isa GT
        z₀ = solve_knapsackseries(c₀, neuron.input_domain, w, h)
        z₁ = solve_knapsackseries(c₁, neuron.input_domain, w, h)
    elseif sign isa LT
        z₀ = -solve_knapsackseries(-c₀, neuron.input_domain, w, h)
        z₁ = -solve_knapsackseries(-c₁, neuron.input_domain, w, h)
    end
    a = neuron.activation.slopes
    return [a[i] > 0.0 ? z₁[i] + a[i]*bias + b[i] : z₀[i] + b[i] for i in 1:length(a)]
end

function solve_knapsackseries(
        c::Vector{Float64}, #what does c represent in this case?
        box::BoxDomain,
        w::Vector{Float64},
        h::Vector{Float64},
    )
    U = box.upperbound
    L = box.lowerbound
    n = length(w)
    k = length(h) - 1
    z_coef = zeros(k)
    
    x₀ = [c[i] > 0 ? U[i] : L[i] for i in 1:n]
    i₀ = max(min(searchsortedlast(h, w ⋅ x₀), k),1)

    var_score = [(i, w[i] != 0 ? c[i]/w[i] : Inf) for i in 1:n]
    upperhalf_var_order = sort(filter(x -> x[2] <= 0, var_score), by = x -> abs(x[2]))
    lowerhalf_var_order = sort(filter(x -> x[2] >= 0, var_score), by = x -> abs(x[2]))
    # Why do we knapsack twice? -- since we have to consider the upper and lower half of the activation function
    z_coef[i₀] = c ⋅ x₀
    j = 1
    optsol = copy(x₀)
    # upper half operation (from i₀ to k)
    for i in i₀+1:1:k
        while h[i] > w ⋅ optsol + 1e-12 && j <= n
            gap = h[i] - w ⋅ optsol
            var, value = upperhalf_var_order[j]
            update = optsol[var] + gap / w[var]
            if w[var] < 0
                if L[var] >= update
                    optsol[var] = L[var]
                    j += 1
                else
                    optsol[var] = update
                end
            else
                if U[var] <= update
                    optsol[var] = U[var]
                    j += 1
                else
                    optsol[var] = update
                end
            end
        end
        z_coef[i] = c ⋅ optsol
    end
    j = 1
    optsol = copy(x₀)
    # lower half operation (from i₀ to 1)
    for i in i₀-1:-1:1
        while h[i+1] < w ⋅ optsol - 1e-12 && j <= n
            gap = h[i+1] - w ⋅ optsol
            var, value = lowerhalf_var_order[j]
            update = optsol[var] + gap / w[var]
            if w[var] > 0
                if L[var] >= update
                    optsol[var] = L[var]
                    j += 1
                else
                    optsol[var] = update
                end
            else
                if U[var] <= update
                    optsol[var] = U[var]
                    j += 1
                else
                    optsol[var] = update
                end
            end
        end
        z_coef[i] = c ⋅ optsol
    end
    return z_coef
end

function optimal_ψ(
        x::Vector{Float64},
        z::Vector{Float64},
        Δ::Vector{Float64},
        H::Vector{Vector{Float64}},
    )
    n = length(x)
    k = length(z)
    breakpoints = [x[i]/Δ[i]  for i in 1:n]
    p = sortperm(breakpoints)
    sort!(breakpoints)
    pushfirst!(breakpoints, 0.0)
    push!(breakpoints, 1.0)

    optimal_value = 0.0
    optimal_solution = zeros(k)
    weighted_sum = 0.0 # sum of z_iq_i
    i = 2
    j = 1
    while optimal_value > -1e-6 && i <= n+2
        if i >= 3
            optimal_value = optimal_value - weighted_sum*Δ[i-2] + x[i-2]
        end
        for var in j:k
            if z[var] == 0.0
                j += 1
                continue
            end
            if H[i-1][var] <= 0
                if z[var]*(1 - optimal_solution[var]) <= breakpoints[i] - weighted_sum
                    optimal_value += H[i-1][var]*(1 - optimal_solution[var])*z[var]
                    weighted_sum += z[var]*(1 - optimal_solution[var])
                    optimal_solution[var] = 1
                    j += 1
                else
                    optimal_solution[var] += (breakpoints[i] - weighted_sum) / z[var]
                    optimal_value += H[i-1][var]*(breakpoints[i] - weighted_sum)
                    weighted_sum = breakpoints[i]
                    break
                end
            else
                if weighted_sum >= breakpoints[i-1]
                    break
                end
                if z[var]*(1 - optimal_solution[var]) <= breakpoints[i-1] - weighted_sum
                    optimal_value += H[i-1][var]*(1 - optimal_solution[var])*z[var]
                    weighted_sum += z[var]*(1 - optimal_solution[var])
                    optimal_solution[var] = 1
                    j += 1
                else
                    optimal_solution[var] += (breakpoints[i-1] - weighted_sum) / z[var]
                    optimal_value += H[i-1][var]*(breakpoints[i-1] - weighted_sum)
                    weighted_sum = breakpoints[i-1]
                    break
                end
            end
        end
        i = i + 1
    end
    return optimal_value, i-3, p
end

function generate_alpha(
        neuron::Neuron{StaircaseFunction},
        x::Vector{Float64},
        y::Float64,
        z::Vector{Float64},
    )
    w = neuron.weight
    s = neuron.activation.s
    b = neuron.bias
    h = neuron.activation.breakpoints
    L = neuron.input_domain.lowerbound
    U = neuron.input_domain.upperbound
    n = length(w)
    k = length(h) - 1
    
    Δ = neuron.Δ
    H = neuron.H₁
    x_bar = [w[j] > 0 ? (U[j]-x[j])*w[j] : (L[j] - x[j])*w[j] for j in 1:n]
    opt_value1, k₁, p1 = optimal_ψ(x_bar, z, Δ, H)
    
    H = neuron.H₂ #increasing order
    x_bar = [w[j] < 0 ? (x[j]-U[j])*w[j] : (x[j] - L[j])*w[j] for j in 1:n]
    opt_value2, k₂, p2 = optimal_ψ(x_bar, z[k:-1:1], Δ, H) #reordering H also requires reordering z
    
    alpha = zeros(n)
    if opt_value1 >= -1e-6 && opt_value2 >= -1e-6
        return nothing
    elseif opt_value1 < 0
        for i in 1:k₁
            alpha[p1[i]] = w[p1[i]] > 0 ? -w[p1[i]] : (w[p1[i]] < 0 ? -w[p1[i]] : 0)
        end
        return alpha
    elseif opt_value2 < 0
        for i in 1:k₂
            alpha[p2[i]] = w[p2[i]] < 0 ? w[p2[i]] : (w[p2[i]] > 0 ? w[p2[i]] : 0)
        end
        return alpha
    end
end

function update_alpha!(
        neuron::Neuron{StaircaseFunction},
        x::Vector{Float64},
        y::Float64,
        z::Vector{Float64},
        alpha,
    )
    w = neuron.weight
    s = neuron.activation.s
    b = neuron.bias
    h = neuron.activation.breakpoints
    L = neuron.input_domain.lowerbound
    U = neuron.input_domain.upperbound
    n = length(w)
    k = length(h) - 1
    
    Δ = neuron.Δ
    H = neuron.H₁
    x_bar = [w[j] > 0 ? (U[j]-x[j])*w[j] : (L[j] - x[j])*w[j] for j in 1:n]
    opt_value1, k₁, p1 = optimal_ψ(x_bar, z, Δ, H)
    
    H = neuron.H₂ #increasing order
    x_bar = [w[j] < 0 ? (x[j]-U[j])*w[j] : (x[j] - L[j])*w[j] for j in 1:n]
    opt_value2, k₂, p2 = optimal_ψ(x_bar, z[k:-1:1], Δ, H) #reordering H also requires reordering z
    
    if opt_value1 >= -1e-6 && opt_value2 >= -1e-6
        return false
    end
    
    for (i, val) in enumerate(alpha)
        alpha[i] = 0
    end
        
    if opt_value1 < 0
        for i in 1:k₁
            alpha[p1[i]] = w[p1[i]] > 0 ? -w[p1[i]] : (w[p1[i]] < 0 ? -w[p1[i]] : 0)
        end
    else
        for i in 1:k₂
            alpha[p2[i]] = w[p2[i]] < 0 ? w[p2[i]] : (w[p2[i]] > 0 ? w[p2[i]] : 0)
        end
    end
        
    return true
end

function init_mip_model(
        neural_net::NeuralNetwork, 
        image::Vector{Float64},
        eps::Float64 = 0.01,
    )#only constant piecewise activation function
    
    neurons_by_layer = [length(bias) for bias in neural_net.biases] 
    pushfirst!(neurons_by_layer, size(neural_net.weights[1])[1]) #including both input & output layer
    nums_layer = length(neurons_by_layer)
    mip = Model(Gurobi.Optimizer)
    set_silent(mip)
    @variable(mip, x[i = 1:nums_layer-1, j = 1:neurons_by_layer[i]])
    #=for i in 2:nums_layer-1
        for j in 1:neurons_by_layer[i]
            @constraint(mip, min_act <= mip[:x][i, j] <= max_act)
        end
    end=#
    min_val = zeros(neurons_by_layer[1])
    max_val = zeros(neurons_by_layer[1])
    for (i, val) in enumerate(image)
        min_val[i] = max(val-eps, -1.0)
        max_val[i] = min(val+eps, 1.0)
        #@constraint(mip, max(val-eps, -1.0) <= mip[:x][1, i] <= min(val+eps, 1.0))
    end
    #@variable(mip, 0 <= z[i = 2:nums_layer-1, j = 1:neurons_by_layer[i], k = 1:num_pieces] <= 1)
    
    variable_neuron_dict = Dict()
    neuron_integervar_dict = Dict()
    for i in 1:nums_layer-2
        bias = neural_net.biases[i]
        weight = neural_net.weights[i]
        n, m = size(weight)
        for j in 1:n
            if min_val[j] > max_val[j]
                print(j)
            end
            @constraint(mip, min_val[j] <= mip[:x][i, j] <= max_val[j])
        end
        next_min_val = zeros(m)
        next_max_val = zeros(m)
        for j in 1:m
            w = Array{Float64}(weight[1:n, j])
            b = bias[j]
            l = sum([w[k] > 0 ? min_val[k]*w[k] : max_val[k]*w[k] for k in 1:n]) + b
            u = sum([w[k] > 0 ? max_val[k]*w[k] : min_val[k]*w[k] for k in 1:n]) + b
            f = neural_net.activation[i]
            i₁ = searchsortedfirst(f.breakpoints, l)-1
            i₂ = searchsortedfirst(f.breakpoints, u)-1
            next_min_val[j] = f.constant_terms[i₁]
            next_max_val[j] = f.constant_terms[i₂]
            bp = f.breakpoints[i₁:i₂]

            reduced_f = StaircaseFunction(f.breakpoints[i₁:i₂], f.slopes[i₁:i₂], f.constant_terms[i₁:i₂])
            reduced_f.breakpoints[1] = l

            push!(reduced_f.breakpoints, u)
            D = BoxDomain(min_val, max_val)
            neuron = Neuron(w, b, reduced_f, D)
            variable_neuron_dict[mip[:x][i+1, j]] = neuron
            neuron_integervar_dict[(i+1,j)] = i₂-i₁+1
            
            #@variable(mip, 0 <= z[i+1, j, k = 1:i₂-i₁+1] <= 1)
            #@constraint(mip, sum(mip[:z][i+1,j,k] for k = 1:i₂-i₁+1) == 1)
            #@constraint(mip, mip[:x][i+1,j] ==
            #            sum(f.constant_terms[k]*mip[:z][i+1,j,k] for k = 1:i₂-i₁+1))
        end
        min_val = next_min_val
        max_val = next_max_val
    end
    @variable(mip, 0 <= z[i = 2:nums_layer-1, j = 1:neurons_by_layer[i], 
              k = 1:neuron_integervar_dict[(i,j)]] <= 1)
    for i in 2:nums_layer-1
        for j in 1:neurons_by_layer[i]
            @constraint(mip, sum(mip[:z][i,j,k] for k = 1:neuron_integervar_dict[(i,j)]) == 1)
            cons_term = variable_neuron_dict[mip[:x][i,j]].activation.constant_terms
            num_intvar = neuron_integervar_dict[(i, j)]
            @constraint(mip, mip[:x][i,j] == sum(cons_term[k]*mip[:z][i,j,k] for k in 1:num_intvar))
        end
    end
    for j in 1:neurons_by_layer[end-1]
        @constraint(mip, -1 <= x[nums_layer-1, j] <= 1)
    end
    return mip, variable_neuron_dict, neuron_integervar_dict        
end

function init_mip_deeppoly(
        neural_net::NeuralNetwork, 
        image::Vector{Float64},
        eps::Float64 = 0.008,
    )#only constant piecewise activation function
    
    preact_layers = deep_poly(neural_net, image, eps)
    
    neurons_by_layer = [length(bias) for bias in neural_net.biases] 
    pushfirst!(neurons_by_layer, size(neural_net.weights[1])[1]) #including both input & output layer
    nums_layer = length(neurons_by_layer)
    mip = Model(Gurobi.Optimizer)
    set_silent(mip)
    @variable(mip, x[i = 1:nums_layer-1, j = 1:neurons_by_layer[i]])
    
    min_val = zeros(neurons_by_layer[1])
    max_val = zeros(neurons_by_layer[1])
    for (i, val) in enumerate(image)
         min_val[i] = max(val-eps, -1.0)
         max_val[i] = min(val+eps, 1.0)
        @constraint(mip, max(val-eps, -1.0) <= mip[:x][1, i] <= min(val+eps, 1.0))
    end
#     #@variable(mip, 0 <= z[i = 2:nums_layer-1, j = 1:neurons_by_layer[i], k = 1:num_pieces] <= 1)
    
    variable_neuron_dict = Dict()
    neuron_integervar_dict = Dict()
    for i in 1:nums_layer-2
        bias = neural_net.biases[i]
        weight = neural_net.weights[i]
        n, m = size(weight)
        L = [n.lower_bound for n in preact_layers[i]]
        U = [n.upper_bound for n in preact_layers[i]]
#         for j in 1:n
#             #if min_val[j] > max_val[j]
#             #    print(j)
#             #end
#             @constraint(mip, min_val[j] <= mip[:x][i, j] <= max_val[j])
#         end
        next_min_val = zeros(m)
        next_max_val = zeros(m)
        for j in 1:m
            w = Array{Float64}(weight[1:n, j])
            b = bias[j]
            l = L[j]
            u = U[j]
            f = neural_net.activation[i]
            i₁ = searchsortedfirst(f.breakpoints, l)-1
            i₂ = searchsortedfirst(f.breakpoints, u)-1
            next_min_val[j] = f.constant_terms[i₁]
            next_max_val[j] = f.constant_terms[i₂]
            bp = f.breakpoints[i₁:i₂]
            reduced_f = StaircaseFunction(f.breakpoints[i₁:i₂], f.slopes[i₁:i₂], f.constant_terms[i₁:i₂])
            reduced_f.breakpoints[1] = l
            push!(reduced_f.breakpoints, u)
            D = BoxDomain(min_val, max_val)
            neuron = Neuron(w, b, reduced_f, D)
            variable_neuron_dict[mip[:x][i+1, j]] = neuron
            neuron_integervar_dict[(i+1,j)] = i₂-i₁+1
            
        end
        min_val = next_min_val
        max_val = next_max_val
        for j in 1:m
            @constraint(mip, min_val[j] <= mip[:x][i+1, j] <= max_val[j])
        end
    end
    @variable(mip, 0 <= z[i = 2:nums_layer-1, j = 1:neurons_by_layer[i], k = 1:neuron_integervar_dict[(i,j)]] <= 1)
    for i in 2:nums_layer-1
        weight = neural_net.weights[i-1]
        bias = neural_net.biases[i-1]
        n, m = size(weight)
        for j in 1:m
            act_function = variable_neuron_dict[mip[:x][i, j]].activation
            w = Array{Float64}(weight[1:n, j])
            b = bias[j]
            @constraint(mip, sum(mip[:z][i,j,k]*act_function.breakpoints[k] for k in 1:neuron_integervar_dict[(i, j)]) <=
                            sum(w[p]*mip[:x][i-1,p] for p in 1:n) + b)
            @constraint(mip, sum(mip[:z][i,j,k]*act_function.breakpoints[k+1] for k in 1:neuron_integervar_dict[(i, j)]) >=
                            sum(w[p]*mip[:x][i-1,p] for p in 1:n) + b)
            @constraint(mip, sum(mip[:z][i,j,k] for k = 1:neuron_integervar_dict[(i,j)]) == 1)
            cons_term = variable_neuron_dict[mip[:x][i,j]].activation.constant_terms
            num_intvar = neuron_integervar_dict[(i, j)]
            @constraint(mip, mip[:x][i,j] == sum(cons_term[k]*mip[:z][i,j,k] for k in 1:num_intvar)) #only works with const pwl
        end
    end
    for j in 1:neurons_by_layer[end-1]
        @constraint(mip, -1 <= x[nums_layer-1, j] <= 1)
    end
    return mip, variable_neuron_dict, neuron_integervar_dict        
end

function target_attack(
        neural_net::NeuralNetwork, 
        image::Vector{Float64},
        true_label::Int64,
        target_label::Int64, 
        eps::Float64 = 0.01, 
        max_iter::Int64 = 1000,
    )
    # set objective for mip model
    num_cut = 0
    mip, variable_neuron_dict, neuron_integervar_dict = init_mip_deeppoly(neural_net, image, eps)
    last_layer = last(neural_net.weights)
    objective = zeros(10) # always 10 classes
    objective[target_label] = 1.0
    objective[true_label] = -1.0
    #objective = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
    c = last_layer * objective

    num_layers = length(neural_net.weights)
    final_dim, output_dim = size(last_layer)
    @objective(mip, Max, sum(c[i]*mip[:x][num_layers, i] for i in 1:final_dim))
    
    neurons_by_layer = [length(bias) for bias in neural_net.biases] #including input & output layer
    pushfirst!(neurons_by_layer, size(neural_net.weights[1])[1])
    pop!(neurons_by_layer)
    
    # separation procedure
    feasible = false
    count = 0
    generated_alpha = Dict()
    for (key, value) in variable_neuron_dict
        generated_alpha[key] = Set()
    end
    while !feasible && count < max_iter
        #@printf("solving %d-th problem: \n", count+1)
        optimize!(mip)
        x_val = [value.(mip[:x][i, k] for k in 1:neurons_by_layer[i]) for i in 1:length(neurons_by_layer)]
        z_val = [[value.(mip[:z][i, j, k] for k in 1:neuron_integervar_dict[(i, j)]) for j in 1:neurons_by_layer[i]] 
                  for i in 2:num_layers]
        
        #TODO: parallel this part?
        feasible = true
        for i in 1:num_layers-1
            bias = neural_net.biases[i]
            weight = neural_net.weights[i]
            n, m = size(weight)
            #@printf("   generate violating inequalities of layer %d: \n   ", i+1)
            for j in 1:m
                num_pieces = neuron_integervar_dict[(i+1,j)]
                z = [z_val[i][j][k] for k in 1:num_pieces]
                
                fractional = false
                for val in z
                    if val > 1e-6 && val < 1-1e-6
                        fractional = true
                        break
                    end
                end

                if !fractional
                    continue
                end
                y = x_val[i+1][j]
                x = x_val[i]
                neuron = variable_neuron_dict[mip[:x][i+1, j]]
                alpha = generate_alpha(neuron, x, y, z)
                ## generates cuts
                if alpha != nothing && !(alpha in generated_alpha[mip[:x][i+1, j]])
                    push!(generated_alpha[mip[:x][i+1, j]], alpha)
                    num_cut += 2
                    upper_z = generate_zcoef_from_alpha(neuron, alpha, GT(y))
                    lower_z = generate_zcoef_from_alpha(neuron, alpha, LT(y))
                    @constraint(mip, mip[:x][i+1, j] <= sum(mip[:x][i, k]*alpha[k] for k in 1:n) +
                                     sum(mip[:z][i+1, j, p]*upper_z[p] for p in 1:num_pieces)) 
                    @constraint(mip, mip[:x][i+1, j] >= sum(mip[:x][i, k]*alpha[k] for k in 1:n) +
                                     sum(mip[:z][i+1, j, p]*lower_z[p] for p in 1:num_pieces))
                    feasible = false
                end
            end
        end
        count += 1
    end
    return objective_value(mip), value.(mip[:x]), value.(mip[:z]), mip
end