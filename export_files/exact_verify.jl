using Printf
import Pkg
Pkg.add("Pickle")
Pkg.add("TimerOutputs")
Pkg.add("Suppressor")
Pkg.add("Dates")
Pkg.add("MathOptInterface")
Pkg.add("JuMP")
Pkg.add("Gurobi")
using TimerOutputs
using Pickle
using Suppressor
using Dates

include("../CayleyVerify.jl")
include("../DeepPoly.jl")


function dorefa_to_staircase(k::Int)
    n = 2^k - 1
    slopes = zeros(n+1)
    breakpoints = [-Inf]
    for i in 1:n
        push!(breakpoints, (2*i-1)/n - 1)
    end
    push!(breakpoints, Inf)
    
    constant_terms = [-1.0]
    for i in 1:n
        push!(constant_terms, -1.0 + 2*i/n)
    end
    return StaircaseFunction(breakpoints, slopes, constant_terms)
end

function predict(neural_net, img)
    num_layers = length(neural_net.weights)
    a = img'
    for i in 1:num_layers
        a = a * neural_net.weights[i] + neural_net.biases[i]'
        if i <= num_layers -1
            a = [eval(neural_net.activation[i], a[j]) for j in 1:length(a)]'
        end
    end
    output = a'
    return findmax(output)[2]
end
function reshape(img_array)
    n = Int16(sqrt(length(img_array)))
    return Base.reshape(img_array, (n, n))
end

function load_image(img)
    img = vcat([w' for w in img] ...)
    return vcat(img'...)
end


labels_file = open("./imgs/MNIST_labels-for-verification", "r+")
labels = Pickle.load(labels_file)
close(labels_file)
labels = [(l+1) for l in labels]

raw_imgs = Pickle.load("./imgs/MNIST_images-for-verification")
imgs = [reshape(img) for img in raw_imgs]


t = @elapsed mip, variable_neuron_dict, neuron_integervar_dict = init_mip_deeppoly(neural_net, imgs[1], 0.008)
true_label = labels[1]
target_label = 2

last_layer = last(neural_net.weights)
objective = zeros(10)
objective[target_label] = 1.0
objective[true_label] = -1.0
c = last_layer * objective

for z in mip[:z]
    set_binary(z)
end

num_layers = length(neural_net.weights)
final_dim, output_dim = size(last_layer)
@objective(mip, Max, sum(c[i]*mip[:x][num_layers, i] for i in 1:final_dim))
@time optimize!(mip)


mip, variable_neuron_dict, neuron_integervar_dict = init_mip_deeppoly(neural_net, imgs[1], 0.008)
true_label = labels[1]
target_label = 2

last_layer = last(neural_net.weights)
objective = zeros(10) # always 10 classes
objective[target_label] = 1.0
objective[true_label] = -1.0
c = last_layer * objective

num_layers = length(neural_net.weights)
final_dim, output_dim = size(last_layer)
@objective(mip, Max, sum(c[i]*mip[:x][num_layers, i] for i in 1:final_dim))

neurons_by_layer = [length(bias) for bias in neural_net.biases] #including input & output layer
pushfirst!(neurons_by_layer, size(neural_net.weights[1])[1])
pop!(neurons_by_layer)

for z in mip[:z]
    set_binary(z)
end

alphas = []
for n in neurons_by_layer
    push!(alphas, zeros(n))
end


allocation1 = []
allocation2 = []
count = [0]
function callback_cut(cb_data)
    count[1] += 1
    if count[1]%100 == 0
        x_val = callback_value.(Ref(cb_data), mip[:x])
        z_val = callback_value.(Ref(cb_data), mip[:z])
        for i in 1:num_layers-1
            bias = neural_net.biases[i]
            weight = neural_net.weights[i]
            n, m = size(weight)
            for j in 1:m
                num_pieces = neuron_integervar_dict[(i+1,j)]
                z = [z_val[i+1,j,k] for k in 1:num_pieces]
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
                y = x_val[i+1, j]
                x = [x_val[i, k] for k in 1:n]
                neuron = variable_neuron_dict[mip[:x][i+1, j]]
                t1 = @allocated update = update_alpha!(neuron, x, y, z, alphas[i])
                if update
                    t2 = @allocated upper_z = generate_zcoef_from_alpha(neuron, alphas[i], GT(y))
                    t3 = @allocated lower_z = generate_zcoef_from_alpha(neuron, alphas[i], LT(y))
                    upper_con = @build_constraint(mip[:x][i+1, j] <= sum(mip[:x][i, k]*alphas[i][k] for 
                                k in 1:n) + sum(mip[:z][i+1, j, p]*upper_z[p] for p in 1:num_pieces))
                    lower_con = @build_constraint(mip[:x][i+1, j] >= sum(mip[:x][i, k]*alphas[i][k] for 
                                k in 1:n) + sum(mip[:z][i+1, j, p]*lower_z[p] for p in 1:num_pieces))
                    MOI.submit(mip, MOI.UserCut(cb_data), upper_con)
                    MOI.submit(mip, MOI.UserCut(cb_data), lower_con)
                    push!(allocation2, t2+t3)
                end
                push!(allocation1, t1)
            end
        end
    end
end

MOI.set(mip, MOI.UserCutCallback(), callback_cut)
@time optimize!(mip)

function verify(neural_net, image, true_label, objective, eps = 0.01, max_iter = 1000)
    # set objective for mip model
    mip, variable_neuron_dict = init_mip_model(neural_net, image, true_label, eps)
    last_layer = last(neural_net)[1]
    c = last_layer * objective
    #=for i in 2:10
        @constraint(mip, sum(last_layer[1:100, i][j]*mip[:x][3, j] for j in 1:100) <=
                         sum(last_layer[1:100, i][1]*mip[:x][3, j] for j in 1:100))
    end=#
    num_layers = length(neural_net)
    final_dim, output_dim = size(last(neural_net)[1])
    @objective(mip, Max, sum(c[i]*mip[:x][num_layers, i] for i in 1:final_dim))
    
    # separation procedure
    feasible = false
    count = 0
    generated_alpha = Dict()
    for (key, value) in variable_neuron_dict
        generated_alpha[key] = Set()
    end
    while !feasible && count < max_iter
        @printf("solving %d-th problem: \n", count+1)
        @time optimize!(mip)
        #TODO: parallel this part?
        feasible = true
        for (i, (weights, bias)) in enumerate(neural_net[1:num_layers-1])
            n, m = size(weights)
            @printf("   generate violating inequalities of layer %d: \n   ", i+1)
            @time for j in 1:m
                y = value(mip[:x][i+1, j])
                x = value.([mip[:x][i, k] for k in 1:n])
                z = value.([mip[:z][i+1,j,1], mip[:z][i+1,j,2]])
                neuron = variable_neuron_dict[mip[:x][i+1, j]]
                alpha = generate_alpha(neuron, x, y, z)
                if alpha !== nothing && !(alpha in generated_alpha[mip[:x][i+1, j]])
                    push!(generated_alpha[mip[:x][i+1, j]], alpha)
                    upper_z = generate_zcoef_from_alpha(neuron, alpha, GT(y))
                    lower_z = generate_zcoef_from_alpha(neuron, alpha, LT(y))
                    @constraint(mip, mip[:x][i+1, j] <= sum(mip[:x][i, k]*alpha[k] for k in 1:n) +
                                     sum(mip[:z][i+1, j, k]*upper_z[k] for k in 1:2)) 
                    @constraint(mip, mip[:x][i+1, j] >= sum(mip[:x][i, k]*alpha[k] for k in 1:n) +
                                     sum(mip[:z][i+1, j, k]*lower_z[k] for k in 1:2))
                    feasible = false
                end
            end
        end
        count += 1
    end
    return objective_value(mip), value.(mip[:x]), value.(mip[:z])
end