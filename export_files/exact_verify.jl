import MathOptInterface, Pickle
using Profile, ProfileView
using LinearAlgebra
using JuMP, Gurobi, Test, Ipopt, Juniper, Suppressor, Dates
using Images, ImageView
const GT = MOI.GreaterThan{Float64}
const LT = MOI.LessThan{Float64}

include("../CayleyVerify.jl")
include("../DeepPoly.jl")


function dorefa_to_staircase(k::Int)
    n = 2^k - 1
    slopes = zeros(n + 1)
    breakpoints = [-Inf]
    for i in 1:n
        push!(breakpoints, (2 * i - 1) / n - 1)
    end
    push!(breakpoints, Inf)

    constant_terms = [-1.0]
    for i in 1:n
        push!(constant_terms, -1.0 + 2 * i / n)
    end
    return StaircaseFunction(breakpoints, slopes, constant_terms)
end

function predict(neural_net, img)
    num_layers = length(neural_net.weights)
    a = img'
    for i in 1:num_layers
        a = a * neural_net.weights[i] + neural_net.biases[i]'
        if i <= num_layers - 1
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
    img = vcat([w' for w in img]...)
    return vcat(img'...)
end
function get_neural(path, k)
    net_from_pickle = Pickle.load(open(path))
    f = dorefa_to_staircase(k)
    activation = [f, f]
    print("net loaded\n")
    return NeuralNetwork(net_from_pickle, activation)
end

dorefa2neural = get_neural("../models/MNIST-DoReFa2_Dense256-Dense256.pkl", 2)
dorefa3neural = get_neural("../models/MNIST-DoReFa3_Dense256-Dense256.pkl", 3)
dorefa4neural = get_neural("../models/MNIST-DoReFa4_Dense256-Dense256.pkl", 4)


raw_imgs = Pickle.load(open("../imgs/MNIST_images-for-verification"))
imgs = []
for img in raw_imgs
    img = vcat([w' for w in img]...)
    img = vcat(img'...)
    push!(imgs, img)
end
labels = Pickle.load(open("../imgs/MNIST_labels-for-verification"))
labels = [l + 1 for l in labels]
print("images loaded\n")

image = reshape(imgs[12])
normalized_image = (image .+ 1) ./ 2  # Scale to [0, 1]
save("test.png", normalized_image)

logFilePath = "finalResultLog.txt"
graphFilePath = "final_graph_log.txt"
write(graphFilePath, "obj\tlowerBound\ttime\timg\tmethod\n")
function run_m(neural_net, eps, timelimit, imgIdx, cut_freq)
    mip, variable_neuron_dict, neuron_integervar_dict = init_mip_deeppoly(neural_net, imgs[imgIdx], eps)
    true_label = labels[imgIdx]
    target_label = 2
    set_attribute(mip, "output_flag", false)
    set_optimizer_attribute(mip, "TimeLimit", timelimit) # Set a time limit of 300 seconds
    last_layer = last(neural_net.weights)
    objective = zeros(10) # always 10 classes
    objective[target_label] = 1.0
    objective[true_label] = -1.0
    c = last_layer * objective

    # Define the objective function in a more concise manner
    num_layers = length(neural_net.weights)
    final_dim, _ = size(last_layer) # Assuming output_dim is not used elsewhere
    @objective(mip, Max, sum(c[i] * mip[:x][num_layers, i] for i in 1:final_dim))

    # Efficiently compute neurons_by_layer without mutating the array
    neurons_by_layer = [size(neural_net.weights[1])[1]; [length(bias) for bias in neural_net.biases][1:end-1]]

    # Set integer constraint on z variables in a single loop
    foreach(set_integer, mip[:z])

    # Preallocate alphas with the correct dimensions and fill with zeros
    alphas = [zeros(n) for n in neurons_by_layer]
    start_time = now()
    count = [0]
    optimals = [0]
    function callback_cut(cb_data, cb_where)
        count[1] += 1

        if cb_where == Gurobi.GRB_CB_MIPSOL
            optimals[1] += 1
            println("big-m solution found better objective\n")
            node_status = Ref{Cint}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_STATUS, node_status)
            println("node status: ", node_status[])
            # println("GRB_CB_MIPNODE", Gurobi.GRB_CB_MIPNODE)
            # println("GRB_CB_MIPSOL", Gurobi.GRB_CB_MIPSOL)
            # println("GRB_CB_MIPNODE_OBJBST", Gurobi.GRB_CB_MIPNODE_OBJBST)
            # println("GRB_OPTIMAL", Gurobi.GRB_OPTIMAL)
            if node_status[] != 0
                Gurobi.load_callback_variable_primal(cb_data, cb_where)
                current_time = now()
                time = Dates.value(current_time - start_time) / (1000)
                obj_val = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBST, obj_val)
                dual_bound = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBND, dual_bound)
                open(graphFilePath, "a") do f
                    write(f, "$obj_val\t$dual_bound\t$time\t$imgIdx\tbig-m\t$cut_freq\n")
                end
            end
        end
    end
    MOI.set(mip, Gurobi.CallbackFunction(), callback_cut)
    optimize!(mip)
 
    # print("fractional calls: ", fractional_calls[1], "\n")
    # output = split(output," ")[2]
    print("solution status", termination_status(mip))
    if true
        # if termination_status(mip) ==
        #     open(logFilePath, "a") do f
        #         write(f, "Optimization did not converge")
        #     end
        #     return
        # end
        try
            nodes_explored = node_count(mip)
            lb = dual_objective_value(mip)
            obj = objective_value(mip)
            gap = relative_gap(mip)
        catch e
            open(logFilePath, "a") do file
                write(file, "No solutions found\n")
            end
            return;
        end
        nodes_explored = node_count(mip)
        lb = dual_objective_value(mip)
        obj = objective_value(mip)
        gap = relative_gap(mip)
        open(logFilePath, "a") do file
            write(file, "$gap\t$eps\t$obj\t$lb\t$timelimit\t$nodes_explored\tbig-m\t$imgIdx\n")
        end
    end
    # return output
end


function run_cayley_with_callback(neural_net, eps, timelimit, imgIdx, cut_freq)
    mip, variable_neuron_dict, neuron_integervar_dict = init_mip_deeppoly(neural_net, imgs[imgIdx], eps)
    true_label = labels[imgIdx]
    target_label = 2
    set_attribute(mip, "output_flag", false)
    set_optimizer_attribute(mip, "TimeLimit", timelimit) # Set a time limit of 300 seconds
    # set_optimizer_attribute(mip , "Presolve", 0) # Turn off presolve
    last_layer = last(neural_net.weights)
    objective = zeros(10) # always 10 classes
    objective[target_label] = 1.0
    objective[true_label] = -1.0
    # Calculate 'c' using matrix multiplication for vectorization
    c = last_layer * objective

    # Define the objective function in a more concise manner
    num_layers = length(neural_net.weights)
    final_dim, _ = size(last_layer) # Assuming output_dim is not used elsewhere
    @objective(mip, Max, sum(c[i] * mip[:x][num_layers, i] for i in 1:final_dim))

    # Efficiently compute neurons_by_layer without mutating the array
    neurons_by_layer = [size(neural_net.weights[1])[1]; [length(bias) for bias in neural_net.biases][1:end-1]]

    # Set integer constraint on z variables in a single loop
    foreach(set_integer, mip[:z])

    # Preallocate alphas with the correct dimensions and fill with zeros
    alphas = [zeros(n) for n in neurons_by_layer]

    #allocation1 = []
    #allocation2 = []
    start_time = now()
    count = [0]
    sols = [0]
    function callback_cut(cb_data, cb_where)
        count[1] += 1
        # output = []
        # push!(output, MOI.get(mip, MOI.NodeCount()))
        current_time = now()
        # print(cb_where)
        # push!(output, callback_value(cb_data, mip), Dates.value(current_time - start_time) / (1000))
        if cb_where == Gurobi.GRB_CB_MIPSOL
            sols[1] += 1
            print("MIPSOL_OBJ\n")
            node_status = Ref{Cint}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_STATUS, node_status)
            if node_status[] != Gurobi.GRB_OPTIMAL
                Gurobi.load_callback_variable_primal(cb_data, cb_where)
                current_time = now()
                time = Dates.value(current_time - start_time) / (1000)
                obj_val = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBST, obj_val)
                dual_bound = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBND, dual_bound)
                open(graphFilePath, "a") do f
                    write(f, "$obj_val\t$dual_bound\t$time\t$imgIdx\tcayley\n")
                end
            end
        end
        if cb_where == Gurobi.GRB_CB_MIPNODE_OBJBST &&  Dates.value(current_time - start_time) / (1000) < timelimit * 0.5 && count[1] % cut_freq == 0
            # try
                x_val = callback_value.(Ref(cb_data), mip[:x])
                z_val = callback_value.(Ref(cb_data), mip[:z])
                println(z_val) ## Since we're only adding cut when there is a fractional z value

                for i in 1:num_layers-1
                    bias = neural_net.biases[i]
                    weight = neural_net.weights[i]
                    n, m = size(weight)
                    for j in 1:m
                        num_pieces = neuron_integervar_dict[(i + 1, j)]
                        z = [z_val[i+1, j, k] for k in 1:num_pieces]
                        fractional = any(1e-6 < val < 1 - 1e-6 for val in z)
                        if !fractional
                            continue
                        end
                        y = x_val[i+1, j]
                        x = [x_val[i, k] for k in 1:n]
                        neuron = variable_neuron_dict[mip[:x][i+1, j]]
                        update = update_alpha!(neuron, x, y, z, alphas[i])
                        if update
                            upper_z = generate_zcoef_from_alpha(neuron, alphas[i], GT(y))
                            lower_z = generate_zcoef_from_alpha(neuron, alphas[i], LT(y))
                            upper_con = @build_constraint(mip[:x][i+1, j] <= sum(mip[:x][i, k] * alphas[i][k] for
                                                                                k in 1:n) + sum(mip[:z][i+1, j, p] * upper_z[p] for p in 1:num_pieces))
                            lower_con = @build_constraint(mip[:x][i+1, j] >= sum(mip[:x][i, k] * alphas[i][k] for
                                                                                k in 1:n) + sum(mip[:z][i+1, j, p] * lower_z[p] for p in 1:num_pieces))
                            MOI.submit(mip, MOI.UserCut(cb_data), upper_con)
                            MOI.submit(mip, MOI.UserCut(cb_data), lower_con)
                            #push!(allocation2, t2+t3)
                        end
                        #push!(time, [time_update, time_uzcoef, time_lzcoef])
                    end
                end
            # catch e
            #     println("Error caught", e)
            # end
        end
    end
    if size(graph) != 0
        open("graph.txt", "a") do f
            write(f, "$graph\n")
        end
    end
    
    #record opt gap and nodes using verbose model output
    MOI.set(mip, Gurobi.CallbackFunction(), callback_cut)

    optimize!(mip)
    println("num of user callbacks: ", count[1])
    print("solution status", termination_status(mip))
    #print("fractional calls: ", fractional_calls[1], "\n")
    if true
        if sols[1] == 0
            open(logFilePath, "a") do f
                write(f, "No solutions found\n")
            end
            return
        end
        try
            nodes_explored = node_count(mip)
            lb = dual_objective_value(mip)
            obj = objective_value(mip)
            gap = relative_gap(mip)
        catch e
            open(logFilePath, "a") do f
                write(f, "No solutions found\n")
            end
            return
        end
        nodes_explored = node_count(mip)
        lb = dual_objective_value(mip)
        obj = objective_value(mip)
        gap = relative_gap(mip)
         
        open(logFilePath, "a") do f
            write(f, "$gap\t$eps\t$obj\t$lb\t$timelimit\t$nodes_explored\tcayley\t$imgIdx\t$cut_freq\n")
        end
    end
    #return output
    #return time
end



function run_cayley_from_lp_relax(neural_net, eps, timelimit, imgIdx, cut_freq)

    opt_val, opt_sol_x, opt_sol_z, mip = target_attack(neural_net, imgs[imgIdx], labels[imgIdx], 2, eps)
    set_attribute(mip, "output_flag", false)
    set_optimizer_attribute(mip, "TimeLimit", timelimit) 
    # Set integer constraint on z variables in a single loop
    foreach(set_integer, mip[:z])

    start_time = now()
    count = [0]
    sols = [0]
    function callback_cut(cb_data, cb_where)
        count[1] += 1
        current_time = now()
        if cb_where == Gurobi.GRB_CB_MIPSOL
            sols[1] += 1
            # print("cayley found better obj\n")
            node_status = Ref{Cint}()
            Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_STATUS, node_status)
            println("node status: ", node_status[])
            if node_status[] != 0
                Gurobi.load_callback_variable_primal(cb_data, cb_where)
                current_time = now()
                time = Dates.value(current_time - start_time) / (1000)
                obj_val = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBST, obj_val)
                dual_bound = Ref{Cdouble}()
                Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPNODE_OBJBND, dual_bound)
                open(graphFilePath, "a") do f
                    write(f, "$obj_val\t$dual_bound\t$time\t$imgIdx\tcayley\n")
                end
            end
        end
    end
    
    #record opt gap and nodes using verbose model output
    MOI.set(mip, Gurobi.CallbackFunction(), callback_cut)

    optimize!(mip)
    println("num of user callbacks: ", count[1])
    print("solution status", termination_status(mip))
    #print("fractional calls: ", fractional_calls[1], "\n")
    if true
        if sols[1] == 0
            open(logFilePath, "a") do f
                write(f, "No solutions found\n")
            end
            return
        end
        try
            nodes_explored = node_count(mip)
            lb = dual_objective_value(mip)
            obj = objective_value(mip)
            gap = relative_gap(mip)
        catch e
            open(logFilePath, "a") do f
                write(f, "No solutions found\n")
            end
            return
        end
        nodes_explored = node_count(mip)
        lb = dual_objective_value(mip)
        obj = objective_value(mip)
        gap = relative_gap(mip)
         
        open(logFilePath, "a") do f
            write(f, "$gap\t$eps\t$obj\t$lb\t$timelimit\t$nodes_explored\tcayley\t$imgIdx\t$cut_freq\n")
        end
    end
end

@suppress begin
    run_m(dorefa2neural, 0.001, 100, 1, 10)
    run_cayley_from_lp_relax(dorefa2neural, 0.001, 100, 1, 10)
end

count = 0
while count < 10
    ## FINDING VERIFIABLE IMAGES
    opt_val = 0
    rand_img_idx = rand(1:length(imgs))
    img = imgs[rand_img_idx]
    label = labels[rand_img_idx]
    vulnerable = false
    for target_label in 1:10
        if target_label != label
            @suppress begin
                opt_val, opt_sol_x, opt_sol_z, mip = target_attack(dorefa4neural, img, label, target_label, 0.05)
                if opt_val > 0
                    vulnerable = true
                    adv_img = [opt_sol_x[1, j] for j in 1:784]
                end
            end
        end
    end

    if vulnerable == false
        global count +=1
        for pair in [[0.05, 1000, 100]]
                for network in [dorefa2neural, dorefa3neural, dorefa4neural]
                    run_m(network, pair[1], pair[2], rand_img_idx, pair[3])
                    run_cayley_from_lp_relax(network, pair[1], pair[2], rand_img_idx, pair[3])
                end
            # end
        end
    end
end
