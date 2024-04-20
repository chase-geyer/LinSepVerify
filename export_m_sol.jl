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

include("CayleyVerify.jl")
include("DeepPoly.jl")

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

function collect_files(directory)
    return [joinpath(directory, f) for f in readdir(directory) if isfile(joinpath(directory, f)) && endswith(f, ".pkl")]
end
GC.gc()
write("big_m_results.txt", "Model\tUpper Bound\tLower Bound\tTime\tEpsilon\n")
#print("Model\tUpper_Bound\tLower_Bound\tEpsilon\n")

for file in collect_files("./models")
    dorefa_int = parse(Int, file[22])
    name_of_output = "./big_m_results/results_dorefa_$dorefa_int.txt"
    open(name_of_output, "w") do output_file
        write(output_file, "Img\tTime(s)\tEpsilon\n")
    end
    for eps in [0.008, 0.016, 0.024, 0.032]
        #println("file: $file")
        model = open(file)
        net_from_pickle = Pickle.load(model)
        close(model)
        #println(dorefa_int)
        f = dorefa_to_staircase(dorefa_int)
        activation = [f, f]
        neural_net = NeuralNetwork(net_from_pickle, activation)
        upper_bound = 150
        lower_bound = 0
        count = 1
        for target_label in 1:10
            if target_label != label
                @suppress begin
                    mip, _, _ = init_mip_deeppoly(neural_net, img, 0.024)
                    last_layer = last(neural_net.weights)
                    objective = zeros(10) # always 10 classes
                    objective[target_label] = 1.0
                    objective[label] = -1.0
                    c = last_layer * objective
    
                    num_layers = length(neural_net.weights)
                    final_dim, output_dim = size(last_layer)
                    @objective(mip, Max, sum(c[i]*mip[:x][num_layers, i] for i in 1:final_dim))
                    optimize!(mip)
                    opt_val = objective_value(mip)
                    if opt_val > 0
                        vulnerable = true
                        break
                    end
                end
            end
            if vulnerable == false
                lower_bound += 1
            end
            end_time = now()
            elapsed_time = Dates.value(end_time - start_time) / (1000)
            #print("img $count: $elapsed_time\n")
            open(name_of_output, "a") do output_file
                write(output_file, "$count\t$elapsed_time\t$eps\n")
            end
            count += 1
        end
            
        
        open("big_m_results.txt", "a") do output_file
            write(output_file, "$file\t$upper_bound\t$lower_bound\t$eps\n")
            #print("$file\t$upper_bound\t$lower_bound\t$eps\n")
        end
    end
end
