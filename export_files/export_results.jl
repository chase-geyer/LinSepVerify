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


function collect_files(directory)
    return [joinpath(directory, f) for f in readdir(directory) if isfile(joinpath(directory, f)) && endswith(f, ".pkl")]
end
#print("Model\tUpper_Bound\tLower_Bound\tEpsilon\n")
verification_type = "cayley"
output_folder = "cayley_outputs"
images = Pickle.load(open("./imgs/MNIST_images-for-verification", "r+"))
for file in collect_files("./models")
    dorefa_int = parse(Int, file[22])
    name_of_time_output = "./final/" * output_folder * "/time_values/results_large_dorefa_$dorefa_int.txt"
    name_of_obj_output =  "./final/" * output_folder * "/objective_gaps/results_large_dorefa_$dorefa_int.txt"
    if length(file) > 45
        name_of_time_output = "./final/" * output_folder * "/time_values/results_large_dorefa_double_$dorefa_int.txt"
        name_of_obj_output = "./final/" * output_folder * "/objective_gaps/results_large_dorefa_double_$dorefa_int.txt"
    end
    open(name_of_time_output, "w") do output_file
        write(output_file, "Img\tTime(s)\tEpsilon\n")
    end
    open(name_of_obj_output, "w") do output_file
        write(output_file, "Img\tTarget_Label\tEpsilon\tObjective_Value\tLabel\n")
    end
    for eps in [0.025, 0.05, 0.075, 0.1]
        println("file: $file")
        net_from_pickle = Pickle.load(file)
        #println(dorefa_int)
        f = dorefa_to_staircase(dorefa_int)
        if length(file) > 45
            activation = [f, f, f]
        else
            activation = [f, f]
        end
        neural_net = NeuralNetwork(net_from_pickle, activation)
        upper_bound = 150
        lower_bound = 0
        count = 1
        for (raw_img, label) in collect(zip(images, labels))
            ## can run these individually since it's 1:1 image to target_attack
            opt_val = 0
            img = load_image(raw_img)
            vulnerable = false
            start_time = now()
            for target_label in 1:10
                if target_label != label
                    @suppress begin
                        opt_val, opt_sol_x, opt_sol_z = target_attack(neural_net, img, label, target_label, eps)
                        if opt_val > 0
                            vulnerable = true
                            adv_img = [opt_sol_x[1, j] for j in 1:784]
                            pred = predict(neural_net, adv_img)
                            if pred != label
                                upper_bound -= 1
                                break
                            end
                        end
                    end
                    open(name_of_obj_output, "a") do output_file
                        output_target_label = target_label - 1
                        output_label = label-1
                        write(output_file, "$count\t$output_target_label\t$eps\t$opt_val\t$output_label\n")
                    end
                end
            end

            if vulnerable == false
                lower_bound += 1
            end
            end_time = now()
            elapsed_time = Dates.value(end_time - start_time) / (1000)
            #print("img $count: $elapsed_time\n")
            open(name_of_time_output, "a") do output_file
                write(output_file, "$count\t$elapsed_time\t$eps\n")
            end
            count += 1
        end
        
        open("./final/results.txt", "a") do output_file
            write(output_file, "$file\t$upper_bound\t$lower_bound\t$eps\t$verification_type\n")
            #print("$file\t$upper_bound\t$lower_bound\t$eps\n")
        end
    end
end