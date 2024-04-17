using TimerOutputs
using GLMakie
using Pickle
using Suppressor
using Printf
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
    n = Int(sqrt(length(img_array)))
    return Base.reshape(img_array, (n, n))
end

raw_imgs = Pickle.load(open("./imgs/MNIST_images-for-verification"))
imgs = []
for img in raw_imgs
    img = vcat([w' for w in img] ...)
    img = vcat(img'...)
    push!(imgs, img)
end
labels = Pickle.load(open("./imgs/MNIST_labels-for-verification", "r+"))
labels = [l+1 for l in labels]

function collect_files(directory)
    return [joinpath(directory, f) for f in readdir(directory) if isfile(joinpath(directory, f))]
end

print("images loaded\n")
write("results.txt", "Model\tUpper Bound\tLower Bound\tTime\tEpsilon\n")

for eps in [0.008, 0.016, 0.024, 0.032]
    for file in collect_files("./models")
        net_from_pickle = Pickle.load(open(file))
        f = dorefa_to_staircase(Int(file[22]))
        activation = [f, f]
        neural_net = NeuralNetwork(net_from_pickle, activation)
        start_time = now()
        upper_bound = 150
        lower_bound = 0
        count = 1
        for (img, label) in zip(imgs, labels)
            ## can run these individually since it's 1:1 image to target_attack
            vulnerable = false
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
                end
            end
            if vulnerable == false
                lower_bound += 1
            end
            count += 1
        end
        end_time = now()
        elapsed_time = Dates.value(end_time - start_time) / (1000 * 60)
        open("results.txt", "a") do output_file
            write(output_file, "$file\t$upper_bound\t$lower_bound\t$elapsed_time\t$eps\n")
        end
    end
end