abstract type DeepPolyNeuron end

struct ActNeuron <: DeepPolyNeuron
    ub_weight::Float64
    ub_bias::Float64
    lb_weight::Float64
    lb_bias::Float64
    lower_bound::Float64
    upper_bound::Float64
    pre_act::DeepPolyNeuron
end

struct PreActNeuron <: DeepPolyNeuron
    weight::Vector{Float64}
    bias::Float64
    lower_bound::Float64
    upper_bound::Float64
    prev_layer::Vector{DeepPolyNeuron}
end

struct InputNeuron <: DeepPolyNeuron
    lower_bound::Float64
    upper_bound::Float64
end

function create_act_neuron(
        preact_neuron::PreActNeuron,
        f::StaircaseFunction,
    )
    l = preact_neuron.lower_bound
    u = preact_neuron.upper_bound
    
    i₁ = searchsortedfirst(f.breakpoints, l)
    i₂ = searchsortedfirst(f.breakpoints, u)
    @assert i₂ >= i₁
    
    if i₁ == i₂
        val = f.constant_terms[i₁-1]
        return ActNeuron(0.0, val, 0.0, val, val, val, preact_neuron)
    elseif i₂ == i₁ + 1
        bp = f.breakpoints[i₁]
        u_gap = u - bp
        l_gap = bp - l
        up_val = f.constant_terms[i₂ - 1]
        low_val = f.constant_terms[i₁ - 1]
        
        ub_weight = u_gap > l_gap ? 0 : (up_val - low_val)/l_gap
        ub_bias = u_gap > l_gap ? up_val : up_val - (up_val - low_val)*bp/l_gap
        
        lb_weight = u_gap > l_gap ? (up_val - low_val)/u_gap : 0
        lb_bias = u_gap > l_gap ? low_val - (up_val - low_val)*bp/u_gap : low_val
        act_neuron = ActNeuron(ub_weight, ub_bias, lb_weight, lb_bias, low_val, up_val, preact_neuron)
    else # assuming step size h
        h = f.breakpoints[i₁+1] - f.breakpoints[i₁]
        v = f.constant_terms[i₁+1] - f.constant_terms[i₁]
        bp₁ = f.breakpoints[i₁]
        bp₂ = f.breakpoints[i₂-1]
        u_gap = u - bp₂
        l_gap = bp₁ - l
        
        up_val = f.constant_terms[i₂-1]
        low_val = f.constant_terms[i₁-1]
        
        lb_weight = u_gap > h ? (up_val - low_val)/(u - bp₁) : h/v
        lb_bias = u_gap > h ? low_val - (up_val - low_val)*bp₁/(u - bp₁) : low_val - h*bp₁/v
        
        ub_weight = l_gap > h ? (up_val - low_val)/(bp₂ - l) : h/v
        ub_bias = l_gap > h ? up_val - (up_val - low_val)*bp₂/(bp₂ - l) : up_val - h*bp₂/v
        act_neuron = ActNeuron(ub_weight, ub_bias, lb_weight, lb_bias, low_val, up_val, preact_neuron)
    end
    
    return act_neuron
end

function upperbound_weight_propagate(
        act_layer::Vector{ActNeuron},
        weight::Vector{Float64}, # with respect to activation layer
        bias::Float64,
    )
    # weight and bias with respect to preactivation nodes    
    preact_weight = [weight[i] > 0 ? weight[i]*act_layer[i].ub_weight : weight[i]*act_layer[i].lb_weight for 
                    i in 1:length(act_layer)]
    preact_bias = bias+sum(weight[i] > 0 ? weight[i]*act_layer[i].ub_bias : weight[i]*act_layer[i].lb_bias for 
                          i in 1:length(act_layer))
    
    preact_layer = [act_layer[i].pre_act for i in 1:length(act_layer)]
    prev_layer = act_layer[1].pre_act.prev_layer
    
    while !(prev_layer[1] isa InputNeuron)
        # weight and bias with respect to activation nodes
        act_weight = sum(preact_weight[i]*preact_layer[i].weight for i in 1:length(preact_layer))
        act_bias = preact_bias + sum(preact_weight[i]*preact_layer[i].bias for i in 1:length(preact_weight))
        
        act_layer = prev_layer
                
        preact_weight = [act_weight[i] > 0 ? act_weight[i]*act_layer[i].ub_weight : 
                                             act_weight[i]*act_layer[i].lb_weight for 
                                             i in 1:length(act_layer)]
        preact_bias = act_bias + sum(act_weight[i] > 0 ? act_weight[i]*act_layer[i].ub_bias : 
                                                     act_weight[i]*act_layer[i].lb_bias for 
                                             i in 1:length(act_layer))
        
        preact_layer = [act_layer[i].pre_act for i in 1:length(act_layer)]
        prev_layer = act_layer[1].pre_act.prev_layer
    end
    
    ub_weight = sum(preact_weight[i]*preact_layer[i].weight for i in 1:length(preact_weight))
    ub_bias = preact_bias + sum(preact_weight[i]*preact_layer[i].bias for i in 1:length(preact_weight))
    
    return ub_weight, ub_bias
end

function lowerbound_weight_propagate(
        act_layer::Vector{ActNeuron}, 
        weight::Vector{Float64}, # with respect to activation layer
        bias::Float64,
    )
    # weight and bias with respect to preactivation nodes    
    preact_weight = [weight[i] > 0 ? weight[i]*act_layer[i].lb_weight : weight[i]*act_layer[i].ub_weight for 
                    i in 1:length(act_layer)]
    preact_bias = bias+sum(weight[i] > 0 ? weight[i]*act_layer[i].lb_bias : weight[i]*act_layer[i].ub_bias for 
                          i in 1:length(act_layer))
    
    preact_layer = [act_layer[i].pre_act for i in 1:length(act_layer)]
    prev_layer = act_layer[1].pre_act.prev_layer
    
    while !(prev_layer[1] isa InputNeuron)
        # weight and bias with respect to activation nodes
        act_weight = sum(preact_weight[i]*preact_layer[i].weight for i in 1:length(preact_layer))
        act_bias = preact_bias + sum(preact_weight[i]*preact_layer[i].bias for i in 1:length(preact_weight))
        
        act_layer = prev_layer
                
        preact_weight = [act_weight[i] > 0 ? act_weight[i]*act_layer[i].lb_weight : 
                                             act_weight[i]*act_layer[i].ub_weight for 
                                             i in 1:length(act_layer)]
        preact_bias = act_bias + sum(act_weight[i] > 0 ? act_weight[i]*act_layer[i].lb_bias : 
                                                     act_weight[i]*act_layer[i].ub_bias for 
                                                     i in 1:length(act_layer))
        
        preact_layer = [act_layer[i].pre_act for i in 1:length(act_layer)]
        prev_layer = act_layer[1].pre_act.prev_layer
    end
    
    ub_weight = sum(preact_weight[i]*preact_layer[i].weight for i in 1:length(preact_weight))
    ub_bias = preact_bias + sum(preact_weight[i]*preact_layer[i].bias for i in 1:length(preact_weight))
    
    return ub_weight, ub_bias
end

function compute_upper_bound(
        weight::Vector{Float64},
        bias::Float64,
        layer,
    )
    return sum(weight[i] > 0 ? weight[i]*layer[i].upper_bound : weight[i]*layer[i].lower_bound for
               i in 1:length(layer)) + bias
end

function compute_lower_bound(
        weight::Vector{Float64},
        bias::Float64,
        layer,
    )
    return sum(weight[i] > 0 ? weight[i]*layer[i].lower_bound : weight[i]*layer[i].upper_bound for
               i in 1:length(layer)) + bias
end

function deep_poly(
    net::NeuralNetwork,
    img::Vector{Float64},
    eps::Float64,
    )
    
    input_layer = [InputNeuron(max(-1.0, img[i]-eps), min(1.0, img[i]+eps)) for i in 1:length(img)]
    weights = net.weights
    biases = net.biases
    num_layer = length(weights)

    prev_layer = input_layer
    preact_layer = []
    f = net.activation[1]


        
    for j in 1:length(biases[1])
        bias = biases[1][j]
        weight = weights[1][:, j]
        upper_bound = 
            sum(weight[i] > 0 ? weight[i]*prev_layer[i].upper_bound : weight[i]*prev_layer[i].lower_bound for 
                i in 1:length(prev_layer)) + bias
        lower_bound = 
            sum(weight[i] < 0 ? weight[i]*prev_layer[i].upper_bound : weight[i]*prev_layer[i].lower_bound for 
                i in 1:length(prev_layer)) + bias
        preact_neuron = PreActNeuron(weight, bias, lower_bound, upper_bound, prev_layer)

        push!(preact_layer, preact_neuron)
    end
    prev_layer = [create_act_neuron(preact_neuron, f) for preact_neuron in preact_layer]
    
    preact_layers = [preact_layer]
    print(length(preact_layers))
    current_layer = nothing
    for i in 2:num_layer
        current_layer = []
        f = net.activation[min(i, num_layer-1)]
        for j in 1:length(biases[i])
            bias = biases[i][j]
            weight = weights[i][:, j]
            upper_weight, upper_bias = upperbound_weight_propagate(prev_layer, weight, bias)
            lower_weight, lower_bias = lowerbound_weight_propagate(prev_layer, weight, bias)
            upper_bound = min(compute_upper_bound(upper_weight, upper_bias, input_layer), 
                              compute_upper_bound(weight, bias, prev_layer))
            lower_bound = max(compute_lower_bound(lower_weight, lower_bias, input_layer),
                              compute_lower_bound(weight, bias, prev_layer))
            preact_neuron = PreActNeuron(weight, bias, lower_bound, upper_bound, prev_layer)
            
            push!(current_layer, preact_neuron)
        end
        push!(preact_layers, current_layer)
        prev_layer = [create_act_neuron(preact_neuron, f) for preact_neuron in current_layer]
    end
    return preact_layers
end