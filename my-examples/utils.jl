include("ess.jl")

function computeMixtureComponents(T, model, sampler, variable)

results = sample(model, sampler)
mixtureComponentsRes = results[variable]
mixtureComponents = zeros(T, length(mixtureComponentsRes))

for j in 1:size(mixtureComponents,2)
    mixtureComponents[:,j] = mixtureComponentsRes[j]
end

return mixtureComponents
end

function computeMixtureComponentsESSquartiles(M, T, model, sampler, variable)

mixtureComponentsESS = zeros(M, T)
for l = 1:M
    println(l)
    results = sample(model, sampler)
    mixtureComponentsRes = results[sampler]
    mixtureComponents = zeros(T, length(mixtureComponentsRes))

    for j in 1:size(mixtureComponents,2)
        mixtureComponents[:,j] = mixtureComponentsRes[j]
    end
    for i in 1:size(mixtureComponents,1)
        mixtureComponentsESS[l,i] = ess_factor(mixtureComponents[i,:])
        if isnan(mixtureComponentsESS[l,i])
            mixtureComponentsESS[l,i] = 0.0
        end
    end
end
mixtureComponentsESSquartiles = zeros(3, T)
for i in 1:T
    mixtureComponentsESSquartiles[:, i] = quantile(mixtureComponentsESS[:,i], [.25, .5, .75])
end

return mixtureComponentsESSquartiles
end
