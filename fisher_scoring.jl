using LinearAlgebra
using Statistics

# Create structure
struct Distribution
    score_function::Function
    hessian_function::Function
end    

# Binomial with logit link (canonical)
function binomial_score(y, x, beta)
    x' * (y - exp.(x * beta)./(1 .+ exp.(x * beta)))
end
function binomial_hessian(x, beta)
    x' * Diagonal(exp.(x * beta)./(1 .+ exp.(x * beta)).^2) * x
end
binomial = Distribution(binomial_score,
                        binomial_hessian)

# Exponential with log link (canonical)
function exponential_score(y, x, beta)
    x' * (y .* exp.(-x * beta) .- 1)
end
function exponential_hessian(x, beta)
    x' * Diagonal(ones(size(x)[1])) * x
end
exponential = Distribution(exponential_score,
                           exponential_hessian)

# Function to return score at beta
score(y, x, dist::Distribution, beta) = dist.score_function(y, x, beta)

# Functions to return hessian at beta
hessian(x, dist::Distribution, beta) = dist.hessian_function(x, beta)

# Implement Fisher scoring to find MLE of beta
function Fisher_scoring(y, x, dist::Distribution; beta=[0, 0], tol=0.01)
    """
    Some nice documentation here.
    """
    S = score(y, x, dist, beta)
    B = inv(hessian(x, dist, beta))
    #Newton's method to find optimal beta
    new_beta = beta + B * S
    if sqrt(sum((new_beta-beta).^2)) < tol 
        return round.(new_beta, digits = 3), round.(sqrt.(diag(B)), digits = 3)
    else
        return Fisher_scoring(y, x, dist, beta = new_beta, tol = tol) 
    end
end

# Fit an exponential glm with canonical log link
# Data
x = [6.1, 4.2, 0.5, 8.8, 1.5, 9.2, 8.5, 8.7, 6.7, 6.5, 6.3, 6.7, 0.2, 8.7, 7.5]
y = [0.8, 3.5, 12.4, 1.1, 8.9, 2.4, 0.1, 0.4, 3.5, 8.3, 2.6, 1.5, 16.6, 0.1, 1.3]

# Find MLEs and SE for beta0 and beta1
beta_hat, beta_se = Fisher_scoring(y, hcat(ones(length(x)), x), exponential)
beta_ci = round.([beta_hat - 1.96*beta_se beta_hat + 1.96*beta_se], digits = 3)
beta0_ci, beta1_ci  = beta_ci[1,:], beta_ci[2,:]

# Fit a binomial glm with canonical logistic link
# Data
x = [6.1, 4.2, 0.5, 8.8, 1.5, 9.2, 8.5, 8.7, 6.7, 6.5, 6.3, 6.7, 0.2, 8.7, 7.5]
y_bin = y .> median(y)

# Find MLEs and SE for beta0 and beta1
beta_hat, beta_se = Fisher_scoring(y_bin, hcat(ones(length(x)), x), binomial)
beta_ci = round.([beta_hat - 1.96*beta_se beta_hat + 1.96*beta_se], digits = 3)
beta0_ci, beta1_ci  = beta_ci[1,:], beta_ci[2,:]