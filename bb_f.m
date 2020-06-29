function [fs] = bb_f(A, I, I0, K, A_init, C_est, mu_x, sig_x, alpha0, beta0)
                    

for prior = 1:length(beta0)
    
    % Call Gibbs
    tic
    [A_s] = gibbs_beta_bernoulli(I, I0, K, A_init, C_est, mu_x, sig_x, alpha0, beta0(prior));
    toc 
    
    % Estimate as the mode
    A_est = mode(A_s, 3);

    % Performance
    [~, ~, fs(prior)] = adj_eval(A, A_est);    

end


end