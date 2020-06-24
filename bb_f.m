function [fs_3] = bb_f(A, I, I0, K, A_init, C, mu_x, sig_x, alpha0, beta0)

parfor prior = 1:length(beta0)
    
    % Calls Gibbs 
    tic
    [A_s] = gibbs_beta_bernoulli(I, I0, K, A_init, C, mu_x, sig_x, alpha0, beta0(prior));
    toc

    % Compute fscore
    [~,~, fs_3(prior)] = adj_eval(A, mode(A_s, 3));

end


end