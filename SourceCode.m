function [s_star, f_best, history] = resilin_flea_feature_selection(N, d, T, tau, eps_val, rho, tol_no_imp, target_acc)
% Inputs:
%   N          - Number of fleas (population size), e.g., 30
%   d          - Number of features (dimensions), e.g., 50
%   T          - Maximum number of iterations, e.g., 100
%   tau        - Binarization threshold in (0, 1), e.g., 0.5
%   eps_val    - Elite fraction epsilon in (0, 1], e.g., 0.2
%   rho        - Pheromone evaporation/learning rate in [0, 1], e.g., 0.3
%   tol_no_imp - Max iterations without improvement before early stop, e.g., 20
%   target_acc - Target fitness/accuracy threshold, e.g., 0.99
%
% Outputs:
%   s_star     - Best binary feature mask (1 x d logical)
%   f_best     - Best fitness score achieved
%   history    - Struct storing best fitness per iteration

    % Note these values are for test purpose and may chnage, read the paper
    % for accurate values. moreover other settings may differ as the code
    % is taken from experimentals runs.
    if nargin < 1 || isempty(N),          N = 30;      end
    if nargin < 2 || isempty(d),          d = 50;      end
    if nargin < 3 || isempty(T),          T = 100;     end
    if nargin < 4 || isempty(tau),        tau = 0.5;   end
    if nargin < 5 || isempty(eps_val),    eps_val = 0.2; end
    if nargin < 6 || isempty(rho),        rho = 0.3;   end
    if nargin < 7 || isempty(tol_no_imp), tol_no_imp = 25; end
    if nargin < 8 || isempty(target_acc), target_acc = 1.0; end

    % 1. Initialization
    X = rand(N, d);                 % x_i ~ U(0,1)^d for i = 1..N
    S = X > tau;                    % s_i = [x_i > tau] (N x d logical)
    p = mean(double(S), 1);         % p = mean of s_i (1 x d)
    f_best = -inf;                  % Use -inf so any valid fitness beats it
    s_star = false(1, d);           % Best binary vector s*

    no_imp_count = 0;
    history.f_best = nan(T, 1);

    % 2. Main iteration loop: t = 0 to T - 1
    for t = 0:(T - 1)
        % 3. Decay factor kappa
        kappa = 1 - (t / T);

        % Preallocate candidate positions, binary masks, and fitnesses
        X_tilde = zeros(N, d);
        S_tilde = false(N, d);
        f = zeros(N, 1);

        % 4. Phase 1: Resilin-controlled random jump for each flea i
        for i = 1:N
            % 5. Draw Resilin r_i ~ U(0,1) and landing point u_i ~ U(0,1)^d
            r_i = rand();
            u_i = rand(1, d);

            % 6. Eq. (5) - clamp to [0, 1] to keep positions bounded
            x_tilde_i = X(i, :) + kappa * r_i * (u_i - X(i, :));
            x_tilde_i = min(max(x_tilde_i, 0), 1);
            X_tilde(i, :) = x_tilde_i;

            % 7. Eq. (4) - binarize and evaluate fitness
            s_tilde_i = x_tilde_i > tau;
            S_tilde(i, :) = s_tilde_i;
            f_i = evaluateFitness(s_tilde_i);
            f(i) = f_i;

            % 8. Elite memory update
            if f_i > f_best
                f_best = f_i;
                s_star = s_tilde_i;
                no_imp_count = 0;
            end
        end

        if f_i <= f_best
            % Track consecutive iterations with no global improvement
            if t == 0 || f_best <= history.f_best(max(t, 1))
                no_imp_count = no_imp_count + 1;
            end
        end
        history.f_best(t + 1) = f_best;

        % 10. Select top ceil(eps * N) fleas
        num_elite = max(1, min(N, ceil(eps_val * N)));
        [~, sort_idx] = sort(f, 'descend');
        E = sort_idx(1:num_elite);

        % Compute normalized elite weights w_j (sum(w_j) = 1)
        f_elite = f(E);
        f_shifted = f_elite - min(f_elite) + eps; % Avoid division by zero
        w = f_shifted / sum(f_shifted);

        % 11. Phase 2: Pheromone update, Eq. (6)
        elite_weighted_sum = w' * double(S_tilde(E, :)); % 1 x d
        p = (1 - rho) * p + rho * elite_weighted_sum;
        p = min(max(p, 0), 1);

        % 12. Position update toward pheromone vector p, Eq. (7)
        for i = 1:N
            beta_i = rand();
            X(i, :) = (1 - beta_i) * X_tilde(i, :) + beta_i * p;
        end

        % 13. Termination check
        iter_completed = (t + 1) >= T;
        if terminationCriteriaMet(f_best, target_acc, iter_completed, no_imp_count, tol_no_imp)
            history.f_best = history.f_best(1:(t + 1));
            break;
        end
    end
    % 15. Return s* (pass s_star indices to your LASSO post-processing step)
end

%% --- Helper Functions ---

function fit = evaluateFitness(s)
% Replace this placeholder with your actual classifier + feature subset fitness,
% e.g., cross-validation accuracy minus a feature-count penalty.
    if ~any(s)
        fit = 0; % Penalize empty feature subset
        return;
    end
    num_selected = sum(s);
    d = numel(s);
    % Example synthetic objective: reward selected features, penalize subset size
    fit = 0.85 * (sum(s(1:min(10, d))) / min(10, d)) - 0.10 * (num_selected / d);
end

function stop_flag = terminationCriteriaMet(f_best, target_acc, iter_completed, no_imp_count, tol_no_imp)
% Implements Lines 17-25: TerminationCriteriaMet()
    acc_attained = (f_best >= target_acc);
    no_improvement = (no_imp_count >= tol_no_imp);

    stop_flag = acc_attained || iter_completed || no_improvement;
end