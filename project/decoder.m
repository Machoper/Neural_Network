function res = decoder(W, input)
theta = 1.0;
epsilon = 1e-2;
x = input' * W;

for i = randperm(10)
        %Wpos = W .* (W >= WeightThreshold);
        % postsynaptic excitation
        z = x(:, i)' * W > 1.0;
        % get (x_i - w_i)
        presynTerm = repmat(x(:, i), 1, 10) - W;
        % get delta w = e * z * (x_i - w_i)
        deltaW = (Epsilon * repmat(z, [25 1]) .* presynTerm)/sum(W);
        W = W + deltaW;
    end
end
