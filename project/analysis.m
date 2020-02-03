
load('chinese_characters.mat')

%% Entropy of X

num_char = 25;
p_x_i = 1/num_char;
h_x = 25 * -1/25 * log2(1/25);
%h_x = 4.4929;
h_x_ = log2(25);

%% Entropy of Z
num_neur = 30;
theta = 1.0;
load('results/finalWeights.mat');
imagesc(W); waitforbuttonpress
z = W' * chinese_character > theta;
imagesc(z); waitforbuttonpress
z_unique = unique(z','rows')';
imagesc(z_unique); waitforbuttonpress
[~,num_unique] = size(z_unique);
z_counts = zeros(1,num_unique);
bar(z_counts); waitforbuttonpress;
for i = 1 : num_char
    [~,index] = ismember(z(:,i)',z_unique','rows');
    z_counts(index) = z_counts(index) + 1;
end
bar(z_counts); title(i); waitforbuttonpress;
p_z = z_counts/num_char;
bar(p_z); title('P(Z)'); waitforbuttonpress;
h_z = sum(-1 * p_z .* log2(p_z));

%% Mutual Information
%x_i = eye(82);
xnz = zeros(num_char, num_unique);
for i = 1 : num_char
    [~,index] = ismember(z(:,i)',z_unique','rows');
    xnz(i,index) = 1;
end
p_xnz = xnz / num_char;
p_x_mat = p_x_i * ones(size(p_xnz));
p_z_mat = repmat(p_z, [num_char 1]);
l = log2(p_xnz ./ p_x_mat ./ p_z_mat);
l(isinf(l)) = 0;
mi = sum(sum(p_xnz .* l));

%% Mutual Information (no noise)
mi_ = h_z - 0; % H(Z|X)

%% Statistical Dependence of X
p_x_i = mean(chinese_character,2);
l = log2(p_x_i);  
l(isinf(l)) = 0;
h_x_i = -1 * p_x_i .* l;
sd_x = sum(h_x_i) - h_x;

%% Statistical Dependenec of Z
p_z_i = mean(z, 2);
l = log2(p_z_i);
l(isinf(l)) = 0;
h_z_i = -1 * p_z_i .* l;
sd_z = sum(h_z_i) - h_z;
