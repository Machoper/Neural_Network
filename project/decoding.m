function decoding()
%% FUNCTION Runs simulation of neurons using
% adaptive synaptogenesis by Adelsberger-Magnan and Levy
% Chris Kaylin & Blake Thomas, 2014
% modified by Harang Ju, 2017

tic

% --------------------
%%   Random Seeds
% --------------------

seed = 1;
rng(seed);

% --------------------
%%     Functions
% --------------------

% input function
% returns a block of input patterns
% column: an input pattern
% row: an input line
characterLoad = load('chinese_characters.mat');
weightLoad = load('results/finalWeights.mat');
%asciiLoad = load('ascii.mat');
%asciiInputs = repmat(asciiLoad.ascii(:,[9 11]), 1, 100);
%characterInputs = repmat(characterLoad.new(:,[1 6]), 1, 100);
% 9 is +
% 11 is 0
function inputset = getInput
    inputset = characterLoad.chinese_character' * weightLoad.W;
    %inputset = characterInputs;
    %inputset = asciiInputs;
end

% returns a batch of input blocks
BatchSize = 10;
function batchInput = getBatchInput(batchSize)
    [featureCount, inputPatternCount] = size(getInput);
    batchInput = zeros(featureCount, inputPatternCount * batchSize);
    for b = 1 : batchSize
        batchInput(:, (b-1)*inputPatternCount + 1 : ...
            b*inputPatternCount) = getInput;
    end
end

% --------------------
%%     Constants
% --------------------

disp('Setting parameters.');

Directory = 'finalResults'; % to which to save results
NeuronCount = 30; % Number of output neurons
TotalPresentations = 20000; % max number of presentations of input blocks
FireThreshold = 1.0;
ReceptivityThreshold = 0.1; % receptive to synaptogenesis
Epsilon = 1e-2; % learning constant of the synaptic modification rule
Gamma = 1e-2; % constant of the probability of synaptogenesis
InitialWeightValue = 0.1; % value of the weight of new connections
WeightThreshold = 0.05; % connection gets deleted if the weight falls below 0.05
Alpha = 0.25;
StabilityThreshold = 200; % network stable after 
    % [threshold] cycles w/o changes in synapse count
RandomInitializationOn = false; % random initialization of weights
SheddingOn = true;
SynaptogenesisOn = true;
InitialSynapseCountPerNeuron = 1;
DisplayCycle = 50;
[FeatureCount, InputPatternCount] = size(getInput);

% --------------------
%%   Initialization
% --------------------

disp('Initializing variables.');

connectivity = zeros(FeatureCount, NeuronCount);
recentActivity = ReceptivityThreshold / 2 * ones(1, NeuronCount);
stabilityCount = 0;
Ex = repmat(mean(getBatchInput(BatchSize), 2), 1, InputPatternCount);

% synaptogenesis & shedding data
synapseCount = zeros(NeuronCount, TotalPresentations);
synapseCountPresyn = zeros(FeatureCount, TotalPresentations);
synaptogCount = zeros(NeuronCount, TotalPresentations);
sheddingCount = zeros(NeuronCount, TotalPresentations);
excitation = zeros(NeuronCount, TotalPresentations);
meanFirings = zeros(NeuronCount, TotalPresentations);
activity = zeros(NeuronCount, TotalPresentations);

% initialize W with some random connections
for n = 1 : NeuronCount
    randpermFeatures = randperm(FeatureCount);
    connectivity(randpermFeatures(1:InitialSynapseCountPerNeuron), n) = 1;
end
W = InitialWeightValue .* connectivity(:, :);

% Random weight initialization
if RandomInitializationOn
    W = W .* (rand(size(W)) + WeightThreshold);
end

% Initialize time
t = 1;

% track synapse count
synapseCount(:, t) = sum(connectivity);
synapseCountPresyn(:, t) = sum(connectivity, 2);

% --------------------
%%  Begin simulation
% --------------------

disp('Started simulation.')

for t = 2 : TotalPresentations

    % Update Stability Count
    stabilityCount = stabilityCount + 1;

    % Get Inputs
    inputset = getInput;
    % calculate x_i - E[X] as "zeroCenteredInput"
    %zeroCenteredInput = inputset - Ex;

    % Associative Synaptic Modification
    for i = randperm(InputPatternCount)
        %Wpos = W .* (W >= WeightThreshold);
        % postsynaptic excitation
        % get z
        z = inputset(:, i)' * W > FireThreshold;
        % get (x_i - w_i)
        presynTerm = repmat(inputset(:, i), 1, NeuronCount) - W;
        % get delta w = e * z * (x_i - w_i)
        deltaW = (Epsilon * repmat(z, [FeatureCount 1]) .* presynTerm);
        W = W + deltaW .* connectivity;
    end

    % Track Data
    %z = inputset' * W > FireThreshold;
    recentActivity = (1 - Alpha) .* recentActivity + Alpha * mean(z);

    % Synaptic Shedding
    connectionsToShed = (W < WeightThreshold) & connectivity;
    connectivity = connectivity - (SheddingOn .* connectionsToShed);
    W = W .* connectivity;
    stabilityCount = stabilityCount * ~nnz(connectionsToShed);

    % Check for Stability
    if (stabilityCount > StabilityThreshold)
        break
    end

    % Synaptogenesis
    receptivity = recentActivity < ReceptivityThreshold;
    probSynaptog = repmat(Gamma * receptivity, FeatureCount, 1);
    newConnections = probSynaptog > rand(size(connectivity));
    newConnections = SynaptogenesisOn .* newConnections & ~connectivity;
    connectivity = connectivity + newConnections;
    W = W + InitialWeightValue * newConnections;
    stabilityCount = stabilityCount * ~nnz(newConnections);
    
    % Track Data
    sheddingCount(:, t) = sum(connectionsToShed);
    synaptogCount(:, t) = sum(newConnections);
    synapseCount(:, t) = sum(connectivity);
    synapseCountPresyn(:, t) = sum(connectivity, 2);
    activity(:, t) = recentActivity;
    meanFirings(:, t) = mean(z);
    %excitation(:, t) = y;
    
    % Display Data
    if mod(t, DisplayCycle) == 0
        imagesc(W); title(['cycle ', num2str(t)]);
        xlabel('neurons'); ylabel('input lines');
        colorbar; caxis([-0.1 0.1])
        pause(0.001);
    end
end

disp('Done.');

toc

%% Truncate & Save Data
disp('Saving data.');

if exist(Directory, 'dir') == 0
    mkdir(Directory);
elseif exist(Directory, 'dir') ~= 7
    disp([Directory, ' is not a directory. Change "Directory".']);
    return
end

synapseCount(:, t:end) = [];
synapseCountPresyn(:, t:end) = [];
synaptogCount(:, t:end) = [];
sheddingCount(:, t:end) = [];
activity(:, t:end) = [];
meanFirings(:, t:end) = [];
excitation(:, t:end) = [];

% save data
save([Directory, '/', 'finalWeights.mat'], 'W');
save([Directory, '/', 'synapseCount.mat'], 'synapseCount');
save([Directory, '/', 'synapseCountPresyn.mat'], 'synapseCountPresyn');
save([Directory, '/', 'synaptogCount.mat'], 'synaptogCount');
save([Directory, '/', 'sheddingCount.mat'], 'sheddingCount');
save([Directory, '/', 'activity.mat'], 'activity');
save([Directory, '/', 'meanFirings.mat'], 'meanFirings');
save([Directory, '/', 'excitation.mat'], 'excitation');

disp(['Saved results to folder named "' Directory '"']);

end
