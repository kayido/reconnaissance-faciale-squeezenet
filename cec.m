dataset = imageDatastore('dataset','IncludeSubfolders',true,'LabelSource','foldernames');

[training_data, validation_date] = splitEachLabel(dataset,0.7,'randomized');
net =  squeezenet;

analyzeNetwork(net);

input_layer_size  = net.Layers(1).InputSize(1:2);
resized_training_data = augmentedImageDatastore(input_layer_size,validation_date);
resized_validation_date = augmentedImageDatastore(input_layer_size, validation_date);

network_architecture = layerGraph(net);

number_of_classes = numel(categories(training_data.Labels));
%nav_convolutional_layer = convolution2dLayer([1,1], Number_of_Classes,"NeighLearnkateFactor",10, "BiesLearnRateFactor",10, 'Name', 'Facial Feature Learner');
nav_convolutional_layer = convolution2dLayer([1,1], ...
    number_of_classes,"WeightLearnRateFactor",10, ...
    "BiasLearnRateFactor",10, ...
    "Name","Facial Feature Learner");

nav_classification_layer = classificationLayer('Name', 'Face Classifier');

new_network = replaceLayer(network_architecture, 'conv10', nav_convolutional_layer);

new_network = replaceLayer(new_network, 'ClassificationLayer_predictions', nav_classification_layer);

training_options = trainingOptions('sgdm', ...
    'MiniBatchSize',4, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',4e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',resized_validation_date, ...
    'Verbose',false, ...
    'Plots','training-progress' ...
    );


trained_network = trainNetwork(resized_training_data,new_network,training_options);

save("FaceReconized.mat","trained_network");