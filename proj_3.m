train_data_1 = csvread('train_data_1.txt');
train_data_2 = csvread('train_data_2.txt');
train_data_3 = csvread('train_data_3.txt');
test_data_4 = csvread('test_data_4.txt');

train_data = [train_data_1;train_data_2;train_data_3];

Alpha = .00001;

[weight,error1] = deltaTraining(train_data,Alpha);
error2 = testing(test_data_4,weight);

figure
plot (test_data_4(1:9,1),test_data_4(1:9,2),'+r')
hold on;
x = 5 : 1 : 13 ;
y = -1*(weight(1,3)+weight(1,1)*x)/weight(1,2);
plot(x,y)
title1 = strcat('Delta Training: Error=',num2str(error2(1,1)));
title(title1)
xlabel('time (hr)')
ylabel('temp (F)')

function error = testing(testingData,perceptron)
    np = size(testingData,1);
    weights = perceptron;
    desired = zeros(np,1);
    for i = 1:np
        
        desired(i,2) = testingData(i,2);
        
    end
    outputTotal = testingData(:,1)*weights(1,1)+testingData(:,2)*weights(1,2)+weights(1,3);

    error = sum((desired(:,1)-outputTotal(:,1)).^2);
end

function [weights,error] = deltaTraining(trainingData,learningConst)
%This function trains a neuron using the delta rule
    np = size(trainingData,1);
    ite = 1000000;
    alpha = learningConst;
    epsilon = .0001;
    i = .100;
    error = 1; %default value
    weights = sum(rand(3)); %randomized starting weights
    while (i <= ite) && (error > epsilon)
        desired = zeros(np,1);
        for j = 1:np
            
            X = trainingData(j,1)*weights(1,1)+trainingData(j,2)*weights(1,2)+weights(1,3);
            
            output = 10 * X;
            
            desired(j,2) = trainingData(j,2);

            %wi = wi + ?(yi ? oi)x
            weights = weights + alpha * (desired(j,1) - output);
            
        end
        outputTotal = trainingData(:,1)*weights(1,1)+trainingData(:,2)*weights(1,2)+weights(1,3);
        error = sum((desired(:,1)-outputTotal(:,1)).^2);
        i = i+1;
    end
end