train_data_1 = csvread('train_data_1.txt');
train_data_2 = csvread('train_data_2.txt');
train_data_3 = csvread('train_data_3.txt');
test_data_4 = csvread('test_data_4.txt');

train_data = [train_data_1;train_data_2;train_data_3];

Alpha = .00001;

[weight,error1] = deltaTraining(train_data,Alpha);
[error2,errors2] = testing(test_data_4,weight);

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

disp("Learning Rate:")
disp(Alpha)
disp("Iterations:")
ite = 1000000;
disp(ite)

labels = ["Hour","Pred","Real","Error"];

disp("Day 1: Training Error")
disp(labels)
[errorD1,errorsD1] = testing(train_data_1,weight);
for i = 1:9
    temp = -1*(weight(1,3)+weight(1,1)*(i+4))/weight(1,2);
    output = [i+4,temp,train_data_1(i,2),errorsD1(i,1)];
    disp(output)
end
disp("Total Error:")
disp(errorD1(1,1))

disp("Day 2: Training Error")
disp(labels)
[errorD2,errorsD2] = testing(train_data_2,weight);
for i = 1:9
    temp = -1*(weight(1,3)+weight(1,1)*(i+4))/weight(1,2);
    output = [i+4,temp,train_data_2(i,2),errorsD2(i,1)];
    disp(output)
end
disp("Total Error:")
disp(errorD2(1,1))
   
disp("Day 3: Training Error")
disp(labels)
[errorD3,errorsD3] = testing(train_data_3,weight);
for i = 1:9
    temp = -1*(weight(1,3)+weight(1,1)*(i+4))/weight(1,2);
    output = [i+4,temp,train_data_3(i,2),errorsD3(i,1)];
    disp(output)
end
disp("Total Error:")
disp(errorD3(1,1))

disp("Day 4: Testing Error")
disp(labels)
for i = 1:9
    temp = -1*(weight(1,3)+weight(1,1)*(i+4))/weight(1,2);
    output = [i+4,temp,test_data_4(i,2),errors2(i,1)];
    disp(output)
end
disp("Total Error:")
disp(error2(1,1))

function [error,errors] = testing(testingData,perceptron)
    np = size(testingData,1);
    weights = perceptron;
    desired = zeros(np,1);
    for i = 1:np
        
        desired(i,2) = testingData(i,2);
        
    end
    outputTotal = testingData(:,1)*weights(1,1)+testingData(:,2)*weights(1,2)+weights(1,3);
    error = sum((desired(:,1)-outputTotal(:,1)).^2);
    errors = desired(:,1)-outputTotal(:,1);
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