function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



K=max(y);
Y=zeros(m,K);
for i=1:m
  Y(i,y(i))=1;
endfor

% Add ones to the X data matrix
X = [ones(size(X,1), 1) X];

a2=sigmoid(X*Theta1');

% Add ones to the a2 data matrix
a2 =[ones(size(a2, 1),1) a2];

a3=sigmoid(a2*Theta2');

% delete bias term theta0
th1=Theta1(:,2:end);
th2=Theta2(:,2:end);
% square theta0
th1=th1.^2;
th2=th2.^2;

J = sum(sum(-Y.*log(a3)-(1-Y).*log(1-a3)))./m+lambda/(2*m)*(sum(sum(th1))+sum(sum(th2)));


% part 2


for i=1:m
  % (1)
  a_1=X(i,:);       % 1x401
  z_2=a_1*Theta1';  % 1x25
  a_2=sigmoid(z_2); % 1x25
  a_2=[1,a_2];      % 1x26
  z_3=a_2*Theta2';  % 1x10
  a_3=sigmoid(z_3); % 1x10
  %(2)
  d_3=a_3-Y(i,:);
  %(3)
  d_2=d_3*Theta2(:,2:end).*sigmoidGradient(z_2);
  %(4)
  delta_1 = d_2.*a_1';
  delta_2 = d_3.*a_2';
  Theta1_grad= Theta1_grad+delta_1';
  Theta2_grad= Theta2_grad+delta_2';
  
endfor


Theta1_grad=Theta1_grad./m;
Theta2_grad=Theta2_grad./m;

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda./m*(Theta1(:,2:end));
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda./m*(Theta2(:,2:end));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
