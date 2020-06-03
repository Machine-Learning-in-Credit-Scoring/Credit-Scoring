function S = LogitRegression(X_train,X_test,y_train,y_test,nombrevar)
S = struct();

% Fit the model with training data
modelo=fitglm(X_train,y_train,'Distribution','binomial','link','logit');

% Look the p-values to determine the significance
pvalor=table2array(modelo.Coefficients(2:end,4));

% Remove the biggest value until all are less than 0.05
while max(pvalor)>=0.05
    maxp = find(pvalor==max(pvalor));
    pvalor(maxp) = [];
    nombrevar(maxp) = [];
    X_train(:,maxp) = [];
    X_test(:,maxp) = [];
    modelo = fitglm(X_train,y_train,'Distribution','binomial','link','logit');
    pvalor = table2array(modelo.Coefficients(2:end,4));
    intercepto = table2array(modelo.Coefficients(1,4));
end

% Coefficients of the model
S.CoefMod = table2array(modelo.Coefficients(:,1));

% Significant Variables
S.significativas = nombrevar;

% Predict the probabilities 
ypred = predict(modelo,X_test);
S.y_test_pred=[];

% Convert the probabilities into one of the classes
for j=1:length(ypred)
   if ypred(j)>=0.5
       S.y_test_pred(j)=1;
   else S.y_test_pred(j)=0;
   end
end

% Confusion matrix
CM=confusionmat(y_test,S.y_test_pred);

% To know train accuracy
ytpred=predict(modelo,X_train);
S.y_train_pred=[];
for j=1:length(ytpred)
   if ytpred(j)>=0.5     
       S.y_train_pred(j)=1;
   else S.y_train_pred(j)=0;
   end
end

% Confusion matrix
CM2=confusionmat(y_train,S.y_train_pred);

% Test plot
figure()
confusionchart(y_test,S.y_test_pred)
title('Confusion Matrix Test')

% Train plot
figure()
confusionchart(y_train,S.y_train_pred);
title('Confusion Matrix Train')


% Metrics
S.PrecisionC1 = CM(1,1)/sum(CM(:,1)); 
S.RecallC1 = CM(1,1)/sum(CM(1,:));
S.F1scoreC1= 2*(S.PrecisionC1*S.RecallC1)/(S.PrecisionC1+S.RecallC1);

S.PrecisionC2 = CM(2,2)/sum(CM(:,2));
S.RecallC2 = CM(2,2)/sum(CM(2,:));
S.F1scoreC2 = 2*(S.PrecisionC2*S.RecallC2)/(S.PrecisionC2+S.RecallC2);

S.test_acc=trace(CM)/sum(sum(CM));
S.train_acc=trace(CM2)/sum(sum(CM2));
end
