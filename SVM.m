function S = SVM(X_train,X_test,y_train,y_test)
S = struct();

% Fit the model with the training data
S.modelo = fitcsvm(X_train,y_train,'KernelFunction','linear');

% train prediction
ytrain_pred = predict(S.modelo,X_train);

% test prediction
S.ytest_pred = predict(S.modelo,X_test);

% Test plot
figure()
confusionchart(y_test,S.ytest_pred)
title('Confusion Matrix Test')

% Train plot
figure()
confusionchart(y_train,ytrain_pred);
title('Confusion Matrix Train')

%Metrics
CM=confusionmat(y_test,S.ytest_pred);
CM2=confusionmat(y_train,ytrain_pred);

S.PrecisionC1 = CM(1,1)/sum(CM(:,1)); 
S.RecallC1 = CM(1,1)/sum(CM(1,:));
S.F1scoreC1= 2*(S.PrecisionC1*S.RecallC1)/(S.PrecisionC1+S.RecallC1);

S.PrecisionC2 = CM(2,2)/sum(CM(:,2));
S.RecallC2 = CM(2,2)/sum(CM(2,:));
S.F1scoreC2 = 2*(S.PrecisionC2*S.RecallC2)/(S.PrecisionC2+S.RecallC2);

S.test_acc=trace(CM)/sum(sum(CM));
S.train_acc=trace(CM2)/sum(sum(CM2));

end

