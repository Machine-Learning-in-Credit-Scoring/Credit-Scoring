%% Links apoyo
%https://es.mathworks.com/help/stats/fitglm.html#bt0d8u3-1
%https://es.mathworks.com/help/stats/classificationlinear.predict.html#namevaluepairarguments
%https://es.mathworks.com/help/stats/mnrfit.html#btmaowv-Y
%https://es.mathworks.com/help/stats/fitcknn.html#d120e36198

%% Limpiar
clear
clc

%% Preparación de los datos
%Subo la base de datos
load('BDProyecto.mat')
BD=BDProyecto;

%Variables explicativas
X=BD(:,1:end-1);

%Variable respuesta
Y=BD(:,end)-1;

%Nombres variables
load('nombresvariables.mat')
variables = nombres;

%Separo en train y test
%Para hacerlo de forma aleatoria hago una permutacion aleatoria
n=length(BD);
idx = randperm(n);
X_train = X(idx(1:floor(n*0.8)),:);
y_train = Y(idx(1:floor(n*0.8)));
X_test = X(idx(floor(n*0.8+1):end),:);
y_test = Y(idx(floor(n*0.8+1):end));


%% Métodos para clasificar

%Regresión Logística
Slg = LogitRegression(X_train,X_test,y_train,y_test,variables)

%KNN
k=3;
Sknn = KNN(X_train,X_test,y_train,y_test,k)

%SVM
Ssvm = SVM(X_train,X_test,y_train,y_test)
