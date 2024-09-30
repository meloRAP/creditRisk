rm(list = ls()) #limpar memoria
library(tidyverse)
library(e1071)
library(dplyr)
library(caTools) #semente treino e teste
library(caret)  #matriz confusao
library(pROC)   #AUC
library(lmtest) #qui² 
library(plotly)  #plotar
#install.packages("MLmetrics")
library(MLmetrics)   #calcular f1 score
library(ROCR)
library(yardstick)
library(caret)
library(ipred)
library(grid)
library(randomForest)
library(rpart.plot)

#semente
set.seed(12)

dados = read_csv("credit_risk_dataset.csv")
names(dados)
glimpse(dados)
table(dados$cb_person_default_on_file)
colSums(is.na(dados))

#tratamento dos dados

#se idade > 100 ou idade < 18, colocamos NA

dados$person_age = ifelse((dados$person_age > 100)|(dados$person_age < 18),
                          NA,
                          dados$person_age
                          )
summary(dados$person_age)


#onde tem NA, substituir pela media da idade

dados$person_age = ifelse(is.na(dados$person_age), 
                          mean(dados$person_age,
                          na.rm = TRUE),
                          dados$person_age
                          ) 

summary(dados$person_age)

#y = 1 e N = 0

dados$cb_person_default_on_file = 
  as.factor(ifelse(dados$cb_person_default_on_file == "Y", 1, 0))

dados$cb_person_default_on_file %>% table

#onde tempo de emprego em anos >100, colocar NA

dados$person_emp_length %>% table

dados$person_emp_length = ifelse((dados$person_emp_length > 100), 
                                 NA, dados$person_emp_length) 

#onde tem NA, substituir pela média de tempo
dados$person_emp_length = ifelse(is.na(dados$person_emp_length), 
                                 mean(dados$person_emp_length, na.rm = TRUE), dados$person_emp_length) 

#onde tem NA em loan_int_rate, substituir pela media
dados$loan_int_rate = ifelse(is.na(dados$loan_int_rate), mean(dados$loan_int_rate, na.rm = TRUE), dados$loan_int_rate) 
glimpse(dados)


table(dados$cb_person_default_on_file)
colSums(is.na(dados))


# padronizando pela z-score: media 0 desvio padrao 1


dados$idade = scale(dados$person_age)
dados$renda = scale(dados$person_income)
dados$tempo_emprego = scale(dados$person_emp_length)
dados$vl_financiado = scale(dados$loan_amnt)
dados$taxa_juros = scale(dados$loan_int_rate)
dados$endiv = scale(dados$loan_percent_income)
dados$temp_relac = scale(dados$cb_person_cred_hist_length)  #em anos

# selecionando as variáveis mais relevantes

dados2 = dados %>% select(cb_person_default_on_file, idade,
                          renda, tempo_emprego, vl_financiado,
                          taxa_juros, endiv, temp_relac)

glimpse(dados2)
dados3 <- as.data.frame(lapply(dados2, as.vector))
dados3$cb_person_default_on_file = as.factor(dados3$cb_person_default_on_file)
glimpse(dados3)

# Construindo Treino e Teste

train_index <- createDataPartition(y=dados3$cb_person_default_on_file, 
                                   p = 0.75, 
                                   list=FALSE
                                   )

# Conjunto de treino e teste

treino <- dados3[train_index, ]
teste <- dados3[-train_index, ]

nrow(treino)
ncol(treino)

nrow(teste)
ncol(teste)

str(treino)
str(teste)

treino %>% select(cb_person_default_on_file) %>% table
teste %>% select(cb_person_default_on_file) %>% table

all(names(treino) %in% names(teste))

#---------------------------------------------------------#
#                                                         #
#                   REGRESSÃO LOGISTICA                   #
#                                                         #
#---------------------------------------------------------#

# Estimando o modelo de regressão logística binária
modelo <- glm(formula = cb_person_default_on_file ~ ., 
              data = treino, 
              family = "binomial")

# Obtendo os resultados do modelo
summary(modelo)
logLik(modelo)
lrtest(modelo)
confint.default(modelo, level = 0.95)

valores_preditos <- predict(modelo, 
                            newdata = teste,
                            type = "response")
length(valores_preditos)
round(valores_preditos, 4)
cutofff <- 0.23

# Classificando os valores preditos com base no cutoff
preditos_class = factor(ifelse(valores_preditos > cutofff, 1, 0))

preditos_class %>% table
teste$cb_person_default_on_file %>% table

# Matriz de confusão para o cutoff estabelecido
confusionMatrix(data = preditos_class,
                reference = as.factor(teste$cb_person_default_on_file), 
                positive = "1")

ac1 <- 0.8192

# Gerando a curva ROC para o modelo final
ROC <- roc(response = teste$cb_person_default_on_file, 
           predictor = valores_preditos)

ggplotly(
  ggroc(ROC, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(ROC$auc, 4))) +
    theme_bw())
roc1 <- 0.8624



stepwise <- step(object = modelo,
                 k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))

# Obtendo os resultados do modelo pós procedimento de remoção
summary(stepwise)    #cb person~tempo emprego + vl financiado + taxa juros

# Valor da Log-Likelihood do modelo final
logLik(stepwise)

# Significância estatística geral do modelo (teste qui²)
lrtest(stepwise)

confint.default(stepwise, level = 0.95)

## Analisando a qualidade do ajuste do modelo final

# Identificando os valores preditos pelo modelo na base de dados atual
valores_preditos <- predict(object = stepwise, 
                            newdata = teste,
                            type = "response")

# Podemos estabelecer um cutoff para a classificação entre evento / não evento
cutoff <- 0.23

# Classificando os valores preditos com base no cutoff
preditos_class = factor(ifelse(valores_preditos > cutoff, 1, 0))

# Matriz de confusão para o cutoff estabelecido
confusionMatrix(data = preditos_class,
                reference = as.factor(teste$cb_person_default_on_file), 
                positive = "1")
ac2 <- 0.8201
# Gerando a curva ROC para o modelo final
ROC <- roc(response = teste$cb_person_default_on_file, 
           predictor = valores_preditos)

ggplotly(
  ggroc(ROC, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(ROC$auc, 4))) +
    theme_bw())

roc2 <- 0.8623

# Analisando as odds ratio do modelo final
odds_ratio <- data.frame(odds_ratio = round(exp(coef(stepwise)[-1]),4))

## Em quanto se altera a chance de ter credito:

## taxa de valor financiado mais alta, ceteris paribus 
## Resposta: é multiplicada por um fator de 0.9095, ou seja, chance 9,05% menor

## taxa de juros maior, ceteris paribus
## Resposta: é multiplicada por um fator de 5.2958, ou seja 529,58% maior
# Fim!

#---------------------------------------------------------#
#                                                         #
#                        BAGGING                          #
#                                                         #
#---------------------------------------------------------#


fit1 <- bagging(cb_person_default_on_file ~ ., data = treino, 
                coob = TRUE, nbagg = 100)
summary(fit1)

# nbagg especifica o número de replicações de bootstrap

pred1 <- predict(fit1, teste[-1])
prob_pred1 <- predict(fit1, newdata = teste[-1], type = "prob")[, "1"]
result1 <- data.frame(original = teste$cb_person_default_on_file, predicted = pred1)
result1


str(result1$predicted)
str(result1$original)

#result1$original <- factor(result1$original, levels = levels(result1$predicted))

conf_matrix1 <- confusionMatrix(data = result1$predicted, 
                                reference = result1$original)

conf_matrix1 # matriz de confusão



# Extrair a matriz de confusão e transformá-la em data frame
cm <- as.data.frame(conf_matrix1$table)

# Adicionar uma coluna que identifica se a predição foi correta ou não
cm$Correct <- ifelse(cm$Reference == cm$Prediction, TRUE, FALSE)



# Plotar a matriz de confusão
p <- ggplot(data = cm, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Correct), color = "white") +
  scale_fill_manual(values = c("TRUE" = "#a6bddb", "FALSE" = "#2b8cbe")) +  # Tons de azul para acertos e erros
  geom_text(aes(label = Freq), color = "black", size = 6) +  # Rótulos em preto para contraste
  labs(title = "Matriz de Confusão",
       x = "Valor Verdadeiro",
       y = "Valor Predito",
       fill = "Correção") +
  theme_minimal() +
  theme(legend.position = "none")  # Remover a legenda para a coluna 'Correct'

# Criar a legenda
legend_text <- "0: Não inadimplente\n1: Inadimplente"

# Ajustar a largura da imagem
ggsave("matriz_confusao.png", p, width = 8, height = 6, units = "in", dpi = 300)

# Exibir a imagem e a legenda ao lado
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 2, widths = unit(c(4, 1), "null"))))
print(p, vp = viewport(layout.pos.col = 1, layout.pos.row = 1))
grid.text(legend_text, vp = viewport(layout.pos.col = 2, layout.pos.row = 1),
          just = "center", gp = gpar(fontsize = 9))

ac3<- 0.8289  # valor da acurácia


# Curva Roc
# bagging


# Gerando a curva ROC para o modelo final
roc3 <- roc(response = teste$cb_person_default_on_file, 
            predictor = factor(prob_pred1, 
                               ordered = TRUE))

ggplotly(
  ggroc(roc3, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(roc3$auc, 4))) +
    theme_bw())

auc3 = 0.8868

#---------------------------------------------------------#
#                                                         #
#                    RANDOM FOREST                        #
#                                                         #
#---------------------------------------------------------#

# Ajuste o modelo randomForest

fit2 <- randomForest(cb_person_default_on_file ~ ., 
                     data = treino, importance = TRUE)
fit2           
plot(fit2)

# importância de variáveis

importance(fit2)
varImpPlot(fit2)

pred2<- predict(fit2, teste[,-1])
prob_pred2 <- predict(fit2, newdata = teste[-1], type = "prob")[, "1"]

conf_matrix2 <- confusionMatrix(teste$cb_person_default_on_file, pred2)
conf_matrix2


ac4<- 0.8254

# random forest

roc4 <- roc(response = teste$cb_person_default_on_file, 
            predictor = factor(prob_pred2, ordered = TRUE))

ggplotly(
  ggroc(roc4, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(roc4$auc, 4))) +
    theme_bw())



auc4 = 0.8802

#---------------------------------------------------------#
#                                                         #
#                   ÁRVORE DE DECISÃO                     #
#                                                         #
#---------------------------------------------------------#


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

fit4 <- train(cb_person_default_on_file ~., data = treino, method = "rpart",
              parms = list(split = "information"),
              trControl=trctrl,
              tuneLength = 10)

pred4 <- predict(fit4, newdata = teste[-1])

confusionMatrix(pred4,teste$cb_person_default_on_file)

ac5<- 0.8254       

library(pROC)

#  previsões de probabilidades
prob_pred4 <- predict(fit4, newdata = teste[-1], type = "prob")[, "1"]


# Gráfico da árvore

prp(fit4$finalModel, box.palette="Reds", tweak= 1)
# Crie o objeto roc usando as probabilidades previstas
roc5 <- roc(ifelse(teste$cb_person_default_on_file == "1", 1, 0), prob_pred4)

#  AUC
auc_value5 <- auc(roc5)
auc_value5
auc5 = 0.8331


roc5 <- roc(response = teste$cb_person_default_on_file, 
            predictor = factor(prob_pred4, ordered = TRUE))

ggplotly(
  ggroc(roc5, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(roc5$auc, 4))) +
    theme_bw())


#---------------------------------------------------------#
#                                                         #
#                ENSEMBLE - MAJORITY VOTING               #
#                                                         #
#---------------------------------------------------------#

# REGRESSÃO LOGISTICA
cutofff <- 0.23
pred_logistic <- ifelse(predict(stepwise, teste, type = "response") > cutofff, 1, 0)
prob_logistic <- predict(stepwise, teste, type = "response")
# BAGGING
pred_bagging <- predict(fit1, newdata = teste)
prob_bagging <- predict(fit1, newdata = teste, type = "prob")[, "1"]
# RANDOM FOREST
pred_rf <- predict(fit2, newdata = teste)  
prob_rf <- predict(fit2, newdata = teste, type = "prob")[, "1"]

# DECISION TREE
pred_tree <- predict(fit4, newdata = teste)  # Decision Tree
prob_tree <- predict(fit4, newdata = teste, type = "prob")[, "1"]

# tudo nos conformes
(length(pred_rf) == length(pred_bagging)) && 
  (length(pred_tree) == length(pred_logistic))

df_predict_models <- data.frame(logistica = pred_logistic,
                                bagging = pred_bagging,
                                random_forest = pred_rf,
                                d_tree = pred_tree)

# Função para calcular a moda (votação por maioria)
majority_vote <- function(x) {
  factor(ifelse(mean(as.numeric(x)) >= 0.5, 1, 0), levels = c(0, 1))
}

ensemble <- apply(df_predict_models,1,majority_vote)
prob_ensemble <- apply(cbind(prob_logistic,prob_bagging,prob_rf,prob_tree),
                      1,
                      mean)
prob_ensemble[1]
df_predict_models$ensemble <- ensemble

df_predict_models
df_probability_models <- cbind(prob_logistic,prob_bagging,prob_rf,prob_tree, prob_ensemble)
head(df_probability_models)
# Matriz de confusão para o cutoff estabelecido
confusionMatrix(data = ensemble,
                reference = as.factor(teste$cb_person_default_on_file), 
                positive = "1")

acc_ensemble <- 0.8243 

# ROC Curve

# Gerando a curva ROC para o modelo final
ROC <- roc(response = teste$cb_person_default_on_file, 
           predictor = prob_ensemble)

ggplotly(
  ggroc(ROC, color = "blue", linewidth = 0.7) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1),
                 color="grey",
                 linewidth = 0.2) +
    labs(x = "Especificidade",
         y = "Sensibilidade",
         title = paste("Área abaixo da curva (AUC):",
                       round(ROC$auc, 4))) +
    theme_bw())

roc_ensenble <- 0.8841

