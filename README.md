# desafio-dio-dp100
Desafio da DIO para a Certificaçao DP100

---
updated_date: 2025-03-14
created_date: 2025-03-14
---
# Projeto 1 -  **Previsão de Vendas de Açaí – Um Modelo de Machine Learning Aplicado à Açaiteria Serra do Cipó**

## Sumário
- [ ] Caso de uso 
- [ ] Requisitos 
- [ ] Conjunto de dados 
- [ ] Prepara o ambiente no Azure Machine Learning
- [ ] Carregar e explorar os dados de vendas de Açaí
- [ ] Treinar um modelo de Regressão
- [ ] Registrar o  modelo e os experimentos usando MLFlow
- [ ] Implantar o modelo para fazer previsões em tempo real

##  **Caso de Uso**

A **Açaiteria Serra do Cipó** é uma fábrica do frozien de açaí localizada em **Belo Horizonte** que fornece para Belo Horizonte e toda região metropolitana, onde o consumo de sorvetes é elevado ao longo do ano mas o clima parece influenciar a quantidade de açaí vendidos. Para otimizar a fabricação e as vendas, o setor de negócios levantar a hipótese de realizar um estudo para confirmar a influência da temperatura nas vendas do produtos e assim, construir uma solução de previsão de vendas e impactar, positivamente, na sustentabilidade da Açaiteria e otimizar compras com fornecedores para a fabricaçao do frozien. 

Com base na observação de que a quantidade de baldes de açai de 30L vendidos diariamente apresenta uma forte correlação com a temperatura ambiente, a **Açaiteria** identificou um desafio operacional crítico: **como otimizar a produção para evitar desperdícios e, ao mesmo tempo, garantir que o estoque seja suficiente para atender à demanda?** Produzir além do necessário pode gerar prejuízos devido maior espaço para armazenamento, maior gasto com energia para refrigeração, menor fornecimento de frozien açaí fresco, enquanto uma produção abaixo da demanda pode resultar em oportunidades de venda perdidas.

## 1.3 **Desafio e Solução**

Para solucionar esse problema, foi desenvolvido um modelo de **Machine Learning** capaz de prever a quantidade de baldes de açaí de,30L que serão vendidos com base na temperatura. Esse modelo permite à açaiteria antecipar a demanda, ajustar a produção de forma estratégica e garantir um gerenciamento de estoque mais eficiente.

O projeto faz uso de conceitos fundamentais de aprendizado de máquina, análise exploratória de dados e técnicas de modelagem preditiva para aprimorar o processo de tomada de decisão. Com isso, a **Açaiteria Serra do Cipó** não apenas melhora a eficiência operacional, mas também aprimora a experiência dos clientes, garantindo que apenas Açaí fresco esteja na experiência dos clientes com o produto da fábrica.

### **Impacto e Relevância**

A aplicação de **Machine Learning na cadeia de produção** de pequenas e médias empresas, representa um avanço significativo na forma como os negócios locais podem utilizar a tecnologia para resolver desafios práticos do dia a dia. Ao compartilhar essa solução, este projeto reforça a importância da análise de dados na otimização de processos e demonstra como a inteligência artificial pode ser um diferencial competitivo no setor alimentício.

## **Requisitos**

O sistema desenvolvido deve ser capaz de: 

✅ **Treinar um modelo de Machine Learning** para prever as vendas de açaí com base na temperatura do dia.  
✅ **Registrar e gerenciar o modelo** usando o MLflow.  
✅ **Implementar o modelo para previsões em tempo real** em um ambiente de Cloud Computing, usando a Microsoft Azure.  
✅ **Criar um pipeline estruturado** para garantir reprodutibilidade e facilitar a manutenção por todo o time envolvido.


## Dataset
O dataset representa um base de dados com data, as vendas de baldes de açaí de 30L (Em unidades) e temperatura do dia (ºC). O data set possui cerca 50 linhas e foi criado com o Copilot.



## Preparando o ambiente no Azure Machine Learning
- [ ] Criar um Resource Group: rg-projetoum-dpcem
	- [ ] Atribua um nome: rg-dio

       ![image](https://github.com/user-attachments/assets/95292861-82a4-4606-8523-8688216488b5)
       ![image](https://github.com/user-attachments/assets/934f5e5a-d379-483e-9944-15b8106a1c94)

       


![[Pasted image 20250312120144.png]]

### 1.6.1 Cria o Machine Learning Workspace
![[Pasted image 20250312123511.png]]
### 1.6.2 Criar uma VM (Instância)
- Obs:
	- Recomendamos pegar a mais barata.
	-![[Pasted image 20250312135012.png]] 
	
### 1.6.3 Crie um Cluster


### 1.6.4 Upload o Dataset
- Pegue o csv criado no LLM e faça o upload
- Usamos o local file
- No caso não utilizamos a coluna Data
- ![[Pasted image 20250312135700.png]]
- ![[Pasted image 20250312135754.png]]
## 1.7 Treinar um modelo de Regressão

### 1.7.1 Criar um AutoML do Modelo
- Fazemos um AutoML: 
- New experiment name: experimento-automl
- TaskType: Regress

![[Pasted image 20250312135858.png]]

- Escolha o target:
- Target column: Vendas de sorvete
- Limits:
	- Max nodes: 2
	- Experiment timeout: 15
	- Interation timeout: 15 
- Em Task settigns:
- É recomendado bloquear os demais modelos e usar somente de regressao
	- Vai em additional configuration settings:
	- Blocked Models:
		- Marque todos menos o XGBoostRegressor
- ![[Pasted image 20250312140901.png]]
- ![[Pasted image 20250312141128.png]]
## 1.8 Criar um Designer
- Vamos criar um designer um outra maneira de treinamos os modelos
![[Pasted image 20250312141242.png]]
### 1.8.1 Selecione o dataset e arraste para o flow
![[Pasted image 20250312141411.png]]

### 1.8.2 Criar o fluxo no Designer
![[Pasted image 20250312142337.png]]
#### 1.8.2.1 Configure e Submit o fluxo
![[Pasted image 20250312142509.png]]

![[Pasted image 20250312142614.png]]

#### 1.8.2.2 Cruxe os dedos
![[Pasted image 20250312142641.png]]
## 1.9 Parabens Voce acabou o primeiro projeto


