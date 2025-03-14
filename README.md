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

![image](https://github.com/user-attachments/assets/bb1dd30a-0567-400e-82f1-91f20e2221d3)
![image](https://github.com/user-attachments/assets/e8326ea5-ebb6-4086-a29a-5c1249ab2a15)




## Preparando o ambiente no Azure Machine Learning
- [ ] Criar um Resource Group

  - [ ] Atribua um nome

       ![image](https://github.com/user-attachments/assets/95292861-82a4-4606-8523-8688216488b5)
       ![image](https://github.com/user-attachments/assets/934f5e5a-d379-483e-9944-15b8106a1c94)



### Cria o Machine Learning Workspace

  - [ ] Atribua um nome
        ![image](https://github.com/user-attachments/assets/ee02897b-ed58-4d4f-b63c-c70cb2bdbcb5)
        ![image](https://github.com/user-attachments/assets/399fdb54-43e1-4e43-b439-434fcf74817d)
        ![image](https://github.com/user-attachments/assets/47a5fd1c-211c-4749-8d24-28f6d07c141c)


        
        



### 1.6.2 Criar uma VM (Instância)
- Obs:
	- Recomendamos pegar a mais barata.
   ![image](https://github.com/user-attachments/assets/fa4b00ef-3296-44fa-8747-6b62a71e6e32)

	
### 1.6.3 Criar um Cluster

![image](https://github.com/user-attachments/assets/8c9dde4d-5cba-4c8e-a10d-acf5e2518f54)


### Upload o Dataset no Notebook
- Pegue o csv criado no LLM e faça o upload dentro da seção Notebook
- Usamos o local file
- Ir até a instância de Computação e abrir o aplicativo jupyter LAB.

  ### Criar jobs pelo script Python e visualizar no Workspace do Azure Machine Learning

  ![image](https://github.com/user-attachments/assets/90191209-ee2a-4e30-bf73-1cb72b6eb48a)
  ![image](https://github.com/user-attachments/assets/aa80f326-e56b-4ea1-a57d-3cc738facd6f)
  ![image](https://github.com/user-attachments/assets/16e8696b-8216-41d9-8adc-4ae5d319609a)



  



Fazer Pré processamento dos dados, transformando o dataset inicial em um sample para o output do experimento. 

  ![image](https://github.com/user-attachments/assets/6a15fc41-1b0b-468b-a19a-90481f5ca465)

sample.csv
  ![image](https://github.com/user-attachments/assets/7a22e0eb-2e9d-4eb7-85d8-64afcdc38428)

Retrieve experiment details using the SDK
The run variable in the code you ran previously is an instance of a Run object, which is a reference to an individual run of an experiment in Azure Machine Learning. You can use this reference to get information about the run and its outputs:

  ![image](https://github.com/user-attachments/assets/1d142d55-2014-47d0-80e8-5e1559f23576)


You can download the files produced by the experiment, either individually by using the download_file method, or by using the download_files method to retrieve multiple files. The following code downloads all of the files in the run's output folder:

![image](https://github.com/user-attachments/assets/1d9d76cc-f36a-42d2-9fd9-23e1cd9cca18)

If ou need to troubleshoot the experiment run, you can use the get_details method to retrieve basic details about the run, or you can use the get_details_with_logs method to retrieve the run details as well as the contents of log files generated during the run:

![image](https://github.com/user-attachments/assets/136f6470-6739-4c81-8c7c-ddfe2061c1c7)


Note that the details include information about the compute target on which the experiment was run, the date and time when it started and ended. Additionally, because the notebook containing the experiment code (this one) is in a cloned Git repository, details about the repo, branch, and status are recorded in the run history.

In this case, note that the logFiles entry in the details indicates that no log files were generated. That's typical for an inline experiment like the one you ran, but things get more interesting when you run a script as an experiment; which is what we'll look at next.

Run an experiment script
In the previous example, you ran an experiment inline in this notebook. A more flexible solution is to create a separate script for the experiment, and store it in a folder along with any other files it needs, and then use Azure ML to run the experiment based on the script in the folder.

First, let's create a folder for the experiment files, and copy the data into it:

![image](https://github.com/user-attachments/assets/41835688-d1ce-47fb-9c3e-462d1369fed9)




# Verify the files have been downloaded
for root, directories, filenames in os.walk(download_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

## 1.7 Treinar um modelo de Regressão


Now we'll create a Python script containing the code for our experiment, and save it in the experiment folder.

Note: running the following cell just creates the script file - it doesn't run it!

![image](https://github.com/user-attachments/assets/3b5ddc8c-51a0-44de-97b2-c3fa5f554f63)

This code is a simplified version of the inline code used before. However, note the following:

It uses the Run.get_context() method to retrieve the experiment run context when the script is run.
It loads the diabetes data from the folder where the script is located.
It creates a folder named outputs and writes the sample file to it - this folder is automatically uploaded to the experiment run
Now you're almost ready to run the experiment. To run the script, you must create a ScriptRunConfig that identifies the Python script file to be run in the experiment, and then run an experiment based on it.

Note: The ScriptRunConfig also determines the compute target and Python environment. If you don't specify these, a default environment is created automatically on the local compute where the code is being run (in this case, where this notebook is being run).



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


