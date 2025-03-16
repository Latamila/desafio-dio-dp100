# desafio-dio-dp100
Desafio da DIO para a Certificaçao DP100

---
updated_date: 2025-03-14
created_date: 2025-03-14
---
<h1>Projeto 1 -  **Previsão de Vendas de Açaí – Um Modelo de Machine Learning Aplicado à Açaiteria Serra do Cipó**</h1>


<h2> Sumário </h2>
- [ ] Caso de uso 
- [ ] Requisitos 
- [ ] Conjunto de dados 
- [ ] Prepara o ambiente no Azure Machine Learning
- [ ] Carregar e explorar os dados de vendas de Açaí
- [ ] Treinar um modelo de Regressão
- [ ] Registrar o  modelo e os experimentos usando MLFlow
- [ ] Implantar o modelo para fazer previsões em tempo real

<h3> Caso de Uso </h3>

A **Açaiteria Serra do Cipó** é uma fábrica do frozien de açaí localizada em **Belo Horizonte** que fornece para Belo Horizonte e toda região metropolitana, onde o consumo de sorvetes é elevado ao longo do ano mas o clima parece influenciar a quantidade de açaí vendidos. Para otimizar a fabricação e as vendas, a área de negócios levantou a hipótese de realizar um estudo para confirmar a influência da temperatura nas vendas do produtos e assim, construir uma solução de previsão de vendas e impactar, positivamente, na sustentabilidade da Açaiteria e otimizar compras com fornecedores para a fabricaçao do frozie açaí. 

Com base na observação de que a quantidade de baldes de açai de 30L vendidos diariamente apresenta uma forte correlação com a temperatura ambiente, a **Açaiteria** identificou um desafio operacional crítico: **como otimizar a produção para evitar desperdícios e, ao mesmo tempo, garantir que o estoque seja suficiente para atender à demanda?** Produzir além do necessário pode gerar prejuízos devido maior espaço para armazenamento, maior gasto com energia para refrigeração, menor fornecimento de frozie açaí fresco, enquanto uma produção abaixo da demanda pode resultar em oportunidades de venda perdidas.

<h3> Desafio e Solução </h3>

Para solucionar esse problema, foi desenvolvido um modelo de **Machine Learning** capaz de prever a quantidade de baldes de açaí de 30L que serão vendidos com base na temperatura. Esse modelo permite à açaiteria antecipar a demanda, ajustar a produção de forma estratégica e garantir um gerenciamento de estoque mais eficiente.

Os steakholders pretendem criar um planejamento de 15 dias integrando dados meteorológicos futuros da região metropolitana de Belo horizonte, através da aplicação do Google Clima, extraindo os dados das previsões média diárias para cada cidade atendida pela empresa,fazendo a ingestão dos dados no nosso modelo para uma Inferência em Tempo Real.

O projeto faz uso de conceitos fundamentais de aprendizado de máquina, análise exploratória de dados e técnicas de modelagem preditiva para aprimorar o processo de tomada de decisão. Com isso, a **Açaiteria Serra do Cipó** não apenas melhora a eficiência operacional, mas também aprimora a experiência dos clientes, garantindo que apenas Açaí Frozie fresco esteja na experiência dos clientes, ao consumir o produto da fábrica.

### **Impacto e Relevância**

A aplicação de **Machine Learning na cadeia de produção** de pequenas e médias empresas, representa um avanço significativo na forma como os negócios locais podem utilizar a tecnologia para resolver desafios práticos do dia a dia. Ao compartilhar essa solução, este projeto reforça a importância da análise de dados na otimização de processos e demonstra como a inteligência artificial pode ser um diferencial competitivo no setor alimentício.

## **Requisitos**

O sistema desenvolvido deve ser capaz de: 

✅ **Treinar um modelo de Machine Learning** para prever as vendas de açaí com base na temperatura do dia.  
✅ **Registrar e gerenciar o modelo** usando o MLflow.  
✅ **Implementar o modelo para previsões em tempo real** em um ambiente de Cloud Computing, usando a Microsoft Azure.  
✅ **Criar um pipeline estruturado** para garantir reprodutibilidade e facilitar a manutenção por todo o time envolvido.


## Dataset
O dataset representa uma base de dados com data, as vendas de baldes de açaí de 30L (Em unidades) e temperatura do dia (ºC). O data set possui cerca 50 linhas e foi criado com o Copilot.

A tabela criada deve ser do tipo mltable. 
A coluna data foi retirada antes de ser inserida no Workspace do Azure. 

![image](https://github.com/user-attachments/assets/bb1dd30a-0567-400e-82f1-91f20e2221d3)
![image](https://github.com/user-attachments/assets/e8326ea5-ebb6-4086-a29a-5c1249ab2a15)




<h1> Preparando o ambiente no Azure Machine Learning </h1>

- [ ] Criar um Resource Group
- [ ] Atribua um nome

![image](https://github.com/user-attachments/assets/95292861-82a4-4606-8523-8688216488b5)
![image](https://github.com/user-attachments/assets/934f5e5a-d379-483e-9944-15b8106a1c94)



<h2> Cria o Machine Learning Workspace </h2>

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

  ### Criar jobs pelo script Python e visualizar no Workspace do Azure Machine Learning

  ![image](https://github.com/user-attachments/assets/90191209-ee2a-4e30-bf73-1cb72b6eb48a)
  ![image](https://github.com/user-attachments/assets/aa80f326-e56b-4ea1-a57d-3cc738facd6f)


  <h2> Criar Notebook para o Experimento com MLFlow </h2>
   ---

- Ir até a instância de Computação e abrir o aplicativo jupyter LAB.

  Neste estudo de caso, me atentarei a demonstrar como integrar o MLFlow a um experimento genérico, mas o experimento completo caso queira está no arquivo run_experiment_MLFlow, neste repositório, dado que iremos implantar de ponta-a-ponta um modelo com AutoML e com Designer Azure Machine Learning. Apenas por questões de custos pois paguei pela execução do projeto no Portal Azure. 
  
  
  ![image](https://github.com/user-attachments/assets/16e8696b-8216-41d9-8adc-4ae5d319609a)



  



Fazer Pré processamento dos dados, transformando o dataset inicial dadosExperimento.csv em um sample.csv para o output do experimento. 

![image](https://github.com/user-attachments/assets/6a15fc41-1b0b-468b-a19a-90481f5ca465)

sample.csv

![image](https://github.com/user-attachments/assets/7a22e0eb-2e9d-4eb7-85d8-64afcdc38428)

### Retrieve experiment details using the SDK
---

The *run* variable in the code I ran previously is an instance of a Run object, which is a reference to an individual run of an experiment in Azure Machine Learning. You can use this reference to get information about the run and its outputs:

  ![image](https://github.com/user-attachments/assets/1d142d55-2014-47d0-80e8-5e1559f23576)

Aqui mostrará métricas geradas a partir do experimento. Note-se que o experimento detectou que há 50 amostras. Se ele produzir outras métricas, estas serão exibidas neste momento. 

You can download the files produced by the experiment, either individually by using the download_file method, or by using the download_files method to retrieve multiple files. The following code downloads all of the files in the run's output folder:

![image](https://github.com/user-attachments/assets/1d9d76cc-f36a-42d2-9fd9-23e1cd9cca18)

If ou need to troubleshoot the experiment run, you can use the get_details method to retrieve basic details about the run, or you can use the get_details_with_logs method to retrieve the run details as well as the contents of log files generated during the run:

![image](https://github.com/user-attachments/assets/136f6470-6739-4c81-8c7c-ddfe2061c1c7)


Note that the details include information about the compute target on which the experiment was run, the date and time when it started and ended. 


<h1> Run an experiment script </h1>
---


In the previous example, I ran an experiment inline in this notebook. A more flexible solution is to create a separate script for the experiment, and store it in a folder along with any other files it needs, and then use Azure ML to run the experiment based on the script in the folder.


Now we'll create a Python script containing the code for our experiment, and save it in the experiment folder.

Note: running the following cell just creates the script file - it doesn't run it!
First, let's create a folder for the experiment files, and copy the data into it:

![image](https://github.com/user-attachments/assets/41835688-d1ce-47fb-9c3e-462d1369fed9)



Now we'll create a Python script containing the code for our experiment, and save it in the experiment folder.

Note: running the following cell just creates the script file - it doesn't run it!

This code is a simplified version of the inline code used before. However, note the following:

It uses the Run.get_context() method to retrieve the experiment run context when the script is run.
It loads the açai Sales'data from the folder where the script is located.
It creates a folder named outputs and writes the sample file to it - this folder is automatically uploaded to the experiment run
Now you're almost ready to run the experiment. 

To run the script, you must create a ScriptRunConfig that identifies the Python script file to be run in the experiment, and then run an experiment based on it.

Note: The ScriptRunConfig also determines the compute target and Python environment. If you don't specify these, a default environment is created automatically on the local compute where the code is being run (in this case, where this notebook is being run).

![image](https://github.com/user-attachments/assets/ac394457-b7f3-4704-9143-d50920da306c)
![image](https://github.com/user-attachments/assets/86afac44-5f95-4cad-9668-5dbd991fc685)


As before, you can use the widget or the link to the experiment in Azure Machine Learning studio to view the outputs generated by the experiment, and you can also write code to retrieve the metrics and files it generated:

![image](https://github.com/user-attachments/assets/143b1dd3-dbd4-41c2-be2d-7b6b055f39c6)

Note that this time, the run generated some log files. You can view these in the widget, or you can use the get_details_with_logs method like we did before, only this time the output will include the log data.

![image](https://github.com/user-attachments/assets/478f8e6b-0b05-4488-9be1-5fd7c2d6ff8e)


Although you can view the log details in the output above, it's usually easier to download the log files and view them in a text editor.

![image](https://github.com/user-attachments/assets/547b00f3-ced8-4dd5-86a7-951a9a824039)

![image](https://github.com/user-attachments/assets/52217df2-3da9-41a0-a92c-62f9ed3b38f5)

### View experiment run history
---

Now that you've run the same experiment multiple times, you can view the history in Azure Machine Learning studio and explore each logged run. Or you can retrieve an experiment by name from the workspace and iterate through its runs using the SDK:

![image](https://github.com/user-attachments/assets/b95f5237-84d5-4c42-89d1-120ecfb8a01e)


<h1> **Use MLflow** </h1>
---


MLflow is an open source platform for managing machine learning processes. It's commonly (but not exclusively) used in Databricks environments to coordinate experiments and track metrics. In Azure Machine Learning experiments, you can use MLflow to track metrics as an alternative to the native log functionality.
Acompanhamento é o processo de salvar informações relevantes sobre experimentos. Neste estudo de caso, você aprenderá a usar o MLflow para acompanhar seus experimentos e execuções em workspaces do Azure Machine Learning.

To take advantage of this capability, you'll need the mlflow and azureml-mlflow packages.

![image](https://github.com/user-attachments/assets/ce6aae46-6eae-44ec-9c93-1f7610e007b5)

Use MLflow with an inline experiment
To use MLflow to track metrics for an inline experiment, you must set the MLflow tracking URI to the workspace where the experiment is being run. This enables you to use mlflow tracking methods to log data to the experiment run.

![image](https://github.com/user-attachments/assets/36d3e83a-ab2d-4c58-8bbf-bb9a88b200e4)

No workspace do Azure Machine Learning

![image](https://github.com/user-attachments/assets/a7b69b07-cb54-494a-b440-e5dbed552f1b)

![image](https://github.com/user-attachments/assets/eedb0ad1-146d-4370-b427-7cc517f3e02c)

![image](https://github.com/user-attachments/assets/c94f3bdd-8a91-46fe-ab07-c787506c306f)

![image](https://github.com/user-attachments/assets/fbfbed14-53f1-4f95-906d-e6c4685f91f1)

After running the code above, you can use the link that is displayed to view the experiment in Azure Machine Learning studio. Then select the latest run of the experiment and view its Metrics tab to see the logged metric.


Use MLflow in an experiment script
You can also use MLflow to track metrics in an experiment script.

Run the following two cells to create a folder and a script for an experiment that uses MLflow

![image](https://github.com/user-attachments/assets/34add555-fc20-4656-8f78-0eee1a0df7c2)

When you use MLflow tracking in an Azure ML experiment script, the MLflow tracking URI is set automatically when you start the experiment run. However, the environment in which the script is to be run must include the required mlflow packages.


![image](https://github.com/user-attachments/assets/728d7a2b-c02d-41e4-89b6-59443ac8f1d6)

logs gerados para monitoramento cliquei no link em destaque para visualizar em .txt.

![image](https://github.com/user-attachments/assets/e6adfc1f-f228-4ee6-b508-475d9e8cd9be)


![image](https://github.com/user-attachments/assets/806a6dcd-90c1-4a96-aee8-cb1c1dc6a299)


![image](https://github.com/user-attachments/assets/2b52d2b2-b146-4ebe-a458-663cbcd153ba)

As usual, you can get the logged metrics from the experiment run when it's finished.


<h1> 1.7.1 Criar um AutoML do Modelo </h1>
-----

- Fazemos um AutoML: 
- New experiment name: experimento-automl
- TaskType: Regression



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
  ![image](https://github.com/user-attachments/assets/60b5531b-7f9c-41cf-b78e-850c7d7ce3b4)
  ![image](https://github.com/user-attachments/assets/9fb012d0-2786-48de-a5f3-405518fbb898)

Escolha compute cluster para executar o treinamento

![image](https://github.com/user-attachments/assets/66b67493-9dc9-4e82-ae14-1f45b8dbac50)

![image](https://github.com/user-attachments/assets/d30519d2-8f58-477c-9598-477c0bfcb625)

![image](https://github.com/user-attachments/assets/8e9df4ba-7a25-43fb-b014-4ac3dc0e43ec)

![image](https://github.com/user-attachments/assets/631ca405-5767-4d12-ba5a-b6b09d9eca99)


### Durante o Treinamento 

![image](https://github.com/user-attachments/assets/ed3cfd1b-dc0e-449c-b36a-0a7d182f078e)
![image](https://github.com/user-attachments/assets/96d6d5d4-e52c-4669-bd1d-4d2fb5c2c7f7)

###
Previsão 

![image](https://github.com/user-attachments/assets/4b73aeab-99df-44c2-80cc-f814e8cc6228)


### RESULTADOS 
![image](https://github.com/user-attachments/assets/34ceaf18-d733-4467-a26b-fb589a17f3ee)

![image](https://github.com/user-attachments/assets/92c7abd0-644e-4d8e-bfbe-153f158f41d8)

![image](https://github.com/user-attachments/assets/3a55aeb0-a299-4c27-b643-a23cd6fb9802)
![image](https://github.com/user-attachments/assets/1d5ddc5d-0814-4f84-9617-08524ca89166)
![image](https://github.com/user-attachments/assets/38b29a3c-284b-406a-bfb2-b5d45f2622c1)
![image](https://github.com/user-attachments/assets/a0f66f05-ec2b-4ca1-9dca-ec4ea4c57f8d)



<h1> EXPLICAÇÕES E MÉTRICAS </h1>

<h2>Métricas de Avaliação </h2>

As seguintes métricas foram utilizadas para avaliar a performance do modelo:

### Erro Absoluto Médio Normalizado (Normalized MAE): 0,096680

Indica a média das diferenças absolutas entre as previsões e os valores reais, normalizada. Um valor próximo de zero é desejável e este resultado sugere um bom desempenho nesse aspecto.

## Erro Mediano Absoluto Normalizado (Normalized Median AE): 0,075984

Métrica robusta contra outliers, indicando que a maioria dos erros está concentrada em valores pequenos. O modelo apresenta bom resultado.

# Erro de Quadrado da Média Raiz Normalizado (Normalized RMSE): 0,12917

Mede a magnitude do erro. Embora o valor seja maior que o MAE, ele está relativamente baixo, mostrando que os erros não se espalham de forma descontrolada.

# Erro de Log do Quadrado da Média Raiz Normalizado: 0,13318

Indica que o logaritmo dos erros também está bem controlado, sugerindo estabilidade do modelo.

# Pontuação R²: 0,66160

Avalia a proporção da variabilidade dos dados explicada pelo modelo. Um valor de 0,66160 indica que aproximadamente 66% da variabilidade dos dados é explicada pelo modelo, o que é satisfatório em muitos casos, mas pode depender do contexto do domínio.

# Erro de Raiz do Valor Quadrático Médio (RMSE): 7,7504

Embora esta métrica seja influenciada por outliers, os valores absolutos devem ser comparados com a escala dos dados para uma interpretação mais robusta.

# Erro de Log de Raiz do Valor Quadrático Médio: 0,11634

Complementa o RMSE ao analisar os logs, mostrando que o erro, em escala logarítmica, está baixo.

# Correlação de Spearman: 0,84747

Mede a força e direção da relação monotônica entre variáveis. O valor alto (0,84747) demonstra que o modelo captura bem as tendências dos dados.

<h2> Interpretação e Conclusão </h2>

Com base nas métricas analisadas, destacam-se os seguintes pontos:

O Normalized MAE e Normalized RMSE estão em níveis aceitáveis, indicando que os erros são baixos em média e têm variabilidade controlada.

O valor de R² (0,66160) demonstra que o modelo é capaz de explicar uma parte significativa da variabilidade dos dados, embora haja espaço para melhorias.

A Correlação de Spearman (0,84747) indica uma forte relação monotônica, evidenciando que o modelo respeita bem as tendências dos dados.

Métricas baseadas em logarítmos confirmam que os erros estão bem distribuídos.

<h1> Conclusão Geral</h1> 

O modelo apresenta bom desempenho, especialmente para aplicações onde um nível moderado de erro é aceitável. No entanto, se o contexto exigir alta precisão (por exemplo, áreas como medicina ou finanças), pode ser necessário otimizar ou adotar técnicas avançadas de ajuste. De forma geral, o modelo é adequado para uso inicial.

<h2>Recomendações para os Stakeholders</h2>

Considere o modelo para implementação inicial, mas monitore seu desempenho em dados reais.

Explore técnicas de ajuste fino para melhorar a explicação de variabilidade (R²).

Realize uma análise de importância das características (features) para identificar possíveis variáveis de maior impacto que possam aprimorar a precisão.

![image](https://github.com/user-attachments/assets/c1c8aee2-37e9-4f1a-a38d-1026a1e81d21)

![image](https://github.com/user-attachments/assets/0ae1f81b-01a9-4fe5-a029-9c9fd4a689b8)
![image](https://github.com/user-attachments/assets/844e0b2e-0685-456a-b1b3-f0f2ae357818)

Aqui está as métricas de uma outra versão e será comparado com a acima exposta.
![image](https://github.com/user-attachments/assets/a5bf9013-d6b7-4e4a-b746-08df4f25de2e)

**Comparação de Desempenho Entre as Versões do Modelo**

Com base nas métricas fornecidas para as duas versões do modelo, vamos analisar e comparar os desempenhos para determinar qual delas apresenta melhores resultados.

---

### **Métricas de Avaliação - Versão 1 vs. Versão 2**
Aqui está a comparação direta das métricas em formato tabular para facilitar a análise:

| **Métrica**                                 | **Versão 1**           | **Versão 2**           | **Interpretação**                                                                                                                                       |
|---------------------------------------------|------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Erro Absoluto Médio Normalizado**         | 0,096680              | 0,762599              | A **Versão 1** tem um desempenho significativamente melhor, com menor erro absoluto médio normalizado.                                                  |
| **Erro Mediano Absoluto Normalizado**       | 0,075984              | 5,800801              | Novamente, a **Versão 1** apresenta um erro muito menor, mostrando maior precisão em relação à mediana dos erros.                                        |
| **Erro de Quadrado da Média Raiz Normalizado** | 0,12917               | 8,728540              | A **Versão 1** apresenta um erro normalizado muito mais controlado em comparação à Versão 2.                                                            |
| **Erro de Log do Quadrado da Média Raiz Normalizado** | 0,13318               | 4,559015              | O valor logarítmico do erro é bem menor na **Versão 1**, indicando maior estabilidade.                                                                   |
| **Pontuação R²**                            | 0,66160               | 0,09668               | A **Versão 1** explica uma proporção muito maior da variabilidade dos dados (66,16%) do que a Versão 2 (9,67%).                                         |
| **Erro de Raiz do Valor Quadrático Médio**  | 7,7504                | 0,1291733             | Embora as escalas de interpretação precisem ser consideradas, a **Versão 2** parece ter menor erro absoluto médio.                                 |
| **Erro de Log de Raiz do Valor Quadrático Médio** | 0,11634               | 0.8474679             | A **Versão 1** também apresenta valores menores em log para erro absoluto 


<h1> Criar um Designer </h1>
---

- Vamos criar um designer um outra maneira de treinamos os modelos
  
![image](https://github.com/user-attachments/assets/b3508571-3832-4a63-9a5c-e466581dbe97)



### Criar o fluxo no Designer

![image](https://github.com/user-attachments/assets/714f4a1c-3d8c-4f95-899b-6446fb8d9fa9)

#### 1.8.2.1 Configure e Submit o fluxo

![image](https://github.com/user-attachments/assets/e49c6029-e613-4d25-a765-eded9011443b)

![image](https://github.com/user-attachments/assets/54adca11-b78a-4969-9b42-e7736a44f737)


![image](https://github.com/user-attachments/assets/63ccfdf9-4960-4bde-a1b8-86886a4cec33)

![image](https://github.com/user-attachments/assets/7ab4bd60-4bc9-4ccf-bbb3-df7b330adcc2)



#### Fase de Treinamento

![image](https://github.com/user-attachments/assets/feb5c884-23c9-4e7e-98bb-f5583884c80f)

![image](https://github.com/user-attachments/assets/d514d7c8-ca40-49b5-b7cf-392fc37c6681)

![image](https://github.com/user-attachments/assets/b58096f2-dabf-47f5-a2a1-9e1b24c3e630)

![image](https://github.com/user-attachments/assets/62b5c492-15af-461f-be4a-6444f96ca80a)

![image](https://github.com/user-attachments/assets/81ef344f-c7fd-4534-922a-260e9915cb8a)


![image](https://github.com/user-attachments/assets/fd599799-25e5-4ef0-a576-52860857d674)

![image](https://github.com/user-attachments/assets/ac210ca1-7e95-40dd-844d-299615a28436)

![image](https://github.com/user-attachments/assets/9a199b38-a4b7-4e26-a2f9-8ef126dc1d05)

![image](https://github.com/user-attachments/assets/2e189eba-4cef-4e3b-b9f5-ffb5a600a41f)

### Modelo Treinado

![image](https://github.com/user-attachments/assets/7c6b9aeb-df85-47db-950c-561bb38b794e)
![image](https://github.com/user-attachments/assets/6ad8e253-533e-4f36-b60f-77de93143166)

![image](https://github.com/user-attachments/assets/1485cd20-c1a2-4851-89eb-580da69905d7)

Logs gerados 

![image](https://github.com/user-attachments/assets/91693ee4-a0d3-430e-b48f-a662807977bf)


### RESULTADOS DO DESIGNER MACHINE LEARNING 

### Score Model 

![image](https://github.com/user-attachments/assets/7f786b6d-bdba-499b-b27d-fb8af985b096)

Note que a coluna da esquerda são os dados reais, no centro a coluna de Temperatura e à direita, está o que foi previsto pelo Modelo. 

### Métricas de Avaliação do Modelo 

![image](https://github.com/user-attachments/assets/6a78bae0-130a-4891-bd52-f5385b3760b5)


<h2>Interpretação Geral</h2>

A RAE e o R² sugerem que o modelo oferece uma boa explicação para os dados, com um desempenho confiável e consistente.

O valor de R² (0.7037) é superior ao modelo anterior (0.6616) feito com o AUTOML, indicando que esta versão captura uma maior proporção da variabilidade nos dados.

O erro absoluto relativo (RAE) também está em um nível aceitável, reforçando que o modelo reduz o erro de maneira eficiente.

Com base nesses resultados, o novo modelo apresenta bom desempenho, especialmente em termos de erro relativo, conforme indicado pelos valores baixos de RSE (0.296288) e RAE (0.378562). Embora os valores de MAE (5.402082) e RMSE (9.101607) possam parecer elevados, é essencial considerar o contexto e a escala dos dados para interpretá-los adequadamente.

Comparando com os modelos anteriores, este parece ser o modelo mais forte até agora, especialmente por explicar uma maior porcentagem dos dados com um bom equilíbrio entre erro absoluto e relativo.

<h1> DEPLOY DO AUTOML </h1> 

De todos os modelos treinados ao final foi escolhido este para deploy:

![image](https://github.com/user-attachments/assets/976000d7-4e1e-4f78-8623-5a1cf0c55cf0)

Suas métricas estavam dentro do esperado e a correlação entre as variáveis se mostra alta. 

![image](https://github.com/user-attachments/assets/1bfb7bee-eee6-4a68-95cf-56d4b2e991ef)

Para implantar, escolha um dos tipos de endpoint

![image](https://github.com/user-attachments/assets/22d81ce7-21c9-405b-be83-51ec00b3b719)

O deploy, necessariamente, precisa ser em instâncias de Conteineres do Azure.
![image](https://github.com/user-attachments/assets/c73cbdb0-8b35-4b9a-b0c6-40cf22f42ccc)

Ponto de extremidade criado:
(eles podem demorar até uma hora para aparecer o https. Calma...
![image](https://github.com/user-attachments/assets/29387f0f-fb33-4e32-b420-f81b42fc908f)



## DEPLOY DO DESIGNER

Foi feito o registro do modelo 
![image](https://github.com/user-attachments/assets/9d96a5f9-ed19-4310-a3e3-1422a3ccd60d)

E depois que se registra o modelo, daí sim pode-se implantá-lo. 
![image](https://github.com/user-attachments/assets/049a5006-8b88-4e9e-9cec-f08d8daaae9b)

Para este escolhi um endpoint em tempo real:

![image](https://github.com/user-attachments/assets/2ece36b6-62ed-47f0-a16f-299143adde63)

# AGRADECIMENTOS




