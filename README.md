# Llm_Api
This project is a mix of smaller projects, testing llm and AI models
- There is PowerShell scripts to setup dependices, for the enviorment.
- Project is tested with Python ver. 3.13.9

## ApiCall
Diffrent scripts used to acces an instance of ollama, to be implimented into applications.


## Service
Project is made by a series of micro services; a client, sends a promt to a llm, through a RAG.
- dep_download.ps1: Download dependicis for later offline/controlled setup.
- dep_install_service.ps1: Install dependices from wheelhouse.
- data-folder: Holds generated data.
- doc_folder-folder: Takes data in *.txt, *.pdf, *.epub, *.mobi used for RAG data.
- vector_store-folder: RAG data.
- build_db.ps1 is uset to execute build_index.py, to generate the RAG Vector database from provided data.
- client.py: Client to send promt.
- request.ps1: Simple test script to send promt.
- main.py: Main service.
- ollama_client.py: Service to send promt to olama
- rag.py: RAG Service.
- run.ps1: Restart main service.


## Training
Project is to train/tune and then export the model to *.GUFF format
- dep_install_training.ps1: Install dependices (Jackhammer-style).
- dep_updater_training.ps1: Update dependices.
- data-folder: Holds configuration, and generated data.
- doc_folder-folder: Takes data in *.txt, *.pdf, *.epub, *.mobi, *.doc, *.docx used to generate custom dataset.
- dataset_generator.py: Can generate a custom dataset from files in the doc_folder-folder.
- train_model.ps1: Sets parameters for model_trainer.py, to train a model
- tune_model.ps1: Sets parameters for model_tuner.py, to tune pretrained model.
- exportToGUFF.ps1: Sets parameters for convert.py, to convert generated model to *.GUFF format
