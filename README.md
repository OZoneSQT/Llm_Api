# Llm_Api
This project is a mix of smaller projects made for my own learning, testing llm and AI models
- There is PowerShell scripts to setup dependices, for the enviorment.
- Project is tested with Python ver. 3.13.9
- Focus is to output models that can run on ollama, if you need to run it elseware, you probely know how to import it in your application 😈


## ApiCall
Diffrent scripts used to acces an instance of ollama, to be implimented into applications.


## Service
Project is made by a series of micro services; a client, sends a promt to a llm, through a RAG.
- `dep_download.ps1`: Download dependicis for later offline/controlled setup.
- `dep_install_service.ps1`: Install dependices from wheelhouse.
- `data-folder`: Holds generated data.
- `doc_folder-folder`: Takes data in *.txt, *.pdf, *.epub, *.mobi used for RAG data.
- `vector_store-folder`: RAG data.
- `build_db.ps1` is uset to execute `build_index.`py, to generate the RAG Vector database from provided data.
- `client.py`: Client to send promt.
- `request.ps1`: Simple test script to send promt.
- `main.py`: Main service.
- `ollama_client.py`: Service to send promt to olama
- `rag.py`: RAG Service.
- `run.ps1`: Restart main service.


## Training
Project is to train/tune and then export the model to *.GUFF format
- `dep_install_training.ps1`: Install dependices (Jackhammer-style).
- `dep_updater_training.ps1`: Update dependices.
- `data-folder`: Holds configuration, and generated data.
- `doc_folder-folder`: Takes data in *.txt, *.pdf, *.epub, *.mobi, *.doc, *.docx used to generate custom dataset.
- `generate_dataset.ps1`: executes `dataset_generator.py`: Can generate a custom dataset from files in the doc_folder-folder.
- `train_model.ps1`: Runs parameters for `model_trainer.py`, to train a model
- `exportToGGUF.ps1`: Convert generated model to *.GGUF format using the llama.cpp framework


# Setup / Uninstall
## Precondision
- PowerShell must be enabled on your system, and script files unblocked
- Ollama should be installed on your target computer
- Generated llm location must be accebel for ollama to install the model, like; local on your computer or on a network share.


## Setup
To be able to run the test service: Run `dep_download.ps1` and then `dep_install_service.ps1` to setup dependices. `dep_download.ps1` can be downloaded and moved to a computer without internet access and then installed using `dep_install_service.ps1`, just keep the folder structure.

Refer to the [Ollama documentation](https://ollama.com/docs) for more details on remote model management.
Refer to https://github.com/OZoneSQT/Ollama-Model-Dumper to backup/export installed ollama models


### Setup model (localhost)
- In the output folder, `setup.ps1` is created to install the model on your target. Default is to install on local host.

### Installing a Model on a Remote Ollama Instance
To install a model on an Ollama instance that is not running on localhost, you need to specify the remote server's address when using the installation script or command. For example, update the installation command in `setup.ps1` or your script to target the remote Ollama server:

Replace `<REMOTE-IP-ADDRESS>` and `<PORT>` with your server's actual IP address and port number, and replace `<PATH-TO-MODEL-FILE>` to the path to the model and model file accessebel to the target server. Ensure that the remote Ollama instance is accessible from your network and that any required authentication is handled.

```sh
ollama create <model-name> -f <PATH-TO-MODEL-FILE>Modelfile --host <REMOTE-IP-ADDRESS>:<PORT>
```

### Uninstalling model (localhost)
In the output folder, `uninstall.ps1` is created to install the model on your target. Default is to install on local host.

### Uninstalling  a Model on a Remote Ollama Instance
To uninstall a model on an Ollama instance that is not running on localhost, you need to specify the remote server's address when using the uninstallation script or command. For example, update the uninstallation command in `uninstall.ps1` or your script to target the remote Ollama server:

Replace `<REMOTE-IP-ADDRESS>` and `<PORT>` with your server's actual IP address and port number. Ensure that the remote Ollama instance is accessible from your network and that any required authentication is handled.

```sh
ollama rm <model-name> --host <REMOTE-IP-ADDRESS>:<PORT>
```
