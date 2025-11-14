# Architecture and Flow Diagrams

This document contains a high-level architecture diagram and a sequence flow for the dataset generation request described in the repository.

## Clean Architecture Diagram (Mermaid)

```mermaid
graph LR
  subgraph Frameworks & Drivers
    CLI[CLI: frameworks_drivers/cli/*]
  end
  subgraph Interface_Adapters
    Controller[Controllers: interface_adapters/controllers/*]
  end
  subgraph Use_Cases
    UseCase[Use Cases: app/use_cases/*]
  end
  subgraph Domain
    DomainEntities[Domain Entities & PathConfig: domain/*]
  end
  subgraph Drivers
    FS[Filesystem / HF Cache]
  end

  CLI --> Controller --> UseCase --> Domain --> FS

  click CLI "./Training/frameworks_drivers/cli" "Open CLI folder"
  click Controller "./Training/interface_adapters/controllers" "Open Controllers"
  click UseCase "./Training/app/use_cases" "Open Use Cases"
  click DomainEntities "./Training/domain" "Open Domain"
```

## Dataset Generation Sequence (Mermaid Sequence Diagram)

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI
    participant Controller as Controller
    participant UseCase as Use Case
    participant Domain as Domain
    participant FS as Filesystem/Cache

    User->>CLI: run generate_dataset --spec my.yaml
    CLI->>Controller: parse args, validate
    Controller->>UseCase: start generation(use spec, paths)
    UseCase->>Domain: resolve paths (PathConfig)
    UseCase->>FS: download / read source files
    UseCase->>UseCase: sanitize/transform data
    UseCase->>FS: write dataset files + dataset_metadata.json
    UseCase->>Controller: return ValidationResult
    Controller->>CLI: exit code based on ValidationResult
    CLI->>User: print status
```

## Notes
- The diagrams are intended for contributors to quickly locate relevant code paths.
- For detailed interaction, open the modules referenced in each layer (mapping in the README).

## High-Level UML Component Diagram (Training & Service)

```mermaid
classDiagram
    direction LR

    class TrainingDomain {
      +PathConfig
      +DatasetPreparationRequest
      +DatasetSpec
    }

    class TrainingUseCases {
      +prepare_dataset()
      +convert_model()
      +migrate_artifacts()
    }

    class TrainingInterfaceAdapters {
      +DatasetPreparationController
      +ConversionController
    }

    class TrainingFrameworksDrivers {
      +generate_dataset_cli
      +download_runner_cli
      +convert_cli
    }

    class ServiceDomain {
      +DocumentChunk
      +VectorMetadata
    }

    class ServiceUseCases {
      +build_index()
      +query_rag()
    }

    class ServiceAdapters {
      +VectorStoreGateway
      +OllamaGateway
    }

    class ServiceFrameworksDrivers {
      +main.py (FastAPI app)
      +build_index.py
      +client.py
    }

    class SharedInfrastructure {
      +HF paths (HF_MODEL_ROOT, HF_DATA_ROOT, HF_CACHE_DIR)
      +wheelhouse/ (offline deps)
    }

    %% Training relationships
    TrainingFrameworksDrivers --> TrainingInterfaceAdapters : calls
    TrainingInterfaceAdapters --> TrainingUseCases : orchestrates
    TrainingUseCases --> TrainingDomain : uses entities

    %% Service relationships
    ServiceFrameworksDrivers --> ServiceUseCases : invokes
    ServiceUseCases --> ServiceDomain : uses entities
    ServiceUseCases --> ServiceAdapters : via interfaces

    %% Shared infra
    TrainingUseCases --> SharedInfrastructure : reads/writes models & data
    ServiceUseCases --> SharedInfrastructure : reads/writes vector sources
```
