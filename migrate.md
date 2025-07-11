# Plan de Migration vers une Architecture Professionnelle

## 📋 Vue d'ensemble

Ce document détaille la migration de l'architecture actuelle vers une structure professionnelle de package Python.

## 🎯 Objectifs de la migration

1. **Structure modulaire** : Séparation claire des responsabilités
2. **Maintenabilité** : Code plus facile à maintenir et étendre
3. **Testing** : Infrastructure de tests robuste
4. **Documentation** : Documentation complète et accessible
5. **Distribution** : Package facilement installable et distribuable

## 📊 Comparaison des structures

### Structure Actuelle
```
gama-gymnasium/
├── README.md
├── examples/                    # ✅ Bien organisé
├── gama/
├── python_package/
│   ├── gama_gymnasium/         # ❌ Pas dans src/
│   │   ├── __init__.py
│   │   ├── gama_env.py         # ❌ Monolithique
│   │   └── space_converters.py
│   └── pyproject.toml          # ❌ Configuration minimale
└── tests/                      # ❌ Tests éparpillés
```

### Structure Recommandée
```
gama-gymnasium/
├── src/gama_gymnasium/         # ✅ Layout src/
│   ├── core/                   # ✅ Séparation des concerns
│   ├── spaces/
│   ├── wrappers/
│   └── utils/
├── tests/                      # ✅ Tests structurés
├── docs/                       # ✅ Documentation dédiée
├── examples/                   # ✅ Exemples clairs
├── gama_models/               # ✅ Modèles GAMA séparés
└── pyproject.toml             # ✅ Configuration complète
```

## 🚀 Étapes de migration

### Phase 1: Restructuration des fichiers core (2-3h)

1. **Créer la nouvelle structure**
```bash
mkdir -p src/gama_gymnasium/{core,spaces,wrappers,utils}
mkdir -p tests/{unit,integration,performance}
mkdir -p docs/{api,guide,examples}
mkdir -p gama_models
```

2. **Migrer gama_env.py**
```python
# src/gama_gymnasium/core/gama_env.py
# - Garder seulement la classe GamaEnv
# - Extraire la communication dans client.py
# - Extraire le traitement des messages dans message_handler.py
```

3. **Créer les nouveaux modules**
```python
# src/gama_gymnasium/core/client.py
# - Classe GamaClient pour la communication socket
# - Gestion des connexions et reconnexions
# - Protocol de communication

# src/gama_gymnasium/core/message_handler.py  
# - Traitement des messages JSON
# - Validation des données
# - Sérialisation/désérialisation
```

### Phase 2: Amélioration des espaces et wrappers (1-2h)

4. **Améliorer space_converters.py**
```python
# src/gama_gymnasium/spaces/converters.py
# - Fonctions de conversion existantes
# - Meilleure gestion des erreurs
# - Support pour plus de types d'espaces

# src/gama_gymnasium/spaces/validators.py
# - Validation des définitions d'espaces
# - Vérification de cohérence
```

5. **Créer des wrappers utiles**
```python
# src/gama_gymnasium/wrappers/sync.py
# - Wrapper pour rendre l'environnement synchrone
# - Compatible avec Stable-Baselines3

# src/gama_gymnasium/wrappers/monitoring.py
# - Logging des interactions
# - Métriques de performance
```

### Phase 3: Infrastructure et qualité (1-2h)

6. **Améliorer pyproject.toml**
   - Configuration complète (voir fichier créé)
   - Dépendances optionnelles
   - Outils de développement

7. **Restructurer les tests**
```python
# tests/unit/test_gama_env.py
# tests/unit/test_space_converters.py
# tests/integration/test_full_workflow.py
# tests/performance/test_communication_speed.py
```

8. **Ajouter les outils de qualité**
```yaml
# .pre-commit-config.yaml
# Configuration pour black, isort, flake8, mypy
```

### Phase 4: Documentation et exemples (2-3h)

9. **Documentation API complète**
```python
# Docstrings complets pour toutes les classes/fonctions
# Type hints partout
# Documentation des protocoles GAMA
```

10. **Restructurer les exemples**
```
examples/
├── basic/
│   ├── README.md
│   ├── target_seeking.gaml
│   └── train_basic.py
├── rl_training/
│   ├── cartpole/
│   ├── frozen_lake/
│   └── README.md
└── advanced/
    └── multi_agent/
```

## 📝 Script de migration automatique

<function_calls>
<invoke name="create_file">
<parameter name="content">#!/usr/bin/env python3
"""
Script de migration automatique vers la nouvelle architecture
Usage: python migrate.py
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Crée la nouvelle structure de répertoires"""
    directories = [
        "src/gama_gymnasium/core",
        "src/gama_gymnasium/spaces", 
        "src/gama_gymnasium/wrappers",
        "src/gama_gymnasium/utils",
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "docs/api",
        "docs/guide",
        "docs/examples",
        "gama_models/basic",
        "gama_models/rl_training",
        "gama_models/advanced"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")

def migrate_python_package():
    """Migre le package Python existant"""
    old_package = Path("python_package/gama_gymnasium")
    new_package = Path("src/gama_gymnasium")
    
    if old_package.exists():
        # Copier les fichiers existants
        if not new_package.exists():
            shutil.copytree(old_package, new_package)
            print(f"✅ Migrated {old_package} to {new_package}")
        
        # Réorganiser les fichiers
        files_to_move = {
            "gama_env.py": "core/gama_env.py",
            "space_converters.py": "spaces/converters.py"
        }
        
        for old_file, new_file in files_to_move.items():
            old_path = new_package / old_file
            new_path = new_package / new_file
            
            if old_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                print(f"✅ Moved {old_file} to {new_file}")

def migrate_examples():
    """Réorganise les exemples"""
    examples_mapping = {
        "basic_example": "examples/basic",
        "cartpole DQN": "examples/rl_training/cartpole", 
        "frozen lake QLearning": "examples/rl_training/frozen_lake"
    }
    
    for old_name, new_path in examples_mapping.items():
        old_path = Path(f"examples/{old_name}")
        if old_path.exists():
            new_dir = Path(new_path)
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Copier les fichiers
            for file in old_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, new_dir / file.name)
            
            print(f"✅ Migrated {old_name} to {new_path}")

def migrate_gama_models():
    """Déplace les modèles GAMA vers gama_models/"""
    gama_dir = Path("gama")
    if gama_dir.exists():
        for file in gama_dir.iterdir():
            if file.suffix == ".gaml":
                new_path = Path("gama_models/basic") / file.name
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, new_path)
                print(f"✅ Copied {file.name} to gama_models/basic/")

def update_imports():
    """Met à jour les imports dans les fichiers"""
    # Cette fonction devra être adaptée selon les besoins spécifiques
    print("⚠️  N'oubliez pas de mettre à jour les imports dans les fichiers!")
    print("   Exemples:")
    print("   - from gama_gymnasium import GamaEnv")
    print("   - from gama_gymnasium.core import GamaEnv")
    print("   - from gama_gymnasium.spaces import map_to_space")

def create_init_files():
    """Crée les fichiers __init__.py nécessaires"""
    init_files = [
        "src/gama_gymnasium/__init__.py",
        "src/gama_gymnasium/core/__init__.py",
        "src/gama_gymnasium/spaces/__init__.py", 
        "src/gama_gymnasium/wrappers/__init__.py",
        "src/gama_gymnasium/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created {init_file}")

def main():
    """Fonction principale de migration"""
    print("🚀 Début de la migration vers la nouvelle architecture...")
    
    # Sauvegarder l'ancienne structure
    if not Path("backup").exists():
        print("💾 Création d'une sauvegarde...")
        shutil.copytree(".", "backup", ignore=shutil.ignore_patterns("backup"))
    
    # Exécuter les étapes de migration
    create_directory_structure()
    migrate_python_package() 
    migrate_examples()
    migrate_gama_models()
    create_init_files()
    update_imports()
    
    print("\n✨ Migration terminée!")
    print("\n📋 Prochaines étapes manuelles:")
    print("1. Vérifiez les imports dans tous les fichiers")
    print("2. Mettez à jour pyproject.toml avec la nouvelle configuration")
    print("3. Exécutez les tests pour vérifier que tout fonctionne")
    print("4. Supprimez les anciens répertoires si tout est OK")
    print("\n💡 Utilisez 'pip install -e .' pour installer en mode développement")

if __name__ == "__main__":
    main()
