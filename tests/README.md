# Guide de Test pour GAMA-Gymnasium

Ce document décrit la structure de test et les meilleures pratiques pour tester le package GAMA-Gymnasium.

## Structure des Tests

```
tests/
├── conftest.py                 # Configuration pytest et fixtures communes
├── unit/                       # Tests unitaires
│   ├── test_client.py         # Tests du client GAMA
│   ├── test_message_handler.py # Tests du gestionnaire de messages
│   ├── test_space_converters.py # Tests des convertisseurs d'espaces
│   ├── test_space_validators.py # Tests des validateurs d'espaces
│   ├── test_sync_wrapper.py   # Tests du wrapper synchrone
│   └── test_monitoring_wrapper.py # Tests du wrapper de monitoring
├── integration/                # Tests d'intégration
│   └── test_full_integration.py # Tests du workflow complet
└── performance/                # Tests de performance
    └── test_performance.py    # Tests de performance et benchmarks
```

## Types de Tests

### Tests Unitaires

Les tests unitaires testent des composants individuels en isolation :

- **Client GAMA** (`test_client.py`) : Test de la communication avec GAMA
- **Gestionnaire de Messages** (`test_message_handler.py`) : Validation et formatage des messages
- **Convertisseurs d'Espaces** (`test_space_converters.py`) : Conversion entre espaces GAMA et Gymnasium
- **Validateurs d'Espaces** (`test_space_validators.py`) : Validation des espaces et actions
- **Wrapper Synchrone** (`test_sync_wrapper.py`) : Conversion async/sync
- **Wrapper de Monitoring** (`test_monitoring_wrapper.py`) : Suivi et logging

### Tests d'Intégration

Les tests d'intégration vérifient que les composants fonctionnent ensemble :

- Workflow complet GAMA-Gymnasium
- Intégration des wrappers
- Propagation des erreurs
- Gestion des types d'actions complexes

### Tests de Performance

Les tests de performance mesurent les performances et identifient les goulots d'étranglement :

- Vitesse de création d'environnement
- Performance des steps
- Performance des resets
- Overhead des wrappers
- Tests de charge avec environnements concurrents

## Exécution des Tests

### Commandes de Base

```bash
# Tous les tests
python run_tests.py all

# Tests unitaires seulement
python run_tests.py unit

# Tests d'intégration
python run_tests.py integration

# Tests de performance
python run_tests.py performance

# Tests rapides (exclut les tests lents)
python run_tests.py fast
```

### Avec Make (Windows compatible)

```bash
# Configuration de l'environnement de développement
make dev-setup

# Tous les tests
make test

# Tests unitaires
make test-unit

# Tests d'intégration
make test-integration

# Tests de performance
make test-performance

# Tests rapides
make test-fast
```

### Options Avancées

```bash
# Tests avec couverture de code
make coverage

# Tests de benchmark
make benchmark

# Vérification des dépendances
make deps
```

## Qualité du Code

### Linting et Formatage

```bash
# Vérifier le formatage du code
make check-format

# Formater le code automatiquement
make format

# Linting complet
make lint

# Vérification de type
make type-check
```

### Pipeline Pre-commit

```bash
# Vérifications avant commit
make pre-commit
```

## Configuration Pytest

Le fichier `pyproject.toml` contient la configuration pytest :

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "--strict-markers", 
    "--strict-config",
    "--cov=gama_gymnasium",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov"
]
testpaths = ["tests"]
markers = [
    "integration: tests d'intégration",
    "performance: tests de performance",
    "benchmark: tests de benchmark", 
    "slow: tests lents"
]
```

## Markers Pytest

Utilisez les markers pour organiser les tests :

```python
@pytest.mark.integration
def test_full_workflow():
    """Test d'intégration du workflow complet."""
    pass

@pytest.mark.performance
def test_step_performance():
    """Test de performance des steps."""
    pass

@pytest.mark.slow
def test_memory_usage():
    """Test de l'utilisation mémoire (lent)."""
    pass
```

### Exécution Sélective

```bash
# Exécuter seulement les tests d'intégration
pytest -m integration

# Exclure les tests lents
pytest -m "not slow"

# Exécuter les tests de performance
pytest -m performance
```

## Fixtures Communes

Le fichier `conftest.py` fournit des fixtures réutilisables :

### Fixtures de Données

```python
@pytest.fixture
def mock_gama_server_responses():
    """Réponses mock standard du serveur GAMA."""
    return {...}

@pytest.fixture
def sample_space_definitions():
    """Définitions d'espaces pour les tests."""
    return {...}
```

### Fixtures d'Environnement

```python
@pytest.fixture
def temp_gaml_file(tmp_path_factory):
    """Fichier GAML temporaire pour les tests."""
    return "path/to/temp/file.gaml"
```

## Mocking et Tests

### Mock du Client GAMA

```python
@patch('gama_gymnasium.core.client.GamaSyncClient')
def test_with_mock_client(mock_sync_client):
    mock_client = mock_sync_client.return_value
    mock_client.load.return_value = {"type": "CommandExecutedSuccessfully", "content": "exp_123"}
    # ... test logic
```

### Mock des Opérations Async

```python
@pytest.mark.asyncio
async def test_async_operation():
    with patch('module.async_function', new_callable=AsyncMock) as mock_func:
        mock_func.return_value = "result"
        result = await some_async_function()
        assert result == "expected"
```

## Rapports de Couverture

Les rapports de couverture sont générés dans plusieurs formats :

- **Terminal** : Affichage direct des statistiques
- **HTML** : Rapport détaillé dans `htmlcov/index.html`
- **XML** : Format machine dans `coverage.xml`

```bash
# Générer un rapport de couverture
make coverage

# Ouvrir le rapport HTML
start htmlcov/index.html  # Windows
```

## Benchmarking

Les tests de performance incluent des benchmarks mesurables :

```python
@pytest.mark.benchmark
def test_performance_benchmark(benchmark):
    """Test de benchmark pour mesurer les performances."""
    result = benchmark(function_to_test, arg1, arg2)
    return result
```

## Debugging des Tests

### Verbose Output

```bash
# Tests avec sortie détaillée
pytest -v tests/

# Afficher les print statements
pytest -s tests/
```

### Tests Individuels

```bash
# Exécuter un test spécifique
pytest tests/unit/test_client.py::test_connection

# Exécuter une classe de tests
pytest tests/unit/test_client.py::TestGamaClient
```

### Debugging avec Breakpoints

```python
def test_debug_example():
    """Test avec point d'arrêt pour debugging."""
    result = some_function()
    import pdb; pdb.set_trace()  # Point d'arrêt
    assert result == expected
```

## Bonnes Pratiques

### 1. Tests Isolés
- Chaque test doit être indépendant
- Utiliser des mocks pour les dépendances externes
- Nettoyer après chaque test

### 2. Nommage Clair
```python
def test_gama_client_connection_success():
    """Le client GAMA doit se connecter avec succès."""
    pass

def test_gama_client_connection_failure_with_invalid_host():
    """Le client GAMA doit échouer avec un hôte invalide."""
    pass
```

### 3. Structure AAA
```python
def test_example():
    # Arrange
    env = GamaEnv("test.gaml", "test_experiment")
    
    # Act
    obs, info = env.reset()
    
    # Assert
    assert obs is not None
    assert isinstance(info, dict)
```

### 4. Tests Paramétrés
```python
@pytest.mark.parametrize("space_def,expected_type", [
    ({"type": "Discrete", "n": 5}, gymnasium.spaces.Discrete),
    ({"type": "Box", "low": [0], "high": [1], "shape": [1]}, gymnasium.spaces.Box),
])
def test_space_conversion(space_def, expected_type):
    space = map_to_space(space_def)
    assert isinstance(space, expected_type)
```

## Intégration Continue

Le pipeline CI doit exécuter :

1. **Vérification des dépendances** : `make deps`
2. **Linting** : `make lint`
3. **Tests** : `make test`
4. **Couverture** : `make coverage`

```bash
# Pipeline CI complet
make ci
```

## Maintenance des Tests

### Mise à Jour des Mocks
- Maintenir les mocks synchronisés avec l'API GAMA réelle
- Tester périodiquement avec de vraies instances GAMA

### Performance Monitoring
- Surveiller les métriques de performance
- Alerter en cas de régression

### Documentation
- Maintenir la documentation des tests à jour
- Documenter les cas d'usage complexes
