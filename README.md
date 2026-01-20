# Desafio MBA Engenharia de Software com IA - Full Cycle

## Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate
```

## Ordem de execução

Subir o banco de dados:

```
docker compose up -d
```

## Executar ingestão do PDF:

```
python src/ingest.py
```

## Rodar o chat:

```
python src/chat.py
```
