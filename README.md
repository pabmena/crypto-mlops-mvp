# Crypto MLOps MVP — Infra mínima viva

Este paquete es un arranque **mínimo** listo para correr en Docker + FastAPI y con `Makefile` de ayuda.

## Requisitos
- Docker Desktop actualizado
- (Opcional) `make` en PATH

## Pasos rápidos

### 1) Descomprimir
- **Windows (Explorador)**: clic derecho → *Extract All...* → elegí destino (recomendado `C:\Dev`).
- **PowerShell**:
  ```powershell
  cd C:\Dev
  Expand-Archive -Path .\crypto-mlops-mvp-initial.zip -DestinationPath . -Force
  ```
- **Git Bash**:
  ```bash
  cd /c/Dev
  unzip ~/Downloads/crypto-mlops-mvp-initial.zip
  ```

### 2) Preparar `.env`
- **PowerShell**
  ```powershell
  cd C:\Dev\crypto-mlops-mvp-initial
  Copy-Item .env.example .env
  ```
- **Git Bash**
  ```bash
  cd /c/Dev/crypto-mlops-mvp-initial
  cp .env.example .env
  ```

### 3) Levantar
- Con **make**
  ```bash
  make up
  ```
- Sin make
  ```bash
  docker compose up -d --build
  ```

### 4) Probar
- FastAPI docs: http://localhost:8800/docs
- Health: http://localhost:8800/health

_Paquete generado:_ 2025-08-17T04:03:09.760975Z
