# Versionamento CCTB V4

## Formato: vMAJOR.STRATEGY.BUILD

Exemplo: **v4.12.387**

### Componentes

#### 1. **MAJOR** (versão da arquitetura)
Mudanças estruturais e paradigmáticas do robot.

Incrementa **apenas** quando houver:
- Reescrita estrutural significativa
- Mudança de engine de execução
- Novo sistema de risco central
- Mudança de exchange/core
- Mudança de paradigma operacional

**Histórico:**
- **v1**: MVP inicial
- **v2**: Multi-asset
- **v3**: IA + filtros avançados
- **v4**: Arquitetura institucional (atual)
- **v5**: Multi-exchange distribuído (futuro)

#### 2. **STRATEGY** (evolução quantitativa)
Progresso no motor operacional dentro da arquitetura atual.

Incrementa quando houver:
- Nova estratégia/signal adicionada
- Melhoria relevante de P&L
- Alteração do risk model
- Mudança em entry/exit logic
- Novos filtros/gates
- Ajuste material de parâmetros

**Exemplo:** v4.13.xxx significa a 13ª evolução quantitativa da arquitetura V4.

#### 3. **BUILD** (número de commits)
Automático — representa o total de commits do git.

**Propriedades:**
- Nunca repete (count monotônico)
- Rastreável direto no git
- Conecta com CI/CD
- Indica maturidade do projeto

**Exemplo:** v4.12.387 = 387 commits totais no repositório

---

## Uso

### Arquivo VERSION
Localizado na raiz do projeto: `VERSION`

```
4.12
```

### Obter versão completa

**Shell:**
```bash
scripts/get_version.sh
# Output: v4.12.387
```

**Python:**
```python
import subprocess
import os

def get_full_version():
    with open('VERSION') as f:
        major_strategy = f.read().strip()
    
    build = subprocess.run(
        ['git', 'rev-list', '--all', '--count'],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    return f"v{major_strategy}.{build}"
```

### API Endpoint

```
GET /api/version
```

Response:
```json
{
  "version": "v4.12.387"
}
```

---

## Atualizar STRATEGY

Quando atingir um milestone estratégico (nova feature, melhoria de risco, etc):

```bash
# Editar VERSION
echo "4.13" > VERSION

# Commit
git add VERSION
git commit -m "bump: strategy version 4.12 → 4.13 (nova estratégia XYZ)"
git push
```

O BUILD incrementará automaticamente com cada commit.

---

## Dashboard Display

O header do dashboard exibe a versão completa:
- **Mobile**: `CCTB v4.12.387`
- **Desktop**: `Claude Code Trading Bot v4.12.387`

Atualizado dinamicamente via `/api/version` no carregamento da página.
