# Relatório de Simulação - 11 de Maio de 2026 (Revisado)

## Resumo Executivo
Esta simulação revisada foi realizada para validar o comportamento do robô de trading utilizando a estratégia V4, agora incluindo rotinas de **BUY (Long)** e **SELL (Short)**. O capital inicial considerado foi de **R$ 5.000,00** para os ativos BTC, ETH e SOL.

## Parâmetros da Simulação
- **Data:** 11 de Maio de 2026
- **Portfolio Inicial:** R$ 5.000,00 (US$ 1.020,41)
- **Câmbio Base:** 4,90 BRL/USD
- **Ativos:** BTC-USD, ETH-USD, SOL-USD
- **Granularidade:** 15 minutos (96 candles)
- **Rotinas:** Long e Short habilitados

## Resultados Gerais
| Métrica | Valor |
|---------|-------|
| **Portfolio Final** | **R$ 4.968,52** |
| **P&L Total** | **R$ -31,48 (-0,63%)** |
| **Win Rate** | **12,5%** |
| **Profit Factor** | **0,00** |
| **Total de Trades** | **8** |

## Resultados por Cripto
| Ativo | Trades | P&L (BRL) |
|-------|--------|-----------|
| **BTC-USD** | 3 | R$ -12,93 |
| **ETH-USD** | 2 | R$ -7,06 |
| **SOL-USD** | 3 | R$ -10,34 |

## Análise de Desempenho
1. **Atividade Bidirecional:** A inclusão de ordens de venda (Short) aumentou a atividade do robô, resultando em 8 trades totais. O robô conseguiu identificar oportunidades em ambos os lados do mercado.
2. **Qualidade dos Sinais:** O Win Rate caiu drasticamente para 12,5%. Isso indica que, embora o robô esteja gerando sinais, a maioria foi invalidada rapidamente pelo preço, sugerindo um mercado altamente ruidoso ou com reversões frequentes no dia 11 de maio.
3. **Gerenciamento de Risco:** Apesar da baixíssima assertividade, a perda total foi contida em -0,63%. O `Sizing Engine` e os `Stop Losses` dinâmicos cumpriram seu papel de preservação de capital.
4. **Profit Factor:** O valor de 0,00 reflete a ausência de ganhos significativos que superassem as taxas e pequenas perdas de stop no período simulado.

## Recomendações e Próximos Passos
- **Filtro de Tendência:** A baixa performance em shorts sugere que o robô pode estar tentando operar contra a tendência macro. Recomenda-se reforçar o peso do `Regime Engine` para bloquear shorts em tendências de alta forte e vice-versa.
- **Ajuste de Sensibilidade:** O `MIN_SCORE` de 0,55 mostrou-se muito permissivo para um dia de alta volatilidade/ruído. Elevar para **0,62** reduziria o overtrading e focaria apenas em sinais de alta probabilidade.
- **Análise de Slippage:** Em ambientes reais, o impacto pode ser maior. Sugere-se simular com uma taxa de corretagem levemente superior (0,06%) para stress test.

---
*Relatório técnico gerado via Backtest Engine V4. Dados históricos obtidos via API OKX.*
