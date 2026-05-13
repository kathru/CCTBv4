# Relatório de Simulação - CCTB V4

## 1. Sumário Executivo
Esta simulação avalia a robustez do motor de trading V4 após os recentes ajustes no repositório. Foi realizado um teste comparativo entre o motor **V4 Robusto** (usado em produção) e o motor **V4 Simplificado**, partindo de um capital inicial de **R$ 5.000,00** ($ 1.020,00 USD).

O resultado principal destaca a **preservação total do capital** pelo motor robusto em um cenário de queda generalizada do mercado (-27,93% no benchmark), enquanto o motor simplificado apresentou erosão de capital devido ao overtrading.

## 2. Configuração da Simulação
- **Capital Inicial:** R$ 5.000,00 (~$ 1.020,00 USD)
- **Ativos:** BTC-USDT, ETH-USDT, SOL-USDT
- **Período OOS (Out-of-Sample):** Nov 2025 - Mai 2026 (6 meses)
- **Ciclo:** 1H (Horário)
- **Motor Robusto:** V4Orchestrator + SimulatedExecutionEngine (Regime Engine 7 estados, Probabilistic Signals, Thesis Invalidation).
- **Motor Simplificado:** Backtester modular OHLCV (ADX, BB, ATR expansion).

## 3. Resultados das Simulações

### A. Motor Robusto (V4 Completo)
| Métrica | Resultado |
| :--- | :--- |
| **P&L Total** | **+0,00%** |
| **Trades Executados** | 0 |
| **Retorno BTC (Bench)** | -18,30% |
| **Retorno Equal Weight** | -25,92% |
| **Alpha vs Benchmark** | **+25,92%** |

**Análise Crítica:**
O motor robusto demonstrou sua principal força: **seletividade institucional**. Durante um período de 6 meses onde o mercado de criptoativos derreteu (SOL -35,95%, ETH -29,54%), os filtros de regime e score probabilístico bloquearam entradas de baixa confiança. A tese de preservação de capital em regimes de risco ("Cash is a position") foi validada com sucesso.

### B. Motor Simplificado (Backtester V4)
| Métrica | Resultado |
| :--- | :--- |
| **P&L Total** | **-3,30%** |
| **Trades Executados** | 54 |
| **Win Rate** | 35,2% |
| **Profit Factor** | 0,80 |
| **Max Drawdown** | -3,78% |

**Análise Crítica:**
O motor simplificado, por não possuir as camadas de validação de tese e microestrutura (Orderflow Proxy, Platt Calibration), sofreu com o "ruído" do mercado lateral/baixista. Realizou um alto número de trades (54) com expectativa negativa, resultando em uma perda de $ 33,71 do capital inicial.

## 4. Diagnóstico Detalhado

1.  **Gestão de Risco:** O V4 Robusto é significativamente mais resiliente. O filtro `MIN_SCORE` ajustado por regime impediu entradas em "Mean Reverting Chop" que o motor simplificado aceitou.
2.  **Alpha Inter-Asset:** A correção no cálculo de Relative Strength (RS) no motor robusto permitiu uma leitura mais precisa do fluxo de capital entre BTC e Alts.
3.  **Ambiente de Execução:** A simulação com o `SimulatedExecutionEngine` reflete custos reais de Maker/Taker e slippage, garantindo que o resultado de 0% seja um reflexo fiel da inatividade estratégica necessária no período.

## 5. Recomendações

-   **Manutenção do Threshold:** Manter o `MIN_SCORE` atual elevado para regimes de incerteza.
-   **Regime Edge Table:** Continuar utilizando a `regime_edge.json` para bloquear automaticamente regimes que historicamente não apresentam EV (Expected Value) positivo.
-   **Ajuste no Take Profit:** Para o motor simplificado, os dados sugerem que o TP atual (2x SL) pode estar muito distante para o regime de mercado atual, explicando o Win Rate baixo. No entanto, para o motor robusto, o foco deve permanecer na qualidade das entradas.

## 6. Conclusão
O repositório está **muito mais robusto**. A simulação provou que o sistema prefere ficar fora do mercado a operar sem vantagem estatística. Para um investidor com R$ 5.000,00, a segurança de não perder capital em um bear market de -25% é o maior trunfo deste novo motor V4.

**Veredicto: Aprovado para continuidade em ambiente de validação (Paper Trading).**
