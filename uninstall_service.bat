@echo off
echo Removendo servico TradingBot...
nssm stop TradingBot
nssm remove TradingBot confirm
echo Servico removido.
pause
