@echo off
echo ================================================
echo  Instalando Trading Bot como Servico Windows
echo ================================================

set PYTHON=C:\Users\chris\AppData\Local\Programs\Python\Python312\python.exe
set BOT_DIR=D:\Claude Code Trading bot
set SERVICE=TradingBot

:: Remove servico anterior se existir
nssm stop %SERVICE% 2>nul
nssm remove %SERVICE% confirm 2>nul

:: Instala o dashboard como servico
nssm install %SERVICE% "%PYTHON%" "run_dashboard.py"
nssm set %SERVICE% AppDirectory "%BOT_DIR%"
nssm set %SERVICE% AppStdout "%BOT_DIR%\logs\service_stdout.log"
nssm set %SERVICE% AppStderr "%BOT_DIR%\logs\service_stderr.log"
nssm set %SERVICE% AppRotateFiles 1
nssm set %SERVICE% AppRotateOnline 1
nssm set %SERVICE% AppRotateBytes 10485760
nssm set %SERVICE% Start SERVICE_AUTO_START
nssm set %SERVICE% AppRestartDelay 5000
nssm set %SERVICE% Description "Claude Code Trading Bot - Coinbase Paper Trading"

:: Inicia o servico
nssm start %SERVICE%

echo.
echo Servico "%SERVICE%" instalado e iniciado!
echo Dashboard disponivel em: http://localhost:8000
echo.
echo Comandos uteis:
echo   nssm start TradingBot
echo   nssm stop TradingBot
echo   nssm restart TradingBot
echo   nssm status TradingBot
pause
