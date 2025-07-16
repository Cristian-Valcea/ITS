@echo off
REM Log tail script for IntradayJules training logs
echo 🔍 Starting log monitoring...
powershell -ExecutionPolicy Bypass -File "%~dp0log-tail.ps1" %*