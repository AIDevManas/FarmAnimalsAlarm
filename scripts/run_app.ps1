# PowerShell helper to run the v2 GUI
$script = Join-Path -Path $PSScriptRoot -ChildPath "..\app\v2\final_app.py"
python $script
