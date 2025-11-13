# Add Python to PATH for Current Session (Temporary)
# $env:Path += ";C:\Users\Michel\AppData\Local\Programs\Python\Python311;C:\Users\Michel\AppData\Local\Programs\Python\Python311\Scripts"

# For Current User:
# $pythonPath = "C:\Users\Michel\AppData\Local\Programs\Python\Python313"
# $scriptPath = "$pythonPath\Scripts"
# [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$pythonPath;$scriptPath", "User")

# For All Users (requires admin rights):
$pythonPath = "C:\Users\Michel\AppData\Local\Programs\Python\Python313"
$scriptPath = "$pythonPath\Scripts"
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$pythonPath;$scriptPath", "Machine")

python --version
pip --version
