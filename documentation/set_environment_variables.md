# How to set environment variables

nnU-Net requires some environment variables so that it always knows where the raw data, preprocessed data and trained 
models are. Depending on the operating system, these environment variables need to be set in different ways.

Variables can either be set permanently (recommended!) or you can decide to set them every time you call nnU-Net. 

# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export LongiSeg_raw="/path/to/LongiSeg_raw"
export LongiSeg_preprocessed="/path/to/LongiSeg_preprocessed"
export LongiSeg_results="/path/to/LongiSeg_results"
```

(Of course you need to adapt the paths to the actual folders you intend to use).
If you are using a different shell, such as zsh, you will need to find the correct script for it. For zsh this is `.zshrc`.

## Temporary
Just execute the following lines whenever you run nnU-Net:
```bash
export LongiSeg_raw="/path/to/LongiSeg_raw"
export LongiSeg_preprocessed="/path/to/LongiSeg_preprocessed"
export LongiSeg_results="/path/to/LongiSeg_results"
```
(Of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your terminal! They will also only apply to the current 
terminal window and DO NOT transfer to other terminals!

Alternatively you can also just prefix them to your nnU-Net commands:

`LongiSeg_results="/path/to/LongiSeg_results" LongiSeg_preprocessed="/path/to/LongiSeg_preprocessed" nnUNetv2_train[...]`

## Verify that environment parameters are set
You can always execute `echo ${LongiSeg_raw}` etc to print the environment variables. This will return an empty string if 
they were not set.

# Windows
Useful links:
- [https://www3.ntu.edu.sg](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=To%20set%20(or%20change)%20a,it%20to%20an%20empty%20string.)
- [https://phoenixnap.com](https://phoenixnap.com/kb/windows-set-environment-variable)

## Permanent
See `Set Environment Variable in Windows via GUI` [here](https://phoenixnap.com/kb/windows-set-environment-variable). 
Or read about setx (command prompt).

## Temporary
Just execute the following before you run nnU-Net:

(PowerShell)
```PowerShell
$Env:LongiSeg_raw = "C:/path/to/LongiSeg_raw"
$Env:LongiSeg_preprocessed = "C:/path/to/LongiSeg_preprocessed"
$Env:LongiSeg_results = "C:/path/to/LongiSeg_results"
```

(Command Prompt)
```Command Prompt
set LongiSeg_raw=C:/path/to/LongiSeg_raw
set LongiSeg_preprocessed=C:/path/to/LongiSeg_preprocessed
set LongiSeg_results=C:/path/to/LongiSeg_results
```

(Of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your session! They will also only apply to the current 
window and DO NOT transfer to other sessions!

## Verify that environment parameters are set
Printing in Windows works differently depending on the environment you are in:

PowerShell: `echo $Env:[variable_name]`

Command Prompt: `echo %[variable_name]%`
