# How to set environment variables

nnU-Net requires some environment variables so that it always knows where the raw data, preprocessed data and trained 
models are. Depending on the operating system, these environment variables need to be set in different ways.

Variables can either be set permanently (recommended!) or you can decide to set them everytime you call nnU-Net. 

# Linux & MacOS

## Permanent
Locate the `.bashrc` file in your home folder and add the following lines to the bottom:

```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```

(of course you need to adapt the paths to the actual folders you intend to use).
If you are using a different shell, such as zsh, you will need to find the correct script for it. For zsh this is `.zshrc`.

## Temporary
Just execute the following lines whenever you run nnU-Net:
```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```
(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your terminal! They will also only apply to the current 
terminal window and DO NOT transfer to other terminals!

Alternatively you can also just prefix them to your nnU-Net commands:

`nnUNet_results="/media/fabian/nnUNet_results" nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed" nnUNetv2_train[...]`

## Verify that environment parameters are set
You can always execute `echo ${nnUNet_raw}` etc to print the environment variables. This will return an empty string if 
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

(powershell)
```powershell
$Env:nnUNet_raw = "C:/Users/fabian/nnUNet_raw"
$Env:nnUNet_preprocessed = "C:/Users/fabian/nnUNet_preprocessed"
$Env:nnUNet_results = "C:/Users/fabian/fabian/nnUNet_results"
```

(command prompt)
```commandline
set nnUNet_raw=C:/Users/fabian/nnUNet_raw
set nnUNet_preprocessed=C:/Users/fabian/nnUNet_preprocessed
set nnUNet_results=C:/Users/fabian/fabian/nnUNet_results
```

(of course you need to adapt the paths to the actual folders you intend to use).

Important: These variables will be deleted if you close your session! They will also only apply to the current 
window and DO NOT transfer to other sessions!

## Verify that environment parameters are set
Printing in Windows works differently depending on the environment you are in:

powershell: `echo $Env:[variable_name]`

command prompt: `echo %[variable_name]%`
