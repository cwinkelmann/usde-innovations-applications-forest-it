```shell




sudo mkdir -p /private/nfs/cwinkelmann
sudo mount -t nfs -o vers=3,rsize=1048576,wsize=1048576,async,noatime,tcp 192.168.188.190:/volume1/storage /private/nfs/cwinkelmann

```



```

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/christian/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/christian/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/christian/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/christian/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# The following lines have been added by Docker Desktop to enable Docker CLI completions.
fpath=(/Users/christian/.docker/completions $fpath)
autoload -Uz compinit
compinit
# End of Docker CLI completions

# Added by Hugging Face CLI installer
export PATH="/Users/christian/.local/bin:$PATH"

export PATH=$PATH:/Users/christian/.local/bin

alias mountraid='mkdir -p ~/raid/cwinkelmann && sudo mount -t nfs -o vers=3,rsize=1048576,wsize=1048576,async,noatime,tcp 192.168.188.190:/volume1/storage ~/raid/cwinkelmann'
alias umountraid='sudo umount ~/raid/cwinkelmann'
```