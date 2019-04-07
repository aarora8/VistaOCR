export PATH=$PWD/parallel/:$PATH
export PATH=~/SCTK/:$PATH
# Here is a shell snippet that removes duplicates from $PATH. 
# Ref: http://linuxg.net/oneliners-for-removing-the-duplicates-in-your-path/
export PATH=`echo -n $PATH | awk -v RS=: -v ORS=: '{ if (!arr[$0]++) { print $0 } }'`
