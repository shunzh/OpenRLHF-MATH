DIR_NAME="$(basename $(pwd))"

rsync -av --exclude=".git" . satori2:/nobackup/users/shunzh/$DIR_NAME
