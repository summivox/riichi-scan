#!/usr/bin/env bash

# NOTE: use default user `vagrant` to run this script


########################################
# package manager
sudo pacman -Sy --noconfirm
sudo pacman -Syu --noconfirm
echo "\
    TMPDIR=$HOME/tmp/yaourt\
    BUILD_NOCONFIRM=1\
    EDITFILES=0\
" > "~/.yaourtrc"
mkdir -p ~/tmp/yaourt


########################################
# utils
#sudo pacman -S --noconfirm wget net-tools


########################################
# libs
sudo pacman -S --noconfirm lapack gcc-fortran cmake


########################################
# opencv
sudo pacman -S --noconfirm opencv
sudo ln /dev/null /dev/raw1394 # http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394


########################################
## git
#sudo pacman -S --noconfirm git
#git config --global user.name blxlrsmb
#git config --global user.email blxlrsmb


########################################
# python and packages
yaourt -S --noconfirm anaconda2 # includes almost everything...
sudo pip2 install theano


########################################
# lua/torch
#   TODO: CUDA support
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
pushd ~/torch; ./install.sh; popd


########################################
# jupyter kernels
git clone https://github.com/facebook/iTorch.git ~/iTorch
pushd ~/iTorch; luarocks make; popd
