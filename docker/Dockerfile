FROM ubuntu:18.04

# Change the source of apt
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
WORKDIR /etc/apt/
RUN echo "">sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse">>sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse">>sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse">>sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse">>sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse">>sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse">>sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse">>sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse">>sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse">>sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse">>sources.list

# Install python
RUN apt update && apt install -y python3 python3-pip iproute2 net-tools  \
    wget vim nload iftop libgmp-dev libmpfr-dev libmpc-dev --fix-missing

# link the python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip -i https://pypi.doubanio.com/simple

# Install python requirements (using douban image)
RUN pip install setuptools matplotlib PyYAML pympler \
                numpy keras sklearn scipy pillow psutil phe gmpy2 \
                -i https://pypi.doubanio.com/simple

WORKDIR /root/