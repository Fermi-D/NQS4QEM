# latest datascince-notebook image
FROM jupyter/datascience-notebook:latest

# name
MAINTAINER fermid

# add package
# add python package
RUN pip install qutip
RUN pip install qucat
RUN pip install qucumber
RUN pip install qiskit
RUN pip install qulacs
RUN pip install optuna

#RUN apt-get update \
    #&& apt-get install -y git \
    #&& apt-get install -y make \
    #&& apt-get install -y curl \
    #&& apt-get install -y xz-utils \
    #&& apt-get install -y file \
    #&& apt-get install -y sudo


#RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    #&& cd mecab-ipadic-neologd \
    #&& bin/install-mecab-ipadic-neologd -n -y


# add julia package
#RUN julia -e 'import Pkg; Pkg.update()' && \
    #julia -e 'import Pkg; Pkg.add("Plots")'

# jupyter extension
#RUN jupyter labextension install @lckr/jupyterlab_variableinspector