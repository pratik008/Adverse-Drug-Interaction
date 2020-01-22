FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
#RUN conda create -n DrIP python
#ENV PATH /opt/conda/envs/DrIP/bin:$PATH
#RUN /bin/bash -c "source activate DrIP"

RUN echo "source activate DrIP" > ~/.bashrc
ENV PATH /opt/conda/envs/DrIP/bin:$PATH

ADD download_model.sh /

RUN bash download_model.sh
